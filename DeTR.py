import torch
import torch.nn as nn
import torch.nn.functional as F
from Pos_encoder import PositionEmbeddingSine
from Backbone import BackboneNetwork
from Transformer import Transformer
from tqdm import tqdm
from torchvision import ops
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.optimize import linear_sum_assignment
from Dataloader import VOCDataset
from torch.utils.data.dataloader import DataLoader
from fvcore.nn import FlopCountAnalysis

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DeTr(nn.Module):
    def __init__(self, model_config):
        super(DeTr, self).__init__()
        self.model_config = model_config
        self.backbone = BackboneNetwork(finetune=model_config["train_backbone"])
        self.transformer = Transformer(model_config["num_layers"], model_config["d_model"], model_config["num_heads"], model_config["dropout_rate"])
        self.class_embed = nn.Linear(model_config["d_model"], model_config["num_classes"] + 1)
        self.bbox_embed = MLP(model_config["d_model"], model_config["d_model"], 4, 3)
        self.query_embed = nn.Embedding(model_config["num_queries"], model_config["d_model"])
        self.position_embedding = PositionEmbeddingSine(model_config["d_model"]//2)
        self.conv1x1 = nn.Conv2d(2048, model_config["d_model"], 1)
        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]
        self.min_size = model_config['min_im_size']
        self.max_size = model_config['max_im_size']
    
    def forward(self, x):
        B, _, _, _ = x.shape
        x = self.backbone(x)
        x = self.conv1x1(x)
        mask = torch.ones(x.shape[0], x.shape[2], x.shape[3]).bool().cuda()
        pos = self.position_embedding(x, mask)
        query_embed = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        out = self.transformer(x, query_embed, pos, mask)
        out_cls = self.class_embed(out)
        out_bbox = self.bbox_embed(out).sigmoid()
        if self.model_config["aux_loss"]:
            out= {'pred_logits': out_cls, 'pred_boxes': out_bbox} # outputs from every decoders
            return out
        else:
            out = {'pred_logits': out_cls[-1], 'pred_boxes': out_bbox[-1]} # output from last decoder
            return out
   
class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def custom_collate_fn(batch):
    """
    Pads all images to the size of the largest image in the batch and adjusts bounding boxes accordingly.
    """
    images, targets = zip(*batch)  
    max_height = max(img.shape[1] for img in images)  
    max_width = max(img.shape[2] for img in images)   

    padded_images = []
    updated_targets = []

    for img, target in zip(images, targets):
        _, h, w = img.shape
        pad_h = max_height - h
        pad_w = max_width - w

        padded_img = F.pad(img, (0, pad_w, 0, pad_h)) 
        padded_images.append(padded_img)
        updated_target = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in target.items()}

        if "boxes" in updated_target:
            bboxes = updated_target["boxes"]  # Shape: (N, 4) -> (x_min, y_min, x_max, y_max)
            if isinstance(bboxes, torch.Tensor):
                bboxes[:, [0, 2]] += pad_w  
                bboxes[:, [1, 3]] += pad_h 
                updated_target["boxes"] = bboxes

        updated_targets.append(updated_target)
    return torch.stack(padded_images), updated_targets

def compute_sample_loss(o_bbox, t_bbox, o_cl, t_cl, n_queries=100, num_classes=21):
    if len(t_cl) > 0:
        
        t_bbox = t_bbox.cuda()
        t_cl = t_cl.cuda()

        o_probs = o_cl.softmax(dim=-1)
        C_classes = -o_probs[..., t_cl]
        C_boxes = torch.cdist(o_bbox, t_bbox, p=1)

        C_giou = -ops.generalized_box_iou(
            ops.box_convert(o_bbox, in_fmt='xyxy', out_fmt='xyxy'),
            ops.box_convert(t_bbox, in_fmt='xyxy', out_fmt='xyxy')
        )
        C_total = 1*C_classes + 5*C_boxes + 2*C_giou
        C_total = C_total.cpu().detach().numpy()

        o_ixs, t_ixs = linear_sum_assignment(C_total)            

        o_ixs = torch.IntTensor(o_ixs)
        t_ixs = torch.IntTensor(t_ixs)
        o_ixs = o_ixs[t_ixs.argsort()]
        
        num_boxes = len(t_bbox)
        loss_bbox = F.l1_loss(
            o_bbox[o_ixs], t_bbox, reduce='sum') / num_boxes
        
        target_gIoU = ops.generalized_box_iou(
            ops.box_convert(o_bbox[o_ixs], in_fmt='xyxy', out_fmt='xyxy'),
            ops.box_convert(t_bbox, in_fmt='xyxy', out_fmt='xyxy')
        )
        loss_giou = 1 - torch.diag(target_gIoU).mean()

        queries_classes_label = torch.full(o_probs.shape[:1], num_classes).cuda()
        queries_classes_label[o_ixs] = t_cl
        loss_class = F.cross_entropy(o_cl, queries_classes_label)

    else:
        queries_classes_label = torch.full((n_queries,), num_classes).cuda()
        loss_class = F.cross_entropy(o_cl, queries_classes_label)
        loss_bbox = loss_giou = torch.tensor(0)
    
    return loss_class, loss_bbox, loss_giou






# backbone_params = [p for n, p in detr.named_parameters() if 'backbone.' in n]
# transformer_params = [p for n, p in detr.named_parameters() if 'backbone.' not in n]



# flops = FlopCountAnalysis(detr, x)
# flops.total()

