import torch
import argparse
import torch.nn as nn
import numpy as np
import yaml
import random
from DeTR import DeTr
from DeTR import compute_sample_loss
from DeTR import custom_collate_fn
from tqdm import tqdm
from Dataloader import VOCDataset
from torch.utils.data.dataloader import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(args):
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config) 

    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']

    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    batch_size = train_config['batch_size']
    niters = train_config['niters']
    print_every_n = train_config['print_every_n']
    save_every_n = train_config['save_every_n']

    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)

    voc = VOCDataset('train',
                     im_dir=dataset_config['im_train_path'],
                     ann_dir=dataset_config['ann_train_path'])
    train_dataset = DataLoader(voc,
                               batch_size=batch_size,
                               shuffle=True,
                               num_workers=4,
                               collate_fn=custom_collate_fn)
    
    Detr = DeTr(model_config=model_config)
    Detr.train()
    Detr.cuda()

    transformer_params = [
    p for n, p in Detr.named_parameters() if 'backbone.' not in n]
    backbone_params = [
    p for n, p in Detr.named_parameters() if 'backbone.' in n]
    
    nbackparams = sum([p.nelement() for p in backbone_params]) / 1e6
    ntransparams = sum([p.nelement() for p in transformer_params]) / 1e6
    totalparams = nbackparams + ntransparams
    
    print(f'Backbone params: {nbackparams:.1f}M')
    print(f'Transformer params: {ntransparams:.1f}M')
    print(f'Total params: {totalparams:.1f}M')

    if model_config['train_backbone']:
        optimizer = torch.optim.AdamW([
            {'params': transformer_params, 'lr': 1e-4},
            {'params': backbone_params, 'lr': 1e-5},
        ], weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW([
            {'params': transformer_params, 'lr': 1e-4},]
            , weight_decay=1e-4)

    losses=[]
    hist=[]
    iters=1
    while iters <= niters:
        for im, target in tqdm(train_dataset):
            im = im.cuda()
            output = Detr(im)

            loss = torch.Tensor([0]).cuda()
            for batch_idx in range(batch_size):
                target[batch_idx]['bboxes'] = target[batch_idx]['bboxes'].float()
                target[batch_idx]['labels'] = target[batch_idx]['labels'].long()

                loss_class, loss_bbox, loss_giou = compute_sample_loss(output['pred_boxes'][batch_idx], target[batch_idx]['bboxes'], 
                                                                       output['pred_logits'][batch_idx], target[batch_idx]['labels'], 
                                                                       model_config["num_queries"], model_config["num_classes"])
                sample_loss = 1*loss_class + 5*loss_bbox + 2*loss_giou
                loss = (loss + sample_loss)/batch_size
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(Detr.parameters(), .1)
            optimizer.step()
            
            losses.append(loss.item())
            if iters % print_every_n == 0:
                loss_avg = np.mean(losses[-10:])
                print_text = f'iters: {iters},\tloss: {loss_avg:.4f}'
                print(print_text)
                print(f'loss_class: {loss_class.item():.4f}\tloss_bbox: {loss_bbox.item():.4f}\tloss_giou: {loss_giou.item():.4f}\t')
                
                hist.append(loss_avg)
                losses = []
            
            if iters % save_every_n == 0 and iters > 0:
                str_iters = str(iters)
                str_iters = '0'*(6-len(str_iters)) + str_iters
                torch.save(Detr.state_dict(), f'ckpts/model_it{str_iters}.pt')

            iters += 1
            if iters > niters:
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for Detection Transformer training')
    parser.add_argument('--config', dest='config_path',
                        default='D:\ObjDet\DeTR\config.yaml', type=str)
    args = parser.parse_args()
    train(args)


