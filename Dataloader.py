import glob
import os
import random
import torch
import torchvision
from PIL import Image
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
import xml.etree.ElementTree as ET
import torch.nn.functional as F

def load_images_and_anns(im_dir, ann_dir, label2idx):
    r"""
    Method to get the xml files and for each file
    get all the objects and their ground truth detection
    information for the dataset
    :param im_dir: Path of the images
    :param ann_dir: Path of annotation xmlfiles
    :param label2idx: Class Name to index mapping for dataset
    :return:
    """
    im_infos = []
    for ann_file in tqdm(glob.glob(os.path.join(ann_dir, '*.xml'))):
        im_info = {}
        im_info['img_id'] = os.path.basename(ann_file).split('.xml')[0]
        im_info['filename'] = os.path.join(im_dir, '{}.jpg'.format(im_info['img_id']))
        ann_info = ET.parse(ann_file)
        root = ann_info.getroot()
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        im_info['width'] = width
        im_info['height'] = height
        detections = []
        
        for obj in ann_info.findall('object'):
            det = {}
            label = label2idx[obj.find('name').text]
            bbox_info = obj.find('bndbox')
            bbox = [
                int(float(bbox_info.find('xmin').text))-1,
                int(float(bbox_info.find('ymin').text))-1,
                int(float(bbox_info.find('xmax').text))-1,
                int(float(bbox_info.find('ymax').text))-1
            ]
            det['label'] = label
            det['bbox'] = bbox
            detections.append(det)
        im_info['detections'] = detections
        im_infos.append(im_info)
    print('Total {} images found'.format(len(im_infos)))
    return im_infos

class VOCDataset(Dataset):
    def __init__(self, split, im_dir, ann_dir):
        self.split = split
        self.im_dir = im_dir
        self.ann_dir = ann_dir
        self.min_size = 600
        self.max_size = 1000
        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]
        classes = [
            'person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
            'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
            'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor'
        ]
        classes = sorted(classes)
        classes = ['background'] + classes
        self.label2idx = {classes[idx]: idx for idx in range(len(classes))}
        self.idx2label = {idx: classes[idx] for idx in range(len(classes))}
        print(self.idx2label)
        self.images_info = load_images_and_anns(im_dir, ann_dir, self.label2idx)
    
    def __len__(self):
        return len(self.images_info)
    
    def tensor_to_numpy(tensor):
        return tensor.detach().cpu().numpy()
    
    def normalize_resize_image_and_boxes(self, image, bboxes):
        dtype, device = image.dtype, image.device

        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        image = (image - mean[:, None, None]) / std[:, None, None]
        
        h, w = image.shape[-2:]
        min_size, max_size = min(h, w), max(h, w)

        scale = min(self.max_size / max_size, self.min_size / min_size)
        new_min_size = int(round(min_size * scale))
        new_max_size = int(round(max_size * scale))
        
        if new_min_size < 480:
            scale = 480 / min_size
        elif new_min_size > 800:
            scale = 800 / min_size
        
        if new_max_size > 1333:
            scale = min(scale, 1333 / max_size)
        
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))

        image = torch.nn.functional.interpolate(
            image,
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        )
        if bboxes is not None:
            ratio_height = new_h / h
            ratio_width = new_w / w
            xmin, ymin, xmax, ymax = bboxes.unbind(2)
            xmin = xmin * ratio_width
            xmax = xmax * ratio_width
            ymin = ymin * ratio_height
            ymax = ymax * ratio_height
            bboxes = torch.stack((xmin, ymin, xmax, ymax), dim=2) 
        
        return image, bboxes
    
    def random_crop(self, image, target, crop_size=(800, 800)):
        orig_width, orig_height = image.shape[-2:]
        crop_w, crop_h = crop_size

        if orig_width <= crop_w or orig_height <= crop_h:
            return image, target  

        x_min = random.randint(0, orig_width - crop_w)
        y_min = random.randint(0, orig_height - crop_h)
        x_max = x_min + crop_w
        y_max = y_min + crop_h

        cropped_image = image.crop((x_min, y_min, x_max, y_max))

        bboxes = target['boxes'].clone()  
        labels = target['labels']

        bboxes[:, [0, 2]] *= orig_width  
        bboxes[:, [1, 3]] *= orig_height  

        bboxes[:, [0, 2]] -= x_min  
        bboxes[:, [1, 3]] -= y_min  
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]].clamp(0, crop_w)
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]].clamp(0, crop_h)

        keep = (bboxes[:, 2] - bboxes[:, 0] > 1) & (bboxes[:, 3] - bboxes[:, 1] > 1)
        bboxes, labels = bboxes[keep], labels[keep]
        new_target = {'boxes': bboxes, 'labels': labels}

        return cropped_image, new_target

    def __getitem__(self, index):
        im_info = self.images_info[index]
        im = Image.open(im_info['filename'])
        targets = {}
        targets['bboxes'] = torch.as_tensor([detection['bbox'] for detection in im_info['detections']])
        targets['labels'] = torch.as_tensor([detection['label'] for detection in im_info['detections']])
        im_tensor = torchvision.transforms.ToTensor()(im)
        old_shape = im_tensor.shape[-2:]
        to_random_crop = False

        # Normalize and resize first
        if self.split == 'train':
            to_random_crop = True
            im_tensor = im_tensor.unsqueeze(0)
            targets['bboxes'] = targets['bboxes'].unsqueeze(0)
            im_tensor, targets['bboxes'] = self.normalize_resize_image_and_boxes(im_tensor, targets['bboxes'])
            im_tensor = im_tensor.squeeze(0)
            targets['bboxes'] = targets['bboxes'].squeeze(0)
            
        # Apply Random Crop with 50% probability
        if self.split == 'train' and random.random() < 0.5:
            im_tensor, targets['bboxes'] = self.random_crop(im_tensor, targets['bboxes'])
            im_tensor = im_tensor.unsqueeze(0)
            targets['bboxes'] = targets['bboxes'].unsqueeze(0)
            im_tensor, targets['bboxes'] = self.normalize_resize_image_and_boxes(im_tensor, targets['bboxes'])
            im_tensor = im_tensor.squeeze(0)
            targets['bboxes'] = targets['bboxes'].squeeze(0)
        
        # elif self.split == "test":
        #     im_tensor = im_tensor.unsqueeze(0)
        #     im_tensor, targets['bboxes'] = self.normalize_resize_image_and_boxes(im_tensor, None)
        #     im_tensor = im_tensor.squeeze(0)
        
        if self.split == 'train':
            return im_tensor, targets
        
        elif self.split == 'test':
            return im_tensor, targets, im_info['filename'], old_shape
        