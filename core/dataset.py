from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO

class COCOPersonDataset:
    def __init__(self, root_dir='data/coco', split='train'):
        """
        Inicializa o dataset COCO focado em pessoas.
        
        Args:
            root_dir: Diretório base para o dataset
            split: 'train' ou 'val'
        """
        self.root_dir = root_dir
        self.split = split
        self.image_dir = os.path.join(root_dir, split)
        self.annot_file = os.path.join(
            root_dir, 
            'annotations',
            f'instances_{split}.json'
        )
        
    def get_person_data(self):
        """Retorna todas as imagens (já são apenas com pessoas)."""
        coco = COCO(self.annot_file)
        
        # Como já filtramos pelo FiftyOne, todas as imagens têm pessoas
        image_ids = coco.getImgIds()
        
        return {
            'coco': coco,
            'image_ids': image_ids,
            'image_dir': self.image_dir
        }

class BasePersonDataset(Dataset):
    def __init__(self, root_dir='data/coco', split='train'):
        self.root_dir = root_dir
        self.split = split
        self.image_dir = os.path.join(root_dir, split)
        self.annot_file = os.path.join(root_dir, 'annotations', f'instances_{split}.json')
        
    def _load_image(self, img_id):
        img_info = self.coco.loadImgs(img_id)[0]
        image = Image.open(os.path.join(self.image_dir, img_info['file_name']))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image, img_info

class PersonClassificationDataset(BasePersonDataset):
    def __init__(self, coco_data, transform=None):
        self.coco = coco_data['coco']
        self.image_dir = coco_data['image_dir']
        self.image_ids = coco_data['image_ids']
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Cache das labels para cada imagem
        self.labels = {}
        for img_id in self.image_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            # categoria 1 é pessoa, categoria 2 é no-person
            self.labels[img_id] = 1.0 if any(ann['category_id'] == 1 for ann in anns) else 0.0
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        image, _ = self._load_image(img_id)
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(self.labels[img_id])

class PersonDetectionDataset(BasePersonDataset):
    def __init__(self, coco_data, transform=None):
        self.coco = coco_data['coco']
        self.image_dir = coco_data['image_dir']
        self.image_ids = coco_data['image_ids']
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        image, _ = self._load_image(img_id)
        
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=[1])
        anns = self.coco.loadAnns(ann_ids)
        
        boxes = []
        for ann in anns:
            bbox = ann['bbox']
            boxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((len(boxes),), dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id]),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
        }
        
        if self.transform:
            image = self.transform(image)
        
        return image, target

class PersonSegmentationDataset(BasePersonDataset):
    def __init__(self, coco_data, transform=None, mask_transform=None):
        self.coco = coco_data['coco']
        self.image_dir = coco_data['image_dir']
        self.image_ids = coco_data['image_ids']
        self.transform = transform or transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.mask_transform = mask_transform or transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        image, img_info = self._load_image(img_id)
        
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=[1])
        anns = self.coco.loadAnns(ann_ids)
        
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        for ann in anns:
            if 'segmentation' in ann:
                if isinstance(ann['segmentation'], dict):
                    mask_ann = mask_utils.decode(ann['segmentation'])
                else:
                    mask_ann = self.coco.annToMask(ann)
                mask = np.maximum(mask, mask_ann)
        
        mask = Image.fromarray(mask)
        
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
            mask = mask.squeeze(0)
        
        return image, mask.long()