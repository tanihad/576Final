import os
import numpy as np
import torch

from PIL import Image, ExifTags
from torchvision import transforms

from torch.utils.data.dataset import Dataset
from pycocotools.coco import COCO

class TACO(Dataset):
    def __init__(self, root, imset='train', input_transform=None):
        self.root = root
        if imset is None:
            self.annotation_file = os.path.join(root, f'annotations.json')
        else :
            self.annotation_file = os.path.join(root, f'{imset}_annotations.json')
        self.imgs_dir = root

        # Initialize the COCO api for instance annotations
        self.taco=COCO(self.annotation_file)

        # Load the categories in a variable
        self.catIds = self.taco.getCatIds()
        self.cats = self.taco.loadCats(self.catIds)

        self.img_ids = list(sorted(self.taco.imgs.keys()))

        if input_transform is None:
            self.im_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        else :
            self.im_transform = input_transform


    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, index):
        
        # img id
        img_id = self.img_ids[index]

        # List: get annotation id from coco
        ann_ids = self.taco.getAnnIds(imgIds=img_id, catIds=[], iscrowd=None)

        # List with all annotations for each object
        coco_annotation = self.taco.loadAnns(ann_ids)
        num_objs = len(coco_annotation)

        path = self.taco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.imgs_dir, path))
        # resize the bboxes if necessary
        original_size = torch.tensor([img.size[0], img.size[1]], dtype=torch.float32)

        # Obtain Exif orientation tag code
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        
        if img._getexif():
            exif = dict(img._getexif().items())
            # Rotate portrait and upside down images if necessary
            if orientation in exif:
                if exif[orientation] == 3:
                    img = img.rotate(180,expand=True)
                if exif[orientation] == 6:
                    img = img.rotate(270,expand=True)
                if exif[orientation] == 8:
                    img = img.rotate(90,expand=True)

        img = self.im_transform(img)
        target_size = torch.tensor([img.shape[-1], img.shape[-2]], dtype=torch.float32)

        boxes = []
        labels = []
        for annot in coco_annotation:
            boxes.append(annot['bbox'])
            labels.append(annot['category_id'])

        boxes = torch.tensor(boxes)
        labels = torch.tensor(labels)
        
        # resize bbox if necessary
        if target_size[0] != original_size[0] or target_size[1] != original_size[1]:
            scale_factors = target_size / original_size
            for ii in range(len(boxes)):
                bbox = boxes[ii]
                x, y, w, h = bbox
                x_scaled = x * scale_factors[0]
                y_scaled = y * scale_factors[1]
                w_scaled = w * scale_factors[0]
                h_scaled = h * scale_factors[1]

                # Update the bounding box coordinates
                bbox_scaled = [x_scaled, y_scaled, w_scaled, h_scaled]

                boxes[ii] = torch.tensor(bbox_scaled)

        data = {
            'img': img,
            'gt_boxes': boxes,
            'img_id': img_id,
            'num_objs': num_objs,
            'labels': labels
        }
        return data
    
        