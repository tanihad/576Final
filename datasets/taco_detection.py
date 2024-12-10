import os
import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image, ExifTags
from torchvision import transforms

from torch.utils.data.dataset import Dataset
from pycocotools.coco import COCO
from .helpers import convert_to_coco_format

class TACODetection(Dataset):
    def __init__(self, root, imset='train', mode='fast'):
        self.root = root
       
        self.annotation_file = os.path.join(root, f'{imset}_annotations.json')
        self.proposals_dir = os.path.join(self.root, f'object_proposals_{mode}', imset)
        self.proposals_labels_dir = os.path.join(self.root, f'object_proposals_{mode}', f'{imset}_labels')
        self.proposals_gt_refinments_dir = os.path.join(self.root, f'object_proposals_{mode}', f'{imset}_gt_refinements')
        
        self.imgs_dir = root

        # Initialize the COCO api for instance annotations
        self.taco=COCO(self.annotation_file)

        # Load the categories in a variable
        self.catIds = self.taco.getCatIds()
        self.cats = self.taco.loadCats(self.catIds)

        self.img_ids = list(sorted(self.taco.imgs.keys()))

        if mode == 'fast':
            self.im_transform = transforms.Compose([
                transforms.Resize((800,600), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
            ])
        elif mode == 'quality':
            self.im_transform = transforms.Compose([
                transforms.Resize((600, 400), interpolation=transforms.InterpolationMode.BICUBIC),     
                transforms.ToTensor(),
            ])

       

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
        
        orig_img_size = torch.tensor([img.size[0], img.size[1]], dtype=torch.float32)
        img = self.im_transform(img)
        target_img_size = torch.tensor([img.shape[-1], img.shape[-2]], dtype=torch.float32)

        ######## load proposals
        obj_proposals = np.load(os.path.join(self.proposals_dir,f'object_proposals_img_{img_id}.npy'))
        proposal_labels = np.load(os.path.join(self.proposals_labels_dir, f'labels_proposals_img_{img_id}.npy'))
        proposal_labels = torch.from_numpy(proposal_labels).t()

        proposal_refinments = np.load(os.path.join(self.proposals_gt_refinments_dir, f'gt_refinements_img_{img_id}.npy'))
        proposal_refinments = torch.from_numpy(proposal_refinments)

        # convert to coco format
        proposals = []
        for prop in obj_proposals:
            proposals.append(list(convert_to_coco_format(*prop)))
            
        obj_proposals = torch.tensor(proposals)

        ####### log gt annotations
        gt_boxes = []
        gt_labels = []
        for annot in coco_annotation:
            gt_boxes.append(annot['bbox'])
            gt_labels.append(annot['category_id'])
            assert annot['category_id'] < 28
            
        gt_boxes = torch.tensor(gt_boxes)
        gt_labels = torch.tensor(gt_labels)

        # rescale gt boxes
        if target_img_size[0] != orig_img_size[0] or target_img_size[1] != orig_img_size[1]:
            scale_factors = target_img_size / orig_img_size
            for ii in range(len(gt_boxes)):
                bbox = gt_boxes[ii]
                x, y, w, h = bbox
                x_scaled = x * scale_factors[0]
                y_scaled = y * scale_factors[1]
                w_scaled = w * scale_factors[0]
                h_scaled = h * scale_factors[1]

                # Update the bounding box coordinates
                bbox_scaled = [x_scaled, y_scaled, w_scaled, h_scaled]

                gt_boxes[ii] = torch.tensor(bbox_scaled)


        data = {
            'img': img,
            'img_id': img_id,
            'num_objs': num_objs,
            'obj_proposals': obj_proposals,
            'proposal_labels': proposal_labels,
            'gt_labels': gt_labels,
            'gt_boxes': gt_boxes,
            'proposal_refinments': proposal_refinments
        }

        return data
    
        