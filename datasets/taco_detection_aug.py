import os
import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image, ExifTags
from torchvision import transforms

from torch.utils.data.dataset import Dataset
from pycocotools.coco import COCO
from .helpers import convert_to_coco_format


class TACODetectionAUG(Dataset):
    def __init__(self, root, aug_transform, imset='train', mode='fast'):
        self.root = root
       
        self.annotation_file = os.path.join(root, f'{imset}_annotations.json')
        self.proposals_dir = os.path.join(self.root, f'object_proposals_{mode}', imset)
        self.proposals_labels_dir = os.path.join(self.root, f'object_proposals_{mode}', f'{imset}_labels')
        self.imgs_dir = root

        # Initialize the COCO api for instance annotations
        self.taco=COCO(self.annotation_file)

        # Load the categories in a variable
        self.catIds = self.taco.getCatIds()
        self.cats = self.taco.loadCats(self.catIds)

        self.img_ids = list(sorted(self.taco.imgs.keys()))

        self.im_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.mode = mode
        self.aug_transform = aug_transform


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
        if self.mode == 'fast':
            img = img.resize((600,800))
        else :
            img = img.resize((400,600))
        target_img_size = torch.tensor([img.size[0], img.size[1]], dtype=torch.float32)
        # PIL to cv2
        img = np.array(img) 
        img = img[:, :, ::-1].copy() 

        # load proposals
        obj_proposals = np.load(os.path.join(self.proposals_dir,f'object_proposals_img_{img_id}.npy'))
        proposal_labels = np.load(os.path.join(self.proposals_labels_dir, f'labels_proposals_img_{img_id}.npy'))
        proposal_labels = proposal_labels.T
        
        # convert to coco format
        proposals = []
        for prop in obj_proposals:
            proposals.append(list(convert_to_coco_format(*prop)))
            
        obj_proposals = np.array(proposals)
        
        ####### log gt annotations
        gt_boxes = []
        gt_labels = []
        for annot in coco_annotation:
            gt_boxes.append(annot['bbox'])
            gt_labels.append(annot['category_id'])
            assert annot['category_id'] < 28
            
        gt_boxes = np.array(gt_boxes)
        gt_labels = np.array(gt_labels)

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

                gt_boxes[ii] = np.array(bbox_scaled)

        gt_boxes = gt_boxes.astype(int)
        
        # cat gt to proposals
        if obj_proposals.shape[0] != 0:
            obj_proposals = np.concatenate((obj_proposals, gt_boxes), axis=0)
            proposal_labels = np.concatenate((proposal_labels, gt_labels), axis=0)
        
        else :
            obj_proposals = gt_boxes
            proposal_labels = gt_labels
        
        # Data augmentation with ambulation library
        if self.aug_transform is not None:
            transformed = self.aug_transform(image=img, bboxes=obj_proposals, category_id=proposal_labels)
            img = transformed['image']
            obj_proposals = transformed['bboxes']
            proposal_labels = transformed['category_id']

        img = self.im_transform(img)
        obj_proposals = torch.from_numpy(np.array(obj_proposals)).to(torch.int)
        proposal_labels = torch.from_numpy(np.array(proposal_labels))

        if torch.where(proposal_labels != 28)[0].sum() == 0:
            print('hi')

        data = {
            'img': img,
            'img_id': img_id,
            'num_objs': num_objs,
            'obj_proposals': obj_proposals,
            'proposal_labels': proposal_labels
        }

        return data
    
        