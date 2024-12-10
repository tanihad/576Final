import os
import json
import torch
import random
import numpy as np
from datasets.helpers import convert_to_coco_format, BACKGROUND_LABEL
from copy import deepcopy

from .mypath import Path

def keep_supercategories(dataset):
    """Keeps only supercategories from a TACO-like dataset"""

    categories = dataset['categories']
    anns = dataset['annotations']
    imgs = dataset['images']
   
    # get super categories
    super_cat_prev_name = ''
    n_super_categories = -1
    super_categories = {}

    # dictionary that maps supercategories to categories
    super_category_to_categories = {}
    for cat_it in categories:
        super_cat_name = cat_it['supercategory']
        # Adding new supercat
        if super_cat_name != super_cat_prev_name:
            n_super_categories += 1

            super_categories[super_cat_name] = n_super_categories
            super_cat_prev_name = super_cat_name
            super_category_to_categories[n_super_categories] = [cat_it['id']]
        
        else :
            super_category_to_categories[n_super_categories].append(cat_it['id'])


    # create new categories dict
    new_categories = []
    for category_name in super_categories.keys():
        category_id = super_categories[category_name]
        new_categories.append({
            'id': category_id,
            'name': category_name
        })
    
    # change annotations categories id
    for annot in anns:
        cat_id = annot['category_id']

        # find supercategory id
        for supercat_id in super_category_to_categories.keys():
            cat_ids = super_category_to_categories[supercat_id]
            if cat_id in cat_ids:
                break
        
        annot['category_id'] = supercat_id

    new_data = {
        'annotations': anns,
        'images': imgs,
        'categories': new_categories
    }
    breakpoint()
    return new_data


def split_dataset(dataset, val_percentage=10, test_percentage=10):
    """Splits the dataset into train/val/test.
    Args :
    ------
        <dataset> : taco-like dataset
        <val_percentage>: percentage of the validation set
        <test_percentage>: percentage of the test set
    """
    imgs = dataset['images']
    anns = dataset['annotations']
    n_images = len(imgs)

    n_testing_images = int(n_images*test_percentage*0.01+0.5)
    n_nontraining_images = int(n_images*(test_percentage+val_percentage)*0.01+0.5)

    random.shuffle(imgs)
    train_set = {
        'images': [],
        'annotations': [],
        'categories': [],
    }

    train_set['categories'] = dataset['categories']

    val_set = deepcopy(train_set)
    test_set = deepcopy(train_set)

    test_set['images'] = imgs[:n_testing_images]
    val_set['images'] = imgs[n_testing_images:n_nontraining_images]
    train_set['images'] = imgs[n_nontraining_images:n_images]

    # Aux Image Ids to split annotations
    test_img_ids, val_img_ids, train_img_ids = [],[],[]
    for img in test_set['images']:
        test_img_ids.append(img['id'])

    for img in val_set['images']:
        val_img_ids.append(img['id'])

    for img in train_set['images']:
        train_img_ids.append(img['id'])
    

    # Split instance annotations
    for ann in anns:
        if ann['image_id'] in test_img_ids:
            test_set['annotations'].append(ann)
        elif ann['image_id'] in val_img_ids:
            val_set['annotations'].append(ann)
        elif ann['image_id'] in train_img_ids:
            train_set['annotations'].append(ann)
    

    root = Path.db_root_dir('TACO')
    with open(os.path.join(root, 'train_annotations.json'), 'w+') as f:
        f.write(json.dumps(train_set))

    with open(os.path.join(root, 'val_annotations.json'), 'w+') as f:
        f.write(json.dumps(val_set))

    with open(os.path.join(root, 'test_annotations.json'), 'w+') as f:
        f.write(json.dumps(test_set))


def preprocess_taco_dataset():
    """Keep only supercategories and split the dataset"""
    root = Path.db_root_dir('TACO')
    annotation_file = os.path.join(root, f'annotations.json')

    with open(annotation_file, 'r') as f:
        dataset = json.loads(f.read())
    
    dataset = keep_supercategories(dataset)

    split_dataset(dataset)



def match_proposals_with_gt_boxes(proposals, gt_boxes, labels, iou_threshold=0.7):
    """
    Match region proposals with ground truth objects based on Intersection over Union (IoU).
    """
    
    num_proposals = proposals.shape[0]  # num_proposals = 5814
    num_ground_truths = gt_boxes.shape[0] # num_ground_truths = 3
    
    # initialize the labels as -1 (Neither bacground nor object) 
    matched_data = {}
    for i in range(num_proposals):
        matched_data[i] = {'label': -1, 'iou': -1, 'gt_refinement': [0, 0, 0, 0]}

    # Compute IoU between each proposal and ground truth object
    for i in range(num_proposals):
        
        proposal = proposals[i] # proposal =  tensor([0,0,600,800])
        for j in range(num_ground_truths):
            ground_truth = gt_boxes[j]
            iou = calculate_iou(proposal, ground_truth) # iou = tensor(0.0532)

            '''
            for the first iteration: 
            gt_boxes = tensor([[284.9707,  49.1947, 107.3520, 237.7745],
                               [313.8582,   0.0000,  86.6623,  62.4695],
                               [304.8796,  48.8043,  33.1815,  22.2548]])
            '''
            
            # If IoU exceeds threshold and the ground truth object is not already matched with a higher IoU
            if iou >= iou_threshold and matched_data[i]['iou'] < iou:
                matched_data[i]['iou'] = iou 
                matched_data[i]['label'] = labels[j]

                # Calculate ground truth refinements as deltas
                d_x = (ground_truth[0] - proposal[0]) / proposal[2]   # (g_x - p_x) / p_w
                d_y = (ground_truth[1] - proposal[1]) / proposal[3]   # (g_y - p_y) / p_h
                d_w = torch.log(ground_truth[2] / proposal[2])         # torch.log(g_w / p_w)
                d_h = torch.log(ground_truth[3] / proposal[3] )        # torch.log(g_h / p_h)

                matched_data[i]['gt_refinement'] = [d_x, d_y, d_w, d_h]

            elif iou <= 0.3 and matched_data[i]['label'] == -1:
                matched_data[i]['iou'] = iou 
                matched_data[i]['label'] = BACKGROUND_LABEL
                matched_data[i]['gt_refinement'] = [0, 0, 0, 0]


    return matched_data


def calculate_iou(box1, box2):
    """Calculates IoU in COCO Format!"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calculate the coordinates of the intersection rectangle
    x_inter = max(x1, x2)
    y_inter = max(y1, y2)
    w_inter = max(0, min(x1 + w1, x2 + w2) - x_inter)
    h_inter = max(0, min(y1 + h1, y2 + h2) - y_inter)

    # Calculate the area of the intersection and union rectangles
    area_inter = w_inter * h_inter
    area_union = w1 * h1 + w2 * h2 - area_inter

    # Calculate the IoU
    iou = area_inter / area_union if area_union > 0 else 0

    return iou