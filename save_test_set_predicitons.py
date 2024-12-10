import os
from tqdm import tqdm

import pickle

# pytorch imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.ops import boxes as box_ops

from model.rcnn import RCNN
from datasets.helpers import *
from datasets import TACODetection
from util.mypath import Path
from util.util import extract_rois, nms, refine_bbox
"""Argument Loading""" 
regr_head = True

if regr_head:
    model_path = os.path.join(Path.models_dir('detector'), 'rcnn_regr_head.pth')
else:
    model_path = os.path.join(Path.models_dir('detector'), 'rcnn.pth')

"""Paths"""
db_root = Path.db_root_dir('TACO')


device='cuda' if torch.cuda.is_available() else 'cpu'

"""Model"""
model = RCNN(arch='resnet18', regr_head=regr_head)
model.load_state_dict(torch.load(model_path))
model.eval()
model.to(device)

softmax_fn = nn.Softmax(dim=1)

"""Data loader"""
test_db = TACODetection(db_root, imset='test', mode='fast')
test_loader = DataLoader(test_db, batch_size=1, shuffle=False)

all_categories = []
pred_data = {}

for data in tqdm(test_loader, total=len(test_loader), desc=f'Running on test set'):
    img = data['img'][0]
    obj_proposals = data['obj_proposals'][0]
    img_id = data['img_id'].item()
    gt_labels = data['gt_labels'][0].to(device)
    gt_boxes = data['gt_boxes'][0]

    
    all_categories.extend(gt_labels.tolist())

    bboxes = box_ops.box_convert(obj_proposals, 'xywh', 'xyxy')
    prop_imgs = extract_rois(img, bboxes)
    tensor_dataset = TensorDataset(prop_imgs, obj_proposals)
    db_loader = DataLoader(tensor_dataset, batch_size=64)
    
    # store not background predictions
    predictions = {
        'scores': [],
        'labels': [],
        'bboxes': []
    }
    
    for proposals, prop_boxes in db_loader:
        proposals = proposals.to(device)
        with torch.no_grad():
            if regr_head:
                y, y_regr = model(proposals)
            else:
                y = model(proposals)
            y = softmax_fn(y)
            pred_scores, pred_labels = y.max(dim=1)
            i = 0
            for score, label, bbox in zip(pred_scores, pred_labels, prop_boxes):
                
                if label.item() != BACKGROUND_LABEL:
                    predictions['scores'].append(score.item())
                    predictions['labels'].append(label.item())
                    if regr_head:
                        bbox = refine_bbox(bbox, y_regr[i]).unsqueeze(0)
                    else:
                        bbox = bbox.unsqueeze(0)
                    predictions['bboxes'].append(bbox)

                i+= 1
    
    n = len(predictions['scores'])
    if n == 0:
        continue
    
    predictions['scores'] = torch.tensor(predictions['scores'])
    predictions['labels'] = torch.tensor(predictions['labels'])
    predictions['bboxes'] = torch.cat(predictions['bboxes'], dim=0)
    
    selected_indices = nms(predictions=predictions, iou_threshold=0.2)

    predictions['scores'] = predictions['scores'][selected_indices]
    predictions['labels'] = predictions['labels'][selected_indices]
    predictions['bboxes'] = predictions['bboxes'][selected_indices]
    pred_data[img_id] = {
        'gt_boxes': gt_boxes,
        'gt_labels': gt_labels,
        'predictions': predictions
    }
    

if regr_head:
    with open('save/pred_data_regr_head.pickle', 'wb') as handle:
        pickle.dump(pred_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    all_categories = set(all_categories)
    with open('save/all_categories_regr_head.pickle', 'wb') as handle:
        pickle.dump(all_categories, handle, protocol=pickle.HIGHEST_PROTOCOL)

else :
    with open('save/pred_data.pickle', 'wb') as handle:
        pickle.dump(pred_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    all_categories = set(all_categories)
    with open('save/all_categories.pickle', 'wb') as handle:
        pickle.dump(all_categories, handle, protocol=pickle.HIGHEST_PROTOCOL)

