import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# pytorch imports
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, TensorDataset
from torchvision.ops import boxes as box_ops

# custom imports and label in gt_labels.tolist()
from model.rcnn import RCNN
from datasets.helpers import *
from datasets import TACODetection
from util.mypath import Path
from util.util import extract_rois, nms, refine_bbox, calculate_iou
"""This script is ONLY FOR VISUALIZATION, IT DOES NOT evaluate"""
"""Argument Loading""" 
regr_head = True

if regr_head:
    path_to_save = 'assets/regr/'
    model_path = os.path.join(Path.models_dir('detector'), 'rcnn_regr_head.pth')
else:
    path_to_save = 'assets/clf_only/'
    model_path = os.path.join(Path.models_dir('detector'), 'rcnn.pth')

os.makedirs(path_to_save, exist_ok=True)
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
test_db = TACODetection(db_root, imset='val', mode='fast')
test_loader = DataLoader(test_db, batch_size=1, shuffle=False)

def get_max_iou(gt_boxes, pred_box):
    max_iou = -1
    for bbox in gt_boxes:
        iou = calculate_iou(bbox, pred_box)
        if iou>max_iou and iou>0.5:
            max_iou = iou
    
    return max_iou

ii=0
for data in tqdm(test_loader, total=12, desc=f'Show best predictions'):
    img = data['img'][0]
    img_id = data['img_id']
    num_objs = data['num_objs']
    obj_proposals = data['obj_proposals'][0]
    gt_labels = data['gt_labels'][0].to(device)
    gt_boxes = data['gt_boxes'][0].to(device)

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
                max_iou = get_max_iou(gt_boxes, bbox)
                
                if label.item() != BACKGROUND_LABEL  and max_iou!=-1:
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
    # boxes = box_ops.box_convert(predictions['bboxes'], 'xywh', 'xyxy').to(torch.float)
    # selected_indices = torchvision.ops.nms(boxes=boxes, scores=predictions['scores'],iou_threshold=0.5) 
    selected_indices = nms(predictions=predictions, iou_threshold=0.2)
    n = len(selected_indices)
    
    bboxes = predictions['bboxes'][selected_indices]
    fig,ax = plt.subplots(1)
    plt.axis('off')
    plt.imshow(im_normalize(tens2image(img)))
    # Show predictions
    for ann in bboxes:
        [x, y, w, h] = [tensor.item() for tensor in ann]
        rect = Rectangle((x,y),w,h,linewidth=2,edgecolor='red',
                         facecolor='none', alpha=0.7, linestyle = '--')
        ax.add_patch(rect)

    for ann in gt_boxes:
        [x, y, w, h] = [tensor.item() for tensor in ann]
        rect = Rectangle((x,y),w,h,linewidth=2,edgecolor='blue',
                         facecolor='none', alpha=0.7, linestyle = '--')
        ax.add_patch(rect)
    
    plt.savefig(f'{path_to_save}/{img_id.item()}.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    ii+=1
    if ii == 13:
        break
    
    