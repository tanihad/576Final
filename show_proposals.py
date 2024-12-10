import os

import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm

# pytorch imports
import torch
from torch.utils.data import DataLoader

# custom imports
from datasets.helpers import *
from datasets import TACODetection
from util.mypath import Path
from vis import bar_plot
from util.util import refine_bbox, extract_rois
from torchvision.ops import boxes as box_ops


"""Load arguments"""
parser = argparse.ArgumentParser()
parser.add_argument('--imset', type=str, default='val')
args = parser.parse_args()
imset = args.imset

assert imset in {'train', 'val', 'test'}

"""Paths"""
db_root = Path.db_root_dir('TACO')
save_path = os.path.join(db_root, f'object_proposals_fast', f'{imset}_labels')
os.makedirs(save_path, exist_ok=True)

db = TACODetection(db_root, imset=imset)
db_loader = DataLoader(db, batch_size=1, shuffle=True)


only_back = 0
for data in tqdm(db_loader, total=len(db_loader), desc=f'Show {imset}'):
    
    img = data['img'][0]
    img_id = data['img_id'].item()
    num_objs = data['num_objs']
    obj_proposals = data['obj_proposals'][0]
    proposal_labels = data['proposal_labels'][0]
    proposal_refinments = data['proposal_refinments'][0]
    
    # Find background proposals with label 28
    background_proposals = obj_proposals[torch.where(proposal_labels == 28)]
    n_background = background_proposals.shape[0]
    n_not_background = proposal_labels.shape[0] - n_background

    # sample background proposals
    perm = torch.randperm(n_background)[:n_not_background]
    background_proposals = background_proposals[perm]

    # get not background proposals
    non_back = torch.where( (proposal_labels != 28) & (proposal_labels != -1))
    obj_proposals = obj_proposals[non_back]
    proposal_refinments = proposal_refinments[non_back]
    

    if n_not_background == 0:
        only_back += 1
    fig,ax = plt.subplots(1)
    plt.axis('off')
    plt.imshow(im_normalize(tens2image(img)))
    # Show annotations
    for ann in obj_proposals:
        #ref = refine_bbox(ann, pred_ref)
        [x, y, w, h] = [tensor.item() for tensor in ann]
        rect = Rectangle((x,y),w,h,linewidth=2,edgecolor='red',
                         facecolor='none', alpha=0.7, linestyle = '--')
        ax.add_patch(rect)
    
    cnt = 0
    # Show annotations
    for ann in background_proposals:
        
        [x, y, w, h] = [tensor.item() for tensor in ann]
        rect = Rectangle((x,y),w,h,linewidth=2,edgecolor='blue',
                         facecolor='none', alpha=0.7, linestyle = '--')
        ax.add_patch(rect)
        cnt+= 1

        if cnt == 10:
            break
    plt.savefig(f'./assets/propsals_{img_id}.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    fig,ax = plt.subplots(1)
    plt.axis('off')
    plt.imshow(im_normalize(tens2image(img)))
    plt.savefig(f'./assets/{img_id}.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    obj_boxes =  box_ops.box_convert(obj_proposals, 'xywh', 'xyxy')
    obj_imgs = extract_rois(img, obj_boxes)
    i = 0
    for img in obj_imgs:
        fig,ax = plt.subplots(1)
        plt.axis('off')
        plt.imshow(im_normalize(tens2image(img)))
        plt.savefig(f'./assets/roi_{i}.png', bbox_inches='tight', pad_inches=0)
        plt.close()
        i += 1
  
    break



    


    