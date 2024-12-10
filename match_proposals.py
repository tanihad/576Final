import os
import argparse
from datasets.helpers import *
from tqdm import tqdm

from torch.utils.data import DataLoader
from datasets import TACOProposals
from util.mypath import Path

from util.preprocess_taco import match_proposals_with_gt_boxes

"""Load arguments"""
parser = argparse.ArgumentParser()
parser.add_argument('--imset', type=str, default='val')
parser.add_argument('--mode', type=str, default='fast')

args = parser.parse_args()
imset = args.imset
mode = args.mode

assert imset in {'train', 'val', 'test'}
assert mode in {'fast', 'quality'}

"""Paths"""
db_root = Path.db_root_dir('TACO')
save_path_labels = os.path.join(db_root, f'object_proposals_{mode}', f'{imset}_labels')
save_path_gt_refinements = os.path.join(db_root, f'object_proposals_{mode}', f'{imset}_gt_refinements' )
os.makedirs(save_path_labels, exist_ok=True)
os.makedirs(save_path_gt_refinements, exist_ok=True)

db = TACOProposals(db_root, imset=imset, mode=mode)
db_loader = DataLoader(db, batch_size=1, shuffle=False)

non_found_props = 0
for data in tqdm(db_loader, total=len(db_loader), desc=f'Matching proposals {imset} for {mode}'):
    
    gt_boxes = data['gt_boxes'][0]
    labels = data['labels'][0].tolist()
  
    img_id = data['img_id'].item()
    num_objs = data['num_objs']
    obj_proposals = data['obj_proposals'][0] # obj_proposals.size() = [5814,4]

    matched = match_proposals_with_gt_boxes(proposals=obj_proposals, gt_boxes=gt_boxes, labels=labels)
    breakpoint()
    proposal_labels = []
    gt_refinements = []
    for prop_id in matched.keys():
        prop_data = matched[prop_id]
        label = prop_data['label']
        proposal_labels.append(label)
        gt_refinement = prop_data['gt_refinement']
        gt_refinements.append(gt_refinement)
    
    if len(np.unique(proposal_labels)) < 3:
        non_found_props += 1

    np.save(os.path.join(save_path_labels, f'labels_proposals_img_{img_id}.npy'), np.array(proposal_labels))
    np.save(os.path.join(save_path_gt_refinements, f'gt_refinements_img_{img_id}.npy'), np.array(gt_refinements))
    

print(f'Imgs with no props: {non_found_props}')