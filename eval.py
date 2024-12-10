import pickle
import numpy as np
import json
import torch
from util.util import calculate_iou

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
from sklearn.metrics import auc
from util.mypath import Path

regr_head = True
if regr_head:
    ap_path = 'ap_regr'
    with open('save/all_categories_regr_head.pickle', 'rb') as handle:
        all_categories = list(pickle.load(handle))

    with open('save/pred_data_regr_head.pickle', 'rb') as handle:
        pred_data = pickle.load(handle)
else:
    ap_path = 'ap'
    with open('save/all_categories.pickle', 'rb') as handle:
        all_categories = list(pickle.load(handle))

    with open('save/pred_data.pickle', 'rb') as handle:
        pred_data = pickle.load(handle)



dataset_path = Path.db_root_dir('TACO')
anns_file_path = dataset_path + '/' + 'annotations.json'

# Read annotations
with open(anns_file_path, 'r') as f:
    dataset = json.loads(f.read())

categories = dataset['categories']
anns = dataset['annotations']
imgs = dataset['images']
nr_cats = len(categories)
nr_annotations = len(anns)
nr_images = len(imgs)

# Load categories and super categories
super_cat_names = []
super_cat_ids = {}
super_cat_last_name = ''
nr_super_cats = 0
for cat_it in categories:
    super_cat_name = cat_it['supercategory']
    # Adding new supercat
    if super_cat_name != super_cat_last_name:
        super_cat_names.append(super_cat_name)
        super_cat_ids[super_cat_name] = nr_super_cats
        super_cat_last_name = super_cat_name
        nr_super_cats += 1


"""pred_data structure
pred_data = {
    1 :{ # img_id of first image
        'gt_boxes': gt_boxes,
        'gt_labels': gt_labels,
        'predictions': {
            'scores': Nx1,
            'labels': Nx1,
            'bboxes': Nx4
        }
    }
    2 :{ # img_id of second image
        'gt_boxes': gt_boxes,
        'gt_labels': gt_labels,
        'predictions': {
            'scores': Nx1,
            'labels': Nx1,
            'bboxes': Nx4
        }
    }
}
"""

def get_max_iou(gt_boxes, pred_box):
    max_iou = -1
    matched_box = -1
    for ii, bbox in enumerate(gt_boxes):
        bbox = torch.tensor(bbox)
        iou = calculate_iou(bbox, pred_box)
        if iou>max_iou:
            max_iou = iou
            matched_box = ii
    
    return max_iou, matched_box

def compute_ap(iou_thresh=0.5, save_figures=False):
    # Initialization
    all_precisions = []
    all_recalls = []

    for cat_id in all_categories:
        
        tp = 0  # true positive count
        fp = 0  # false positive count
        fn = 0  # false negative count

       
        for im_id in pred_data.keys():
            image_predictions = pred_data[im_id]
            gt_labels = image_predictions['gt_labels'].tolist()
            gt_boxes = image_predictions['gt_boxes'].tolist()

            pred_labels = image_predictions['predictions']['labels']
            scores = image_predictions['predictions']['scores']
            bboxes = image_predictions['predictions']['bboxes']
            if cat_id in gt_labels:
                # match to gt boxes
                idxs = torch.argsort(scores, descending=True)
                bboxes = bboxes[idxs]

                # match to gt boxes
                for bbox, pred_label in zip(bboxes, pred_labels):
                    
                    if pred_label.item() != cat_id:
                        fn+=1 # false negative
                    else :
                        max_iou, matched_box = get_max_iou(gt_boxes, bbox)
                        if max_iou > iou_thresh:
                            tp += 1 # mark as postive
                            gt_boxes.pop(matched_box) # remove the gt bbox
                        
                        else:
                            # Otherwise, it's a false positive
                            fp += 1
                            

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0

        if save_figures:
            cat_name = super_cat_names[cat_id]
            plt.figure(figsize=(7.4,4.8))
            plt.scatter(all_recalls, all_precisions)
            plt.xlabel('Recall', fontweight='bold')
            plt.ylabel('Precision', fontweight='bold')
            plt.title(f'Precision-Recall Curve for {cat_name}', fontweight='bold')
            plt.grid(True)
            plt.savefig(f'assets/{ap_path}/plots/pr_{cat_name}.png', bbox_inches='tight')
            plt.close()
        all_precisions.append(precision)
        all_recalls.append(recall)
    

    # Compute AP
    sorted_indices = np.argsort(all_recalls)
    sorted_precisions = [all_precisions[i] for i in sorted_indices]
    sorted_recalls = [all_recalls[i] for i in sorted_indices]
    ap = auc(sorted_recalls, sorted_precisions)

    print(f'Average Precision: {ap:.3f} at IoU: {iou_thresh}')

    with open(f'assets/{ap_path}/ap_{iou_thresh:.2f}.txt', 'w') as f:
        f.write(f'Average Precision: {ap:.3f} at IoU: {iou_thresh:.2f}')

    return ap


                    
iou_thresholds = np.arange(0.2, 0.9, 0.05)
average_precisions = []

for iou_thresh in iou_thresholds:
    ap = compute_ap(iou_thresh)
    average_precisions.append(ap) 

# Plot the average precision curve
plt.figure()
plt.plot(iou_thresholds, average_precisions, marker='o')
plt.xlabel('IoU Threshold', fontweight='bold')
plt.ylabel('Average Precision', fontweight='bold')
plt.title('Average Precision vs. IoU Threshold', fontweight='bold')
plt.grid(True)
plt.savefig(f'assets/{ap_path}/map.png', bbox_inches='tight')
plt.close()

with open(f'assets/{ap_path}/map.txt', 'w') as f:
    f.write(f'Mean Average Precision: {np.mean(average_precisions):.3f}')

print(f'Mean Average Precision: {np.mean(average_precisions):.3f}')