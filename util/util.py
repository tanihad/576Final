import torch
from torchvision import transforms
from torchvision.ops import boxes as box_ops

from .preprocess_taco import calculate_iou

# Define the transformation to resize the images to 224x224
roi_transform = transforms.Compose([
    transforms.Resize((224, 224), antialias=True, interpolation=transforms.InterpolationMode.BICUBIC)
])

def extract_rois(img, boxes):
    resized_images = []
    for box in boxes:
        try :
            # Extract the region of interest (ROI) from the image
            roi = transforms.functional.crop(img, box[1], box[0], box[3]-box[1], box[2]-box[0])
            #Resize the ROI to 224x224
            resized_roi = roi_transform(roi)
        except RuntimeError:
            continue
        
        # Add the resized ROI to the list of resized images
        resized_images.append(resized_roi)
    
    resized_images = torch.stack(resized_images)
    return resized_images


def nms(predictions, iou_threshold):
    scores = predictions['scores']
    labels = predictions['labels']
    bboxes = predictions['bboxes']

    # Sort scores in descending order
    sorted_indices = torch.argsort(scores, descending=True)

    selected_indices = []
    while sorted_indices.numel() > 0:
        # Pick the box with the highest score
        best_index = sorted_indices[0].item()
        selected_indices.append(best_index)

        # Compute IOU between the best box and the rest of the boxes
        best_bbox = bboxes[best_index]
        other_bboxes = bboxes[sorted_indices[1:]]
        ious = []
        for bbox in other_bboxes:
            iou_ = calculate_iou(bbox, best_bbox)
            ious.append(iou_)
        ious = torch.tensor(ious)

        # Get indices of bboxes with IOU less than threshold
        mask = ious <= iou_threshold
        sorted_indices = sorted_indices[1:][mask]
    
    # Return the selected indices as torch tensor
    selected_indices = torch.tensor(selected_indices)

    return selected_indices


def refine_bbox(bbox, pred):
    # Refine bbox in COCO format of [x, y, width, height]
    # bbox is a list of [x, y, width, height]
    # pred is an array of predicted refinements of shape (batch size, 4)
    x, y, w, h = bbox

    newx = x + w * pred[0]
    newy = y + h * pred[1]
    neww = w * torch.exp(pred[2])
    newh = h * torch.exp(pred[3])

    return torch.tensor([newx, newy, neww, newh])


def sample_proposals(img, obj_proposals, proposal_labels, proposal_refinments, 
        gt_boxes, gt_labels, gt_refinments, back_label):
    
    none_idxs = torch.where(proposal_labels == -1)
    nones = obj_proposals[none_idxs]
    n_nones = nones.shape[0]

    # Find background proposals with label 28
    back_idxs = torch.where(proposal_labels == back_label)
    background_proposals = obj_proposals[back_idxs]
    background_refinments = proposal_refinments[back_idxs]

    n_background = background_proposals.shape[0]
    n_not_background = proposal_labels.shape[0] + gt_boxes.shape[0] - n_background - n_nones
    backgroud_labels = proposal_labels[back_idxs][:n_not_background] # a tensor with only BACKGROUND_LABEL

    # sample background proposals
    perm = torch.randperm(n_background)[:n_not_background]
    background_proposals = background_proposals[perm]
    background_refinments = background_refinments[perm]
    # no need to shuffle the background labels

    # get not background proposals
    not_background_idxs = torch.where((proposal_labels != back_label) & (proposal_labels != -1))
    obj_proposals = obj_proposals[not_background_idxs]
    proposal_labels = proposal_labels[not_background_idxs]
    prop_refinments = proposal_refinments[not_background_idxs]

    n_selective_search = proposal_labels.shape[0]

    # Extract the bounding box coordinates
    background_boxes = box_ops.box_convert(background_proposals, 'xywh', 'xyxy')
    if n_selective_search != 0: # rarely it happens
        obj_boxes = box_ops.box_convert(obj_proposals, 'xywh', 'xyxy')
    gt_boxes = box_ops.box_convert(gt_boxes, 'xywh', 'xyxy')

    background_imgs = extract_rois(img, background_boxes)
    if n_selective_search != 0:
        obj_imgs = extract_rois(img, obj_boxes)
    gt_imgs = extract_rois(img, gt_boxes)

    # merge and shuffle background proposals
    if n_selective_search != 0:
        proposals = torch.cat((background_imgs, obj_imgs, gt_imgs), dim=0)
        labels = torch.cat((backgroud_labels, proposal_labels, gt_labels))
        refinments = torch.cat((background_refinments, prop_refinments, gt_refinments), dim=0)
    else :
        proposals = torch.cat((background_imgs, gt_imgs), dim=0)
        labels = torch.cat((backgroud_labels, gt_labels))
        refinments = torch.cat((background_refinments, gt_refinments), dim=0)

    n = proposals.shape[0]
    
    perm = torch.randperm(n)
    proposals = proposals[perm]
    labels = labels[perm]
    refinments = refinments[perm]

    return proposals, labels, refinments