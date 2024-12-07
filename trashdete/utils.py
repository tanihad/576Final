import cv2
import numpy as np
import time
import selective_search

def generate_proposals(image):
    """Function to generate region proposals using Selective Search"""
    start_time = time.time()  # Track time taken by this function
    print("Generating proposals...")

    # Convert image to RGB as required by the selective search
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Use selective search to generate proposals
    ss = selective_search.selective_search(image_rgb)

    proposals = []
    for rect in ss:
        # rect is a tuple (x, y, w, h), not a dictionary
        proposals.append(rect)  # Directly append the tuple

    print(f"Proposals generated in {time.time() - start_time:.2f} seconds.")
    return proposals

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    intersection_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - intersection_area

    return intersection_area / union_area

def match_proposals_with_ground_truth(proposals, ground_truth, iou_threshold=0.5):
    """Assign labels to proposals based on IoU with ground truth bounding boxes."""
    labeled_proposals = []

    for proposal in proposals:
        max_iou = 0
        matched_label = "background"
        matched_bbox = None
        for gt_bbox, gt_label in ground_truth:
            iou = calculate_iou(proposal, gt_bbox)
            if iou > max_iou:
                max_iou = iou
                matched_label = gt_label
                matched_bbox = gt_bbox

        if max_iou >= iou_threshold:
            labeled_proposals.append((proposal, matched_label, matched_bbox))
        else:
            labeled_proposals.append((proposal, "background", None))

    return labeled_proposals

def extract_roi(image, rect):
    """Extracts a Region of Interest (ROI) from the image based on the provided bounding box."""
    x, y, w, h = rect
    roi = image[y:y + h, x:x + w]  # Crop the image using the bounding box coordinates
    return roi

def preprocess_image(roi):
    """Preprocess the extracted ROI (e.g., resize, normalization, etc.)"""
    # Example of preprocessing: Resize to (224, 224) and normalize the pixel values
    roi = cv2.resize(roi, (224, 224))
    roi = roi.astype('float32')
    roi /= 255.0  # Normalize the pixel values to [0, 1]
    return roi
