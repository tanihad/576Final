import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder

from dataset import TacoDataset
from utils import generate_proposals, match_proposals_with_ground_truth, extract_roi, preprocess_image
from model import build_r_cnn_model

def main():
    dataset_dir = "../trashpics/"
    annotations_file = "../trashpics/annotations.json"

    print("Starting main process...")

    # Load the dataset and annotations
    dataset = TacoDataset(dataset_dir, annotations_file)

    rois, labels, bboxes = [], [], []
    label_encoder = LabelEncoder()

    for image_info in dataset.image_info:
        image_path = image_info['file_name']
        if not os.path.exists(image_path):
            print(f"File does not exist: {image_path}")
            continue

        image = cv2.imread(image_path)
        if image is None:
            print(f"Error loading image {image_path}")
            continue

        boxes, class_ids = dataset.load_mask(image_info['image_id'])

        # Generate region proposals
        proposals = generate_proposals(image)

        # Match proposals with ground truth boxes
        labeled_proposals = match_proposals_with_ground_truth(proposals, zip(boxes, class_ids))

        for proposal, label, gt_bbox in labeled_proposals:
            roi = extract_roi(image, proposal)  # Extract region from image using proposal (bounding box)
            roi = preprocess_image(roi)  # Preprocess the extracted ROI
            rois.append(roi)
            labels.append(label)
            bboxes.append(gt_bbox)

    # Encode labels
    labels = label_encoder.fit_transform(labels)

    print("Building the model...")
    model = build_r_cnn_model(num_classes=len(set(labels)))

    print("Starting model training...")
    model.fit(np.array(rois), [np.array(labels), np.array(bboxes)], epochs=10, batch_size=32, validation_split=0.1)

    print("Training completed.")

if __name__ == "__main__":
    main()
