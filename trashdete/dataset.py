import os
import json
import cv2
import numpy as np

class TacoDataset:
    def __init__(self, dataset_dir, annotations_file):
        self.dataset_dir = dataset_dir
        self.annotations_file = annotations_file
        self.image_info = []
        self.load_data()

    def load_data(self):
        print("Loading dataset...")
        start_time = time.time()

        try:
            with open(self.annotations_file) as f:
                annotations = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON file: {e}")
            return
        except FileNotFoundError as e:
            print(f"Error: File not found - {e}")
            return

        # Ensure 'images' is a list
        if 'images' not in annotations or not isinstance(annotations['images'], list):
            print("Error: Expected 'images' to be a list in the JSON file.")
            return

        for ann in annotations['images']:
            image_path = os.path.join(self.dataset_dir, ann['file_name'])
            image_info = {
                'image_id': ann['id'],
                'file_name': image_path,
                'annotations': []
            }

            for ann in annotations['annotations']:
                if ann['image_id'] == ann['id']:
                    image_info['annotations'].append(ann)

            self.image_info.append(image_info)

        print(f"Dataset loaded in {time.time() - start_time:.2f} seconds.")

    def load_mask(self, image_id):
        """Return the bounding boxes and class IDs for the given image."""
        image_info = self.image_info[image_id]
        annotations = image_info['annotations']

        boxes = []
        class_ids = []

        for annotation in annotations:
            class_id = annotation['category_id']  # Assuming category_id is the class label
            bbox = annotation['bbox']  # [x, y, width, height]
            boxes.append(bbox)
            class_ids.append(class_id)

        return np.array(boxes, dtype=np.float32), np.array(class_ids, dtype=np.int32)
