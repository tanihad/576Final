

import os
import json
import numpy as np
import copy
import utils
import cv2  # Required for Selective Search #tanihad

from PIL import Image, ExifTags
from pycocotools.coco import COCO

class Taco(utils.Dataset):

    def load_taco(self, dataset_dir, round, subset, class_ids=None,
                  class_map=None, return_taco=False, auto_download=False):
        """Load a subset of the TACO dataset.
        dataset_dir: The root directory of the TACO dataset.
        round: split number
        subset: which subset to load (train, val, test)
        class_ids: If provided, only loads images that have the given classes.
        class_map: Dictionary used to assign original classes to new class system
        return_coco: If True, returns the COCO object.
        auto_download: Automatically download and unzip MS-COCO images and annotations
        """
        ann_filepath = os.path.join(dataset_dir , 'annotations')
        if round is not None:
            ann_filepath += f"_{round}_{subset}.json"
        else:
            ann_filepath += ".json"

        assert os.path.isfile(ann_filepath)

        # Load dataset
        dataset = json.load(open(ann_filepath, 'r'))

        # Replace dataset original classes before calling the COCO Constructor
        # Some classes may be assigned background to remove them from the dataset
        self.replace_dataset_classes(dataset, class_map)

        taco_alla_coco = COCO()
        taco_alla_coco.dataset = dataset
        taco_alla_coco.createIndex()

        # Add images and classes except Background
        image_ids = []
        background_id = -1
        class_ids = sorted(taco_alla_coco.getCatIds())
        for i in class_ids:
            class_name = taco_alla_coco.loadCats(i)[0]["name"]
            if class_name != 'Background':
                self.add_class("taco", i, class_name)
                image_ids.extend(list(taco_alla_coco.getImgIds(catIds=i)))
            else:
                background_id = i
        image_ids = list(set(image_ids))

        if background_id > -1:
            class_ids.remove(background_id)

        print('Number of images used:', len(image_ids))

        # Add images
        for i in image_ids:
            self.add_image(
                "taco", image_id=i,
                path=os.path.join(dataset_dir, taco_alla_coco.imgs[i]['file_name']),
                width=taco_alla_coco.imgs[i]["width"],
                height=taco_alla_coco.imgs[i]["height"],
                annotations=taco_alla_coco.loadAnns(taco_alla_coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None))
            )
        if return_taco:
            return taco_alla_coco
##STILL A LOT OF CHECKING ;-;
    def load_image_with_proposals(self, image_id):  #tanihad
        """Load an image and generate Selective Search proposals."""
        image_path = self.image_info[image_id]['path']  #tanihad
        image = cv2.imread(image_path)  #tanihad
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        ss.setBaseImage(image)
        ss.switchToSelectiveSearchFast()
        proposals = ss.process()
        return image, proposals

    def replace_dataset_classes(self, dataset, class_map):
        """Replaces classes of dataset based on a dictionary."""
        class_new_names = list(set(class_map.values()))
        class_new_names.sort()
        class_originals = copy.deepcopy(dataset['categories'])
        dataset['categories'] = []
        class_ids_map = {}  # Map from old id to new id

        # Assign background id 0
        has_background = False
        if 'Background' in class_new_names:
            if class_new_names.index('Background') != 0:
                class_new_names.remove('Background')
                class_new_names.insert(0, 'Background')
            has_background = True

        # Replace categories
        for id_new, class_new_name in enumerate(class_new_names):

            # Make sure id:0 is reserved for background
            id_rectified = id_new
            if not has_background:
                id_rectified += 1

            category = {
                'supercategory': '',
                'id': id_rectified,  # Background has id=0
                'name': class_new_name,
            }
            dataset['categories'].append(category)
            # Map class names
            for class_original in class_originals:
                if class_map[class_original['name']] == class_new_name:
                    class_ids_map[class_original['id']] = id_rectified

        # Update annotations category id tag
        for ann in dataset['annotations']:
            ann['category_id'] = class_ids_map[ann['category_id']]
