import os
import json

from util.mypath import Path
from util.preprocess_taco import keep_supercategories, split_dataset

root = Path.db_root_dir('TACO')

annotation_file = os.path.join(root, f'annotations.json')

with open(annotation_file, 'r') as f:
    dataset = json.loads(f.read())

dataset = keep_supercategories(dataset)
#split_dataset(dataset)

