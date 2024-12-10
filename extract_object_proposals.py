import os
import argparse
# selective search
from selective_search import selective_search

from tqdm import tqdm

# pytorch imports
from torch.utils.data import DataLoader
from torchvision import transforms

# custom imports
from datasets import TACO
from datasets.helpers import *

from util.mypath import Path

parser = argparse.ArgumentParser()
parser.add_argument('--imset', type=str, default='train')
parser.add_argument('--mode', type=str, default='fast')

args = parser.parse_args()
imset = args.imset
mode = args.mode
assert mode in {'fast', 'quality', 'single'}
assert imset in {'train', 'val', 'test'}
root = Path.db_root_dir('TACO')

save_path = os.path.join(root, f'object_proposals_{mode}', imset)
os.makedirs(save_path, exist_ok=True)


if mode == 'quality':
    im_transform = transforms.Compose([
        transforms.Resize((600,400), interpolation=transforms.InterpolationMode.BICUBIC),     
        transforms.ToTensor(),
    ])

else :
    im_transform = transforms.Compose([
        transforms.Resize((800,600), interpolation=transforms.InterpolationMode.BICUBIC),     
        transforms.ToTensor(),
    ])

taco = TACO(root=root, imset=imset, input_transform=im_transform)
db = DataLoader(taco, batch_size=1, shuffle=False)

for data in tqdm(db, total=len(db), desc=f'Selective search {mode}'):
    
    img = data['img']
    img_id = data['img_id'].item()
    image = im_normalize(tens2image(img))
    gt_boxes = data['gt_boxes'][0]
    # Convert the image to an integer dtype
    image = (image * 255).astype(np.uint8)
    boxes = selective_search(image, mode=mode, random_sort=False)
    boxes = np.array(boxes)
    np.save(os.path.join(save_path, f'object_proposals_img_{img_id}.npy'), boxes)
        
        