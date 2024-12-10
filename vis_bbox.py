import colorsys
from selective_search import selective_search, box_filter

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle

from tqdm import tqdm

from torch.utils.data import DataLoader
from torchvision import transforms

from datasets import TACO
from datasets.helpers import *

from util.mypath import Path

root = Path.db_root_dir()
imset = 'train'
taco = TACO(root=root, imset=imset)
db = DataLoader(taco, batch_size=1, shuffle=True)

for data in tqdm(db, total=len(db)):
    img = data['img']
    annotation = data['annotations']
    
    fig,ax = plt.subplots(1)
    plt.axis('off')
    plt.imshow(im_normalize(tens2image(img)))
    # Show annotations
    for ann in annotation:
        color = colorsys.hsv_to_rgb(np.random.random(),1,1)
        [x, y, w, h] = [tensor.item() for tensor in ann['bbox']]
        rect = Rectangle((x,y),w,h,linewidth=2,edgecolor=color,
                         facecolor='none', alpha=0.7, linestyle = '--')
        ax.add_patch(rect)
   
    plt.savefig('./assets/img_gt_bboxes.png', bbox_inches='tight')
    plt.close()


    # run selective search
    rgb_trans = transforms.Compose([
        transforms.Resize((800,600), interpolation=transforms.InterpolationMode.BICUBIC),
                
    ])
    image = rgb_trans(img)
    image = im_normalize(tens2image(image))
    # Convert the image to an integer dtype
    image = (image * 255).astype(np.uint8)
    
    boxes = selective_search(image, mode='fast', random_sort=False)
    boxes_filter = box_filter(boxes, min_size=20, topN=80)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)
    for x1, y1, x2, y2 in boxes_filter:
        bbox = mpatches.Rectangle(
            (x1, y1), (x2-x1), (y2-y1), fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(bbox)
    
    plt.axis('off')
    plt.savefig('./assets/img_ss_bboxes.png', bbox_inches='tight')
    break