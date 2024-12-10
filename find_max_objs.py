import matplotlib.pyplot as plt
from datasets.helpers import *
from tqdm import tqdm

from matplotlib.patches import Rectangle
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets import TACOProposals,TACO
from util.mypath import Path

from selective_search import box_filter

db_root = Path.db_root_dir('TACO')

rgb_trans = transforms.Compose([
    transforms.Resize((800,600), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),        
])

db = TACOProposals(db_root, imset='val')
db_loader = DataLoader(db, batch_size=1, shuffle=False)

max_objs = -1
for data in tqdm(db_loader):
    num_gt = data['num_objs'].item()
    if num_gt > max_objs:
        max_objs = num_gt

print(max_objs)