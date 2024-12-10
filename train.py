import os
from tqdm import tqdm
import wandb

# pytorch imports
import torch
import torch.nn as nn
from torch.optim import AdamW, SGD, RMSprop
from torch.optim.lr_scheduler import StepLR

from torch.utils.data import DataLoader, TensorDataset

# custom imports
from model.rcnn import RCNN

from util.hyper_para import HyperParameters
from datasets.helpers import *
from datasets import TACODetection
from util.mypath import Path
from util.util import sample_proposals

"""Load arguments"""
para = HyperParameters()
para.parse()

lr = para['lr']
epochs = para['epochs']
batch_size = para['batch_size']
optim_str = para['optim_str']
lr_decay = para['decay']
wd_id = para['wd_id']

# dataset parameters
ss_mode = para['mode']

# model parameters
arch = para['arch']
bbox_reg = para['bbox_reg']

weight_decays = {
    0: 0,
    1: 0.0000001,
    2: 0.000001,
    3: 0.00001,
    4: 0.0001,
    5: 0.001,
    6: 0.01,
}
wd = weight_decays[wd_id]

device='cuda' if torch.cuda.is_available() else 'cpu'

"""Paths"""
db_root = Path.db_root_dir('TACO')
model_dir = Path.models_dir('detector')

# incoporate training params to path
model_dir = os.path.join(model_dir, f'optim_{optim_str}_lr_{lr}_b_{batch_size}_wd_{wd}')
if lr_decay is not None :
    model_dir += f'_lr_decay_{lr_decay}'

model_dir += f'_arch_{arch}_mode_{ss_mode}'

if bbox_reg:
    model_dir += f'_boxreg'

if para['freeze']:
    model_dir+='_freeze'

if para['extra_layers']:
    model_dir+= '_extra_layers'

os.makedirs(model_dir, exist_ok=True)

"""Datasets and Dataloaders"""
train_db = TACODetection(db_root, imset='train', mode=ss_mode)
train_loader = DataLoader(train_db, batch_size=1, shuffle=True)

val_db = TACODetection(db_root, imset='val', mode=ss_mode)
val_loader = DataLoader(val_db, batch_size=1, shuffle=True)

"""Model"""
model = RCNN(arch=arch, regr_head=bbox_reg, freeze_backbone=para['freeze'], extra_layers=para['extra_layers'])
model.to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

"""Loss function"""
loss_fn = nn.CrossEntropyLoss()
loss_regr_fn = nn.SmoothL1Loss()

activation_fn = nn.Softmax(dim=1)

"""Optimizer"""
if optim_str == 'Adam':    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
elif optim_str == 'SGD' :
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
elif optim_str == 'rmsprop':
    optimizer = RMSprop(model.parameters(), lr=lr, weight_decay=wd)
else :
    raise NotImplementedError('No implementation for this optimizer')

# lr scheduler
if lr_decay is not None:
    scheduler = StepLR(optimizer, step_size=int(lr_decay), gamma=0.1)


"""Print Info"""
wandb.init(project=f"rcnn", entity="ak203")
wandb.config = {
    'lr' : lr,
    'optim': optim_str,
    'lr_decay': lr_decay,
    'batch_size': batch_size,
    'weight_decay' : wd,
    'arch': arch,
    'aug': False,
    'mode': ss_mode,
    'regr head': bbox_reg,
    'freeze_backbone': para['freeze'],
    'extra_layers': para['extra_layers']   
}

print(f'[DB INFO] train len={len(train_loader)}')
print(f'[DB INFO] val len={len(val_loader)}')
print(f'[DB INFO] mode={ss_mode}')
print(f'[MODEL INFO] Architecture: {arch}')
print(f'[MODEL INFO] Bbox head: {bbox_reg}')
print(f"[MODEL INFO] Freeze backbone: {para['freeze']}")
print(f"[MODEL INFO] Extra layers: {para['extra_layers']}")
print(f'[MODEL INFO] Trainable parameters: {total_params/1_000_000:.2f}M')
print(f'[INFO] Optim={optim_str}')
print(f'[INFO] lr={lr}')
if lr_decay is not None:
    print(f'[INFO] decay lr every {lr_decay} epochs')
print(f'[INFO] batch size={batch_size}')
print(f'[INFO] weight decay={wd}')
print(f'[INFO] save_path={model_dir}')
      
"""Training process"""
min_val_loss = 1e10
for e in range(para['epochs']):
    model.train()
    total_train_loss = 0.0
    total_train_acc = 0.0

    for data in tqdm(train_loader, total=len(train_loader), desc=f'Epoch {e+1}/{epochs}'):
        
        img = data['img'][0].to(device)
        img_id = data['img_id'].item()
        num_objs = data['num_objs'].to(device)
        obj_proposals = data['obj_proposals'][0].to(device)
        proposal_labels = data['proposal_labels'][0].to(device)
        gt_labels = data['gt_labels'][0].to(device)
        gt_boxes = data['gt_boxes'][0].to(torch.int).to(device)
        proposal_refinments = data['proposal_refinments'][0].to(device)
        gt_refinments = torch.zeros((gt_boxes.shape[0],4), device=device) # zero refinment for the gt boxes

        
        proposals, labels, refinments = sample_proposals(img=img, obj_proposals=obj_proposals, proposal_labels=proposal_labels, 
            proposal_refinments=proposal_refinments, gt_boxes=gt_boxes, gt_labels=gt_labels, gt_refinments=gt_refinments, 
            back_label=BACKGROUND_LABEL)
        
        # backprop for each proposal
        batch_loss = 0.0
        batch_acc = 0.0
        tensor_dataset = TensorDataset(proposals, labels, refinments)
        db_loader = DataLoader(tensor_dataset, batch_size=batch_size) 

        for proposals, labels, refs in db_loader:
            optimizer.zero_grad()
            # classify each proposal
            if not bbox_reg:
                y_clf = model(proposals)
                propasal_loss = loss_fn(y_clf, labels)
                propasal_loss.backward()
                optimizer.step()
            else:
                y_clf, y_regr = model(proposals)
                clf_loss = loss_fn(y_clf, labels)
                regr_loss = loss_regr_fn(y_regr, refs)
                
                propasal_loss = 0.5*clf_loss + 0.5*regr_loss
                propasal_loss.backward()
                optimizer.step()

            batch_loss += propasal_loss.item()     
            y_clf = activation_fn(y_clf)
            predicted = y_clf.argmax(dim=1)
            batch_acc += (predicted == labels).sum().item()/y_clf.shape[0]       

        total_train_loss += batch_loss/len(db_loader)
        total_train_acc += batch_acc/len(db_loader)

    model.eval()
    total_val_loss = 0.0
    total_val_acc = 0.0

    for data in tqdm(val_loader, total=len(val_loader), desc='Validation'):
        
        img = data['img'][0].to(device)
        img_id = data['img_id'].item()
        num_objs = data['num_objs'].to(device)
        obj_proposals = data['obj_proposals'][0].to(device)
        proposal_labels = data['proposal_labels'][0].to(device)
        gt_labels = data['gt_labels'][0].to(device)
        gt_boxes = data['gt_boxes'][0].to(torch.int).to(device)
        proposal_refinments = data['proposal_refinments'][0].to(device)
        gt_refinments = torch.zeros((gt_boxes.shape[0],4), device=device) # zero refinment for the gt boxes

        proposals, labels, refinments = sample_proposals(img=img, obj_proposals=obj_proposals, proposal_labels=proposal_labels, 
            proposal_refinments=proposal_refinments, gt_boxes=gt_boxes, gt_labels=gt_labels, gt_refinments=gt_refinments, 
            back_label=BACKGROUND_LABEL)
        
        tensor_dataset = TensorDataset(proposals, labels, refinments)
        db_loader = DataLoader(tensor_dataset, batch_size=16) 

        batch_loss = 0.0
        batch_acc = 0.0
        with torch.no_grad():
            for proposals, labels, refs in db_loader:
                
                if not bbox_reg:
                    y_clf = model(proposals)
                    propasal_loss = loss_fn(y_clf, labels)
                else:
                    y_clf, y_regr = model(proposals)
                    clf_loss = loss_fn(y_clf, labels)
                    regr_loss = loss_regr_fn(y_regr, refs)
                    
                    propasal_loss = 0.5*clf_loss + 0.5*regr_loss

            
                batch_loss += propasal_loss.item()     
                y_clf = activation_fn(y_clf)
                predicted = y_clf.argmax(dim=1)
                batch_acc += (predicted == labels).sum().item()/y_clf.shape[0] 

        total_val_loss += batch_loss/len(db_loader)
        total_val_acc += batch_acc/len(db_loader)

    if lr_decay is not None:
        scheduler.step()
    
    total_train_loss /= len(train_loader)
    total_train_acc /= len(train_loader)

    total_val_loss /= len(val_loader)
    total_val_acc /= len(val_loader)

    wandb.log({
        'Train Loss' : total_train_loss,
        'Train Accuracy': total_train_acc,
        'Val Loss' : total_val_loss,
        'Val Accuracy': total_val_acc,
        'Epoch': e+1
    })


    if min_val_loss >= total_val_loss:
        model_path = os.path.join(model_dir, f'rcnn.pth')      
        net_state_dict = model.state_dict()
        torch.save(net_state_dict, model_path)
        last_epoch = e
        min_val_loss = total_val_loss

print(f'Best model at epoch {last_epoch} with val loss: {min_val_loss:.3f}')


    


    