
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights, resnet101, ResNet101_Weights

class RCNN(nn.Module):
    def __init__(self, num_classes=29, arch='resnet18', regr_head=False, freeze_backbone=False, extra_layers=False):
        super(RCNN, self).__init__()

        assert arch in {'resnet18', 'resnet50', 'resnet101'}
        self.arch = arch

        if self.arch == 'resnet18':
            resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
            self.last_layer_num = 512  

        elif self.arch == 'resnet50':
            resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
            self.last_layer_num = 2048
        
        elif self.arch == 'resnet101':
            resnet = resnet101(weights=ResNet101_Weights.DEFAULT)
            self.last_layer_num = 2048
        
        if freeze_backbone:
            for param in resnet.parameters():
                param.requires_grad = False
                    
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1 # 1/4, 256
        self.layer2 = resnet.layer2 # 1/8, 512
        self.layer3 = resnet.layer3 # 1/16, 1024

        self.layer4 = resnet.layer4 # 1/32, 2048
        self.avg_pool_layer = torch.nn.AdaptiveAvgPool2d(output_size=1)  

        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()

        if extra_layers:
            self.clf_layer = nn.Sequential(
                nn.Linear(self.last_layer_num, 512),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(512, num_classes)
            )
        else :
            self.clf_layer = nn.Linear(self.last_layer_num, num_classes)

        if regr_head:
            if extra_layers:
                self.regr_head = nn.Sequential(
                    nn.Linear(self.last_layer_num, 512),
                    nn.ReLU(),
                    nn.Dropout(),
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Dropout(),
                    nn.Linear(512, 4)
                )
            else :
                self.regr_head = nn.Linear(self.last_layer_num, 4)
        else :
            self.regr_head = None
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)   # 1/2, 64
        x = self.maxpool(x)  # 1/4, 64
        x = self.layer1(x)   # 1/4, 256
        x = self.layer2(x) # 1/8, 512
        x = self.layer3(x) # 1/16, 1024

        x = self.layer4(x) # 1/32, 2048

        x  = self.avg_pool_layer(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x_clf = self.clf_layer(x)

        if self.regr_head is not None:
            return x_clf, self.regr_head(x)
        else :
            return x_clf