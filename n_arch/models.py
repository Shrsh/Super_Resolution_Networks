import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import math 
from torch.autograd import Variable

import os
from os import listdir
from os.path import isfile, join 
from PIL import Image
import cv2
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.utils import save_image
import torch.utils.data as data
from PIL import Image 
import pickle
from PIL import Image as im
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.metrics import Metric
from pytorch_lightning import loggers as pl_loggers
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
from IPython.display import clear_output
import argparse 
import pickle as pkl

use_cuda = torch.cuda.is_available()
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
device_ids = [i for i in range(torch.cuda.device_count())]
device = 'cuda' if use_cuda else 'cpu'
torch.backends.cudnn.benchmark = True


class DiscriminativeNet(torch.nn.Module):
    
    def __init__(self):
        super(DiscriminativeNet, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=32, kernel_size=6, 
                stride=2, padding=2, bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=6,
                stride=2, padding=2, bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(64)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(128)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=4,
                stride=4, padding=0, bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),
              nn.BatchNorm2d(256)
        )
#         self.conv5 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=256, out_channels=512, kernel_size=4,
#                 stride=2, padding=1, bias=False
#             ),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.BatchNorm2d(512)
#         )
#         self.conv6 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=512, out_channels=256, kernel_size=4,
#                 stride=2, padding=1, bias=False
#             ),
#             nn.BatchNorm2d(256)
#         )
        self.conv7 = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=128, kernel_size=4,
                stride=4, padding=0, bias=False
            ),
            nn.BatchNorm2d(128)
        )
        self.Fc1 = nn.Sequential(
            nn.Linear(128*4*4, 512),
            nn.LeakyReLU(0.2, inplace=True)
            # nn.Sigmoid(),
        )
        self.Fc2 = nn.Sequential(
            nn.Linear(512,1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
#         x = self.conv5(x)
#         x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        
        # Flatten and apply sigmoid
#         print(x.shape)
        x = x.view(-1, 128*4*4)
#         print(x.shape)
        x = self.Fc1(x)
#         print(x.shape)
        x = self.Fc2(x)
#         print(x.shape)
        # print(x)
        return x
    
class VGGFeatureExtractor(nn.Module):
    def __init__(self,
                 low_feature_layer=22,
                 high_feature_layer = 36,
                 use_bn=False,
                 device=torch.device('cuda')):
        super(VGGFeatureExtractor, self).__init__()
        if use_bn:
            model = torchvision.models.vgg19_bn(pretrained=True)
        else:
            model = torchvision.models.vgg19(pretrained=True)
        self.features_low = nn.Sequential(*list(model.features.children())[:(low_feature_layer + 1)])
        self.features_high = nn.Sequential(*list(model.features.children())[:(high_feature_layer + 1)])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.features_low(x), self.features_high(x)


class Block(nn.Module):

    def __init__(self, channels: int = 64, growth_channels: int = 48, scale_ratio: float = 0.2,negative_slope=0.2,kernel_size=3):
        super(Block, self).__init__()
        self.conv1 = ResNetBlock(kernel_size=kernel_size)
        self.conv2 = ResNetBlock(kernel_size=kernel_size)
        self.conv3 = ResNetBlock(kernel_size=kernel_size)
        self.scale_ratio = scale_ratio


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        conv1 = self.conv1(input)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        return conv3.mul(self.scale_ratio) + input
    
class ResNetBlock(nn.Module):
    r"""Resnet block structure"""

    def __init__(self, in_channels: int = 64,out_channels: int = 64,kernel_size=3,scale_ratio: float = 0.2,negative_slope=0.2):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=int((kernel_size-1)/2)),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=int((kernel_size-1)/2)),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        )
        self.scale_ratio = scale_ratio


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        conv1 = self.conv1(input)
        conv2 = self.conv2(conv1)

        return conv2.mul(self.scale_ratio) + input

class arch(nn.Module):

    def __init__(self, input_dim=3, dim=128, scale_factor=4,scale_ratio=0.2,negative_slope=0.2):
        super(arch, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 9, 1, 4)
        self.conv2 = torch.nn.Conv2d(32, 64, 7, 1,3)
        self.up_image = torch.nn.Upsample(scale_factor=4, mode='bicubic')
        self.up = torch.nn.ConvTranspose2d(64,64,stride=4,kernel_size=4)
        self.block1 = Block(kernel_size=5)
        self.block2 = Block(kernel_size=5)
        self.block3 = Block(kernel_size=3)
        self.block4 = Block(kernel_size=3)
        self.block5 = Block(kernel_size=3)
        self.conv3=torch.nn.Conv2d(64, 16, 3, 1, 1)
        self.conv4=torch.nn.Conv2d(16, 3, 1, 1, 0)
        
    def forward(self, LR):
#         LR_Feat = self.bn(LR)
        LR_feat = F.leaky_relu(self.conv1(LR),negative_slope=0.2)
        LR_feat = F.leaky_relu(self.conv2(LR_feat),negative_slope=0.2)
        out1 = self.block1(LR_feat)
        out2 = self.block2(out1) + LR_feat
        out3 = self.block3(out2)
        out4 = self.block4(out3) + out2
        out5 = self.block5(out4)
        
        out6 = out5 + LR_feat
        
        out7 = self.conv3(self.up(out6))
        out8 = self.conv4(out7)
        return torch.add(out8,self.up_image(LR)),out8, self.up_image(LR)  
        