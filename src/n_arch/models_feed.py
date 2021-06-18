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
# from models import SRSN_RRDB

use_cuda = torch.cuda.is_available()
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
device_ids = [i for i in range(torch.cuda.device_count())]
device = 'cuda' if use_cuda else 'cpu'


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
    
 
    
    
class ResidualDenseBlock(nn.Module):
    r"""The residual block structure of traditional SRGAN and Dense model is defined"""

    def __init__(self, channels: int = 64, growth_channels: int = 64, scale_ratio: float = 0.2,kernel_size: int = 3):
        """

        Args:
            channels (int): Number of channels in the input image. (Default: 64).
            growth_channels (int): how many filters to add each layer (`k` in paper). (Default: 32).
            scale_ratio (float): Residual channel scaling column. (Default: 0.2)
        """
        super(ResidualDenseBlock, self).__init__()
        self.conv1= ResNetBlock(channels + 0*growth_channels, growth_channels,kernel_size=kernel_size,dilation=1)
        self.conv2= ResNetBlock(channels + 0*growth_channels, growth_channels,kernel_size=kernel_size, dilation=1)
        self.conv3= ResNetBlock(channels + 0*growth_channels, growth_channels,kernel_size=kernel_size, dilation=1)
        self.conv4= ResNetBlock(channels + 0*growth_channels, growth_channels,kernel_size=kernel_size, dilation=1)
        self.conv5 = nn.Sequential(
            nn.Conv2d(channels + 4*growth_channels, channels, kernel_size, 1,int((kernel_size-1)/2)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.scale_ratio = scale_ratio


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        conv1 = self.conv1(input)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(torch.cat((input,conv1,conv2,conv3, conv4), 1))
        return (conv5+conv4).mul(self.scale_ratio)+ input



    
class ResNetBlock(nn.Module):
    r"""Resnet block structure"""

    def __init__(self, in_channels: int = 64,out_channels: int = 64,kernel_size=3,scale_ratio: float = 0.2,negative_slope=0.2,dilation=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=int((kernel_size*dilation-1)/2),dilation=dilation),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=int((kernel_size*dilation-1)/2),dilation=dilation),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        )
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels,1,1,0),
# #             nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
#         )
        self.scale_ratio = scale_ratio


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        conv1 = self.conv1(input)
        conv2 = self.conv2(conv1)+input
        return conv2

class SRFSN_RRDB(nn.Module):
    def __init__(self, input_dim=3, dim=128, scale_factor=4,scale_ratio=0.2):
        super(SRFSN_RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock(kernel_size=3)
        self.RDB2 = ResidualDenseBlock(kernel_size=3)
        self.RDB3 = ResidualDenseBlock(kernel_size=3)
        self.RDB4 = ResidualDenseBlock(kernel_size=3)
        self.RDB5 = ResidualDenseBlock(kernel_size=3)
        self.RDB6 = ResidualDenseBlock(kernel_size=3)
        self.RDB7 = ResidualDenseBlock(kernel_size=3)
        self.RDB8 = ResidualDenseBlock(kernel_size=3)
        self.conv1 = nn.Sequential(
            nn.Conv2d(64 , 32, 3, 1,1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, LR):
        
        ##Creating Skip connection between dense blocks 
        out = self.RDB1(LR) 
        out1= self.RDB2(out)
        out2 = self.RDB3(out1)
        out3 = self.RDB4(out2)
        out4 = self.RDB5(out3)
        out5 = self.RDB6(out4)
        out6 = self.RDB7(out5)
        out7 = self.RDB8(out6)
        return out7,self.conv1(out7)  

    
class SRFBN(nn.Module):
    def __init__(self,num_steps):
        super(SRFBN, self).__init__()

        self.conv1 = torch.nn.Conv2d(3, 16, 9, 1, 4)
        self.conv2 = torch.nn.Conv2d(16,32, 7, 1,3)
        self.num_steps = num_steps

        self.block = SRFSN_RRDB()
        self.conv3=torch.nn.Conv2d(64, 16, 3, 1, 1)
        self.conv4=torch.nn.Conv2d(16, 3, 1, 1, 0)
        self.up = torch.nn.Upsample(scale_factor=4, mode='bicubic')
        self.trans = torch.nn.ConvTranspose2d(64,64,stride=4,kernel_size=4)



    def forward(self, x,y):
        upsample = y
        x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.2) 
        hidden=torch.zeros(x.size()).cuda()
        outs = []
        for _ in range(self.num_steps):
            h = torch.cat((x, hidden), dim=1)
            h,hidden = self.block(h)
            out_ = self.trans(h)
            h = F.leaky_relu(self.conv3(out_),negative_slope=0.2)
            SR= self.conv4(h)
  
            outs.append(torch.add(self.up(y),SR))

        return outs 


