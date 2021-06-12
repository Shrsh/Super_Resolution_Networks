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
from models import SRSN_RRDB

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
    
class SRFSN_RRDB(nn.Module):
    def __init__(self, input_dim=3, dim=128, scale_factor=4,scale_ratio=0.2):
        super(SRFSN_RRDB, self).__init__()
#         self.up = nn.ConvTranspose2d(3,3, 1, stride=1,output_size=(512,512))
        self.RDB1 = ResidualInResidualDenseBlock(64, 64, 0.2)
        self.RDB2 =ResidualInResidualDenseBlock(64, 64, 0.2)
        self.RDB3 = ResidualDenseBlock(64, 64, 0.2)
        self.RDB4 = ResidualDenseBlock(64, 64, 0.2)
        self.RDB5 = ResidualDenseBlock(64, 64, 0.2)
        self.RDB6 = ResidualDenseBlock(64, 64, 0.2)
        
#         self.up = torch.nn.Upsample(scale_factor=4, mode='bicubic')
        self.up = torch.nn.ConvTranspose2d(64,64,stride=4,kernel_size=4)
#         self.conv2_1=torch.nn.Conv2d(64, 16*16, 1, 1, 0)
#         self.up = torch.nn.PixelShuffle(4)
        
        self.conv3=torch.nn.Conv2d(64, 16, 3, 1, 1)
        self.conv4=torch.nn.Conv2d(16, 3, 1, 1, 0)
        self.conv5=torch.nn.Conv2d(64*6, 64, 1, 1, 0)
        
#         self.act1 = nn.PReLU()
#         self.act2 = nn.PReLU()
        
#         self.bn = torch.nn.BatchNorm2d(3)
        
        self.scale_ratio = 1

    def forward(self, LR):
#         LR_Feat = self.bn(LR)
#         LR_feat = F.leaky_relu(self.conv1(LR),negative_slope=0.2)
#         LR_feat = F.leaky_relu(self.conv2(LR_feat),negative_slope=0.2)
        
        ##Creating Skip connection between dense blocks 
        out = self.RDB1(LR) 
#         out = out + LR_feat
        out1= self.RDB2(out)
#         out1= out + out1
        
        out2 = self.RDB3(out1)
# #         out2 = out + out2
        
#         out2= out2 + LR_feat
        out3 = self.RDB4(out2)
        out3= out3 + LR
#         out3 = out + out1 + out3
        out4 = self.RDB5(out3)
        out5 = self.RDB6(out4)
        out6=torch.cat((out,out1,out2,out3,out4,out5), dim=1)
        out6=self.conv5(out6)
        out6= out6.mul(self.scale_ratio) + LR
#         out6 = self.up(out6)
#         print("After upsampling:",out6.shape)
        #LR_feat = self.resnet(out3)
#         SR=F.leaky_relu(self.conv3(out6),negative_slope=0.2)
#         SR =self.conv4(SR)
        # print(SR.shape)
        return out6   
    
    
class ResidualDenseBlock(nn.Module):
    r"""The residual block structure of traditional SRGAN and Dense model is defined"""

    def __init__(self, channels: int = 64, growth_channels: int = 48, scale_ratio: float = 0.2):
        """

        Args:
            channels (int): Number of channels in the input image. (Default: 64).
            growth_channels (int): how many filters to add each layer (`k` in paper). (Default: 32).
            scale_ratio (float): Residual channel scaling column. (Default: 0.2)
        """
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels + 0 * growth_channels, growth_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
#             nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels + 1 * growth_channels, growth_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
#             nn.PReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels + 2 * growth_channels, growth_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
#             nn.PReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(channels + 3 * growth_channels, growth_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
#             nn.PReLU()
        )
        self.conv5 = nn.Conv2d(channels + 4 * growth_channels, channels, kernel_size=3, stride=1, padding=1)

        self.scale_ratio = scale_ratio


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        conv1 = self.conv1(input)
        conv2 = self.conv2(torch.cat((input, conv1), 1))
        conv3 = self.conv3(torch.cat((input, conv1, conv2), 1))
        conv4 = self.conv4(torch.cat((input, conv1, conv2, conv3), 1))
        conv5 = self.conv5(torch.cat((input, conv1, conv2, conv3, conv4), 1))

        return conv5.mul(self.scale_ratio) + input
    
    


    
class ResidualInResidualDenseBlock(nn.Module):
    r"""The residual block structure of traditional ESRGAN and Dense model is defined"""

    def __init__(self, channels: int = 64, growth_channels: int = 32, scale_ratio: float = 0.2):
        """

        Args:
            channels (int): Number of channels in the input image. (Default: 64).
            growth_channels (int): how many filters to add each layer (`k` in paper). (Default: 32).
            scale_ratio (float): Residual channel scaling column. (Default: 0.2)
        """
        super(ResidualInResidualDenseBlock, self).__init__()
        self.RDB1 = ResidualDenseBlock(channels, growth_channels, scale_ratio)
        self.RDB2 = ResidualDenseBlock(channels, growth_channels, scale_ratio)
        self.RDB3 = ResidualDenseBlock(channels, growth_channels, scale_ratio)

        self.scale_ratio = scale_ratio

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.RDB1(input)
        out = self.RDB2(out)
        out = self.RDB3(out)

        return out.mul(self.scale_ratio) + input
    
# class SRFBN(nn.Module):
#     def __init__(self,num_steps):
#         super(SRFBN, self).__init__()
# #         self.up = SRSN_RRDB()
# #         self.up = nn.DataParallel(self.up,device_ids = device_ids)
# #         self.up = self.up.to(device)
# #         self.um_optimizer = optim.Adam(self.up.parameters(), lr=0.0002, betas=(0.9, 0.999), eps=1e-8)
# #         self.um_optimizer = nn.DataParallel(self.um_optimizer,device_ids = device_ids)
# #         self.um_optimizer = self.um_optimizer.to(device)
# #         self.upsample_model_file = os.path.join("/home/harsh.shukla/SRCNN/Weights","L1_28.52.pt")
# #         #Load Upsample Model 
# #         if os.path.exists(self.upsample_model_file):
# #             print("Loading Upsample Model from Previous Checkpoint...")
# #             self.checkpoint = torch.load(self.upsample_model_file)
# #             self.up.load_state_dict(self.checkpoint['generator_state_dict'])
# # #             self.um_optimizer.load_state_dict(self.checkpoint['g_state_dict'])
# #             self.up.eval()
# #         else:
# #             print("No previous checkpoints exist for upsample model, initialising network from start...")
        

#         # basic block
#         self.conv1 = torch.nn.Conv2d(3, 64, 9, 1, 4)
#         self.conv2 = torch.nn.Conv2d(64, 32, 5, 1,2)
#         self.num_steps = num_steps
# #         self.model=model
# #         self.up=up
#         self.block = SRFSN_RRDB()
#         self.conv3=torch.nn.Conv2d(64,32 , 1, 1, 0)
#         self.conv4 = torch.nn.Conv2d(64,16,3,1,1)
# #         self.up = torch.nn.Upsample(scale_factor=4, mode='bicubic')
#         self.trans = torch.nn.ConvTranspose2d(64,64,stride=4,kernel_size=4)
#         self.conv5=torch.nn.Conv2d(16,8 , 3, 1, 1)
#         self.conv6=torch.nn.Conv2d(8, 3, 1, 1, 0)
#         self.alpha = 0.7


#     def forward(self, x,y):
#         upsample = y
#         x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
#         x = F.leaky_relu(self.conv2(x), negative_slope=0.2) 
#         hidden=torch.zeros(x.size()).cuda()
#         outs = []
#         for _ in range(self.num_steps):
#             h = torch.cat((x, hidden), dim=1)
#             h = self.block(h)
#             hidden = self.conv3(h)
#             out_ = self.trans(h)
#             out6 = F.leaky_relu(self.conv4(out_), negative_slope = 0.2)
#             SR=F.leaky_relu(self.conv5(out6),negative_slope=0.2)
#             SR =self.conv6(SR)
#             outs.append(torch.add(upsample,SR))

#         return outs 

        
        
# class SRFBN(nn.Module):
#     def __init__(self,num_steps):
#         super(SRFBN, self).__init__()

#         self.conv1 = torch.nn.Conv2d(3, 128, 9, 1, 4)
#         self.conv2 = torch.nn.Conv2d(128, 64, 5, 1,2)
#         self.num_steps = num_steps
# #         self.model=model
# #         self.up=up
#         self.block = SRFSN_RRDB()
#         self.conv3=torch.nn.Conv2d(64,32 , 1, 1, 0)
#         self.conv4 = torch.nn.Conv2d(64,16,3,1,1)
# #         self.up = torch.nn.Upsample(scale_factor=4, mode='bicubic')
#         self.trans = torch.nn.ConvTranspose2d(64,64,stride=4,kernel_size=4)
#         self.conv5=torch.nn.Conv2d(16,3 , 1 ,1, 0)
#         self.conv6=torch.nn.Conv2d(8, 3, 1, 1, 0)
#         self.alpha = 0.7


#     def forward(self, x,y):
#         outs=[]
#         upsample = y
#         x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
#         x = F.leaky_relu(self.conv2(x), negative_slope=0.2) 
#         h = self.block(x)
#         out_ = self.trans(h)
#         out6 = F.leaky_relu(self.conv4(out_), negative_slope = 0.2)
#         SR=F.leaky_relu(self.conv5(out6),negative_slope=0.2)
# #         SR =self.conv6(SR)
#         outs.append(torch.add(upsample,SR))

#         return outs 
    
    
    

class SRFBN(nn.Module):
    def __init__(self, input_dim=3, dim=128, scale_factor=4,scale_ratio=0.2):
        super(SRFBN, self).__init__()
#         self.up = nn.ConvTranspose2d(3,3, 1, stride=1,output_size=(512,512))
        self.conv1 = torch.nn.Conv2d(3, 128, 9, 1, 4)
        self.conv2 = torch.nn.Conv2d(128, 64, 5, 1,2)
        self.RDB1 = ResidualInResidualDenseBlock(64, 64, 0.2)
        self.RDB2 =ResidualInResidualDenseBlock(64, 64, 0.2)
        self.RDB3 = ResidualDenseBlock(64, 64, 0.2)
        self.RDB4 = ResidualDenseBlock(64, 64, 0.2)
        self.RDB5 = ResidualDenseBlock(64, 64, 0.2)
        self.RDB6 = ResidualDenseBlock(64, 64, 0.2)
        
        self.up_image = torch.nn.Upsample(scale_factor=4, mode='bicubic')
        self.up = torch.nn.ConvTranspose2d(64,64,stride=4,kernel_size=4)
#         self.up_image = torch.nn.ConvTranspose2d(3,3,stride=4,kernel_size = 4)
#         self.conv2_1=torch.nn.Conv2d(64, 16*16, 1, 1, 0)
#         self.up = torch.nn.PixelShuffle(4)
        
        self.conv3=torch.nn.Conv2d(64, 16, 3, 1, 1)
        self.conv4=torch.nn.Conv2d(16, 3, 1, 1, 0)
        self.conv5=torch.nn.Conv2d(64*6, 64, 1, 1, 0)
        
#         self.act1 = nn.PReLU()
#         self.act2 = nn.PReLU()
        
#         self.bn = torch.nn.BatchNorm2d(3)
        
        self.scale_ratio = 1

    def forward(self, LR,y):
#         LR_Feat = self.bn(LR)
        LR_feat = F.leaky_relu(self.conv1(LR),negative_slope=0.2)
        LR_feat = F.leaky_relu(self.conv2(LR_feat),negative_slope=0.2)
        
        ##Creating Skip connection between dense blocks 
        out = self.RDB1(LR_feat) 
#         out = out + LR_feat
        out1= self.RDB2(out)
#         out1= out + out1
        
        out2 = self.RDB3(out1)
# #         out2 = out + out2
        
#         out2= out2 + LR_feat
        out3 = self.RDB4(out2)
        out3= out3 + LR_feat
#         out3 = out + out1 + out3
        out4 = self.RDB5(out3)
        out5 = self.RDB6(out4)
        out6=torch.cat((out,out1,out2,out3,out4,out5), dim=1)
        out6=self.conv5(out6)
        out6= out6.mul(self.scale_ratio) + LR_feat
#         print("Before Increasing channels:",out6.shape)
#         out6=self.conv2_1(out6)
#         out6=self.conv2_2(out6)
#         print("After Increasing channels:",out6.shape)
        out6 = self.up(out6)
#         print("After upsampling:",out6.shape)
        #LR_feat = self.resnet(out3)
        SR=F.leaky_relu(self.conv3(out6),negative_slope=0.2)
        SR =self.conv4(SR)
        
        # print(SR.shape)
        return torch.add(SR, y)  
    
    
class SRSN_RRDB1(nn.Module):
    def __init__(self, input_dim=3, dim=128, scale_factor=4,scale_ratio=0.2):
        super(SRSN_RRDB1, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 128, 9, 1, 4)
        self.conv2 = torch.nn.Conv2d(128, 64, 5, 1,2)
        self.RDB1 = ResidualInResidualDenseBlock(64, 64, 0.2)
        self.RDB2 =ResidualInResidualDenseBlock(64, 64, 0.2)
        self.RDB3 = ResidualDenseBlock(64, 64, 0.2)
        self.RDB4 = ResidualDenseBlock(64, 64, 0.2)
        self.RDB5 = ResidualDenseBlock(64, 64, 0.2)
        self.RDB6 = ResidualDenseBlock(64, 64, 0.2)
        
        self.up = torch.nn.Upsample(scale_factor=4, mode='bicubic')
        self.conv3=torch.nn.Conv2d(64, 16, 3, 1, 1)
        self.conv4=torch.nn.Conv2d(16, 3, 1, 1, 0)
        self.conv5=torch.nn.Conv2d(64*6, 64, 1, 1, 0)
        self.scale_ratio = 1

    def forward(self, LR):
        LR_feat = F.leaky_relu(self.conv1(LR),negative_slope=0.2)
        LR_feat = F.leaky_relu(self.conv2(LR_feat),negative_slope=0.2)
        
        out = self.RDB1(LR_feat) 
        out1= self.RDB2(out)
        
        out2 = self.RDB3(out1)        
        out3 = self.RDB4(out2)
        out3= out3 + LR_feat
        out4 = self.RDB5(out3)
        out5 = self.RDB6(out4)
        out6=torch.cat((out,out1,out2,out3,out4,out5), dim=1)
        out6=self.conv5(out6)
        out6= out6.mul(self.scale_ratio) + LR_feat
        out6 = self.up(out6)
        SR=F.leaky_relu(self.conv3(out6),negative_slope=0.2)
        SR =self.conv4(SR)
        return SR 