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
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from __future__ import print_function, division
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
import argparse 
import warnings
warnings.filterwarnings("ignore")
from IPython.display import clear_output
from prettytable import PrettyTable
import math
import torch.nn.init as init
from scipy.ndimage.filters import gaussian_filter




# CUDA for PyTorch
print("Number of GPUs:" + str(torch.cuda.device_count()))


use_cuda = torch.cuda.is_available()
torch.no_grad()
torch.cuda.empty_cache()
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device_ids = [i for i in range(torch.cuda.device_count())]
device = 'cuda' if use_cuda else 'cpu'
torch.backends.cudnn.benchmark = True

trans = transforms.ToPILImage()
trans1 = transforms.ToTensor()
torch.autograd.set_detect_anomaly(True)

### Network Debugging
#########################################################################

### Creating function for Gradient Visualisation 
def plot_grad_flow(result_directory,named_parameters,model_name): 
    
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.figure(figsize=(12,12))
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.savefig(os.path.join(result_directory, model_name + "gradient_flow.png" ))
    
### Get all the children layers 
def get_children(model: torch.nn.Module):
    # get children form model!
    children = list(model.children())
    flatt_children = []
    if children == []:
        # if model has no children; model is last child! :O
        return model
    else:
       # look for children from children... to the last child!
       for child in children:
            try:
                flatt_children.extend(get_children(child))
            except TypeError:
                flatt_children.append(get_children(child))
    return flatt_children
    

### Layer Activation in CNNs 

    
def visualise_layer_activation(model,local_batch,result_directory):
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    layer_name = 'conv1'
    model.module.conv1.register_forward_hook(get_activation(layer_name))
    output = model(local_batch)
    act = activation[layer_name].squeeze()
    print(act.shape)
    #plot subplots for different images
    for i in range(act[0].shape[0]):
        output = im.fromarray(np.uint8(np.moveaxis(act[0][i].cpu().detach().numpy(), 0, -1))).convert('RGB')
        output.save(os.path.join(result_directory,str(i)+'.png'))
        #

### Visualising Conv Filters
def visualise_conv_filters(model,result_directory):
    kernels = model.conv1.weight.detach()
    print(kernels.shape)
    # fig, axarr = plt.subplots(kernels.size(0))
    # for i in range(kernels.shape[0]):
    #     plt.savefig()
    # for idx in range(kernels.size(0)):
    #     axarr[idx].imsave(kernels[idx].squeeze(),result_directory + "1.png")
    #
    
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
#         init.kaiming_uniform_(m.weight.data, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
#         init.xavier_normal_(m.weight.data)
#         init.xavier_uniform_(m.weight.data, gain=1.0)
        torch.nn.init.kaiming_normal_(m.weight.data, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
#         init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
#         init.xavier_normal_(m.weight.data)
        torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
#         init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
#         init.xavier_normal_(m.weight.data)
        torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
#         init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
#         init.xavier_normal_(m.weight.data)
        torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
        


#########################################################################################################################################################################################################



def load_images_from_folder(folder):
    c=0
    images = []
    list_name=[]
    for filename in os.listdir(folder):
        list_name.append(os.path.join(folder,filename))
    list_name.sort()
    for filename in list_name:
        img = cv2.imread(filename)
        if img is not None:
            images.append(img)
    return images


class SRSN_Generator(nn.Module):
    def __init__(self, input_dim=3, dim=64, scale_factor=4):
        super(SRSN_Generator, self).__init__()
#         self.up = nn.ConvTranspose2d(3,3, 1, stride=1,output_size=(512,512))
        self.conv1 = torch.nn.Conv2d(3, 128, 9, 1, 4)
        self.conv2 = torch.nn.Conv2d(128, 64, 1, 1, 0)
        self.resnet1 = Modified_Resnet_Block(dim, 7, 1, 3, bias=True)
        self.resnet2 = Modified_Resnet_Block(dim, 7, 1, 3, bias=True)
        self.resnet3 = Modified_Resnet_Block(dim, 5, 1, 2, bias=True)
        self.resnet4 = Modified_Resnet_Block(dim, 3, 1, 1, bias=True)
        self.up = torch.nn.Upsample(scale_factor=4, mode='bicubic')
        self.conv3=torch.nn.Conv2d(64, 16, 1, 1, 0)
        self.conv4=torch.nn.Conv2d(16, 3, 1, 1, 0)

    def forward(self, LR):
        LR_feat = F.leaky_relu(self.conv1(LR))
        LR_feat = (self.conv2(LR_feat))
        
        ##Creating Skip connection between dense blocks 
        out = self.resnet1(LR_feat) 
        out = out + LR_feat
        out1= self.resnet2(out)
        out1= out + out1
        
        out2 = self.resnet3(out1)
        out2 = out + out2
        
        out3 = self.resnet4(out2)
        out3 = out + out1 + out3 
        out3 = self.up(out3)
        #LR_feat = self.resnet(out3)
        SR=F.leaky_relu(self.conv3(out3))
        SR =self.conv4(SR)
        # print(SR.shape)
        return SR
    
class SRSN_RRDB(nn.Module):
    def __init__(self, input_dim=3, dim=128, scale_factor=4,scale_ratio=0.2):
        super(SRSN_RRDB, self).__init__()
#         self.up = nn.ConvTranspose2d(3,3, 1, stride=1,output_size=(512,512))
        self.conv1 = torch.nn.Conv2d(3, 128, 9, 1, 4)
        self.conv2 = torch.nn.Conv2d(128, 64, 3, 1, 1)
        self.RDB1 = ResidualDenseBlock(64, 32, 0.2)
        self.RDB2 = ResidualDenseBlock(64, 32, 0.2)
        self.RDB3 = ResidualDenseBlock(64, 32, 0.2)
        self.RDB4 = ResidualDenseBlock(64, 32, 0.2)
        self.RDB5 = ResidualDenseBlock(64, 32, 0.2)
        self.RDB6 = ResidualDenseBlock(64, 32, 0.2)
        
        self.up = torch.nn.Upsample(scale_factor=4, mode='bicubic')
        self.conv3=torch.nn.Conv2d(64, 16, 1, 1, 0)
        self.conv4=torch.nn.Conv2d(16, 3, 1, 1, 0)
        self.conv5=torch.nn.Conv2d(64*6, 64, 1, 1, 0)
        
        self.scale_ratio = 1

    def forward(self, LR):
        LR_feat = F.leaky_relu(self.conv1(LR),negative_slope=0.2)
        LR_feat = (self.conv2(LR_feat))
        
        ##Creating Skip connection between dense blocks 
        out = self.RDB1(LR_feat) 
#         out = out + LR_feat
        out1= self.RDB2(out)
#         out1= out + out1
        
        out2 = self.RDB3(out1)
#         out2 = out + out2
        
        out3 = self.RDB4(out2)
#         out3 = out + out1 + out3
        out4 = self.RDB5(out3)
        out5 = self.RDB6(out4)
#         out6=torch.cat((out,out1,out2,out3,out4,out5), dim=1)
#         out6=self.conv5(out6)
        out6= out5.mul(self.scale_ratio) + LR_feat
        out6 = self.up(out6)
        #LR_feat = self.resnet(out3)
        SR=F.leaky_relu(self.conv3(out6),negative_slope=0.2)
        SR =self.conv4(SR)
        # print(SR.shape)
        return SR   
    
class SRSN(nn.Module):
    def __init__(self, input_dim=3, dim=128, scale_factor=4):
        super(SRSN, self).__init__()
#         self.up = nn.ConvTranspose2d(3,3, 1, stride=1,output_size=(512,512))
        self.conv1 = torch.nn.Conv2d(3, 128, 9, 1, 4,)
        self.conv2 = torch.nn.Conv2d(128, 128, 1, 1, 0)
        self.resnet1 = ResnetBlock(dim, 7, 1, 3, bias=True)
        self.resnet2 = ResnetBlock(dim, 7, 1, 3, bias=True)
        self.resnet3 = ResnetBlock(dim, 5, 1, 2, bias=True)
        self.resnet4 = ResnetBlock(dim, 3, 1, 1, bias=True)
        self.up = torch.nn.Upsample(scale_factor=4, mode='bicubic')
        self.conv3=torch.nn.Conv2d(128, 16, 1, 1, 0)
        self.conv4=torch.nn.Conv2d(16, 3, 1, 1, 0)

    def forward(self, LR):
        LR_feat = F.leaky_relu(self.conv1(LR))
        LR_feat = (self.conv2(LR_feat))
        
        ##Creating Skip connection between dense blocks 
        out = self.resnet1(LR_feat) 
#         out = out + LR_feat
        out1= self.resnet2(out)
#         out1= out + out1
        
        out2 = self.resnet3(out1)
#         out2 = out + out2
        
        out3 = self.resnet4(out2)
#         out3 = out + out1 + out3 
        out3= out3 + LR_feat
        out3 = self.up(out3)
        #LR_feat = self.resnet(out3)
        SR=F.leaky_relu(self.conv3(out3))
        SR =self.conv4(SR)
        # print(SR.shape)
        return SR
    
class ResnetBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=3, stride=1, padding=1, bias=True):
        super(ResnetBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)
        self.conv2 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)

        self.act1 = torch.nn.LeakyReLU(inplace=True)
        self.act2 = torch.nn.LeakyReLU(inplace=True)


    def forward(self, x):

        out = self.act1(x)
        out = self.conv1(out)

        out = self.act2(out)
        out = self.conv2(out)

        out = out + x

        return out
    
class ResidualDenseBlock(nn.Module):
    r"""The residual block structure of traditional SRGAN and Dense model is defined"""

    def __init__(self, channels: int = 64, growth_channels: int = 32, scale_ratio: float = 0.2):
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
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels + 1 * growth_channels, growth_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels + 2 * growth_channels, growth_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(channels + 3 * growth_channels, growth_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv5 = nn.Conv2d(channels + 4 * growth_channels, channels, kernel_size=3, stride=1, padding=1)

        self.scale_ratio = scale_ratio

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight)
#                 m.weight.data *= 0.1
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight)
#                 m.weight.data *= 0.1
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias.data, 0.0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        conv1 = self.conv1(input)
        conv2 = self.conv2(torch.cat((input, conv1), 1))
        conv3 = self.conv3(torch.cat((input, conv1, conv2), 1))
        conv4 = self.conv4(torch.cat((input, conv1, conv2, conv3), 1))
        conv5 = self.conv5(torch.cat((input, conv1, conv2, conv3, conv4), 1))

        return conv5.mul(self.scale_ratio) + input
    
    
    
class Modified_Resnet_Block(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=3, stride=1, padding=1, bias=True):
        super(Modified_Resnet_Block, self).__init__()
        self.conv1 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)
        self.conv2 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)
        self.conv3 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)

        self.act1 = torch.nn.LeakyReLU(inplace=True)
        self.act2 = torch.nn.LeakyReLU(inplace=True)
        self.act3 = torch.nn.LeakyReLU(inplace=True)


    def forward(self, x):
        
        out = self.conv1(self.act1(x))
        out = out + x
        
        out1 = self.conv2(self.act2(out))
        out1 = x + out1 + out
        
        out2 = self.conv3(self.act3(out1))
        out2 = out2 + out1 + out + x

        return out2
    
## Data Augmentation ########################################################

def centre_crop(x): 
    left = int(im.size[0]/2-256/2)
    upper = int(im.size[1]/2-256/2)
    right = left + 256
    lower = upper + 256
    x.crop((left, upper,right,lower))
    

def rotate(x):
  # c, h, w = x.shape
  # x += np.random.randn(c, h, w) * 0.15
  # x = Image.fromarray(x)
  # x= x.rotate(125)
  x=np.rot90(x, k=1, axes=(0, 1))
  return np.rot90(x, k=1, axes=(0, 1))


def noise(x): 
  c, h, w = x.shape
  x += np.random.randn(c, h, w) * 0.15
  return x

def shift_horizontal_vertical(x):
  x = np.roll(x, 1, axis=0) # shift 1 place in horizontal axis
  x = np.roll(x, 1, axis=1) # shift 1 place in vertical axis
  return x

def shift_horizontal(x):
  x = np.roll(x, 1, axis=0) # shift 1 place in horizontal axis
  return x

def shift_vertical(x):
  x = np.roll(x, 1, axis=1) # shift 1 place in vertical axis
  return x

def gaussian_blur(x):
  return gaussian_filter(x, sigma=2)

def flip_channel(x):
  beta = np.random.beta(0.5, 0.5)
  if beta>1-beta:
    mix=beta
  else:
    mix=1-beta
  xmix = x * mix + x[::-1] * (1 - mix)
  return xmix

def flip (y):
  z=[0.8*y[:,:,0][: :-1],0.7*y[:,:,1][: :-1],0.9*y[:,:,2][: :-1]]
  z=np.asarray(z)
  z=np.moveaxis(z,0,-1)
  return z


def random_crop(x):
#   x=np.moveaxis(x,1,-1)
#   x=np.moveaxis(x,1,-1)
  z = x[random.randint(0,int(x.shape[0]/2)-10):random.randint(int(x.shape[0]/2)+10,x.shape[0]),random.randint(0,int(x.shape[0]/2)-10):random.randint(int(x.shape[0]/2)+10,x.shape[0]),:]
  z1 = cv2.resize(z[:,:,0], (x.shape[0],x.shape[0]), interpolation = cv2.INTER_AREA)
  z2 = cv2.resize(z[:,:,0], (x.shape[0],x.shape[0]), interpolation = cv2.INTER_AREA)
  z3 = cv2.resize(z[:,:,0], (x.shape[0],x.shape[0]), interpolation = cv2.INTER_AREA)
  z=[z1,z2,z3]
  z=np.asarray(z)
  z=np.moveaxis(z,0,-1)
  return z

def zooming(x):
  z = x[int(x.shape[0]/4):x.shape[0]-int(x.shape[0]/4),int(x.shape[0]/4):x.shape[0]-int(x.shape[0]/4),:]
  z=np.asarray(z)
  print(z.shape)
  z=np.kron(z, np.ones((2,2,1)))
  return z


def brightness(a):
  a = a.astype(int)
  min=np.min(a)        # result=144
  max=np.max(a)        # result=216
  LUT=np.zeros(256,dtype=np.uint8)
  LUT[min:max+1]=np.linspace(start=0,stop=255,num=(max-min)+1,endpoint=True,dtype=np.uint8)
  return LUT[a]
 




def normalize(data):
    size=data[0].shape[0]*data[0].shape[1]*data[0].shape[2]
    for i in range (len(data)):
        x=data[i].reshape(1,size).tolist()
        data[i]=(data[i]-min(x[0]))/(max(x[0])-min(x[0]))
    return data

#############################################################################################


def process_and_train_load_data():
    train_y = []
    train_x = []
    train_yy= load_images_from_folder('/home/harsh.shukla/SRCNN/Flickr/train/y')
    
    train_y=[i for i in train_yy]
    train_xx= load_images_from_folder('/home/harsh.shukla/SRCNN/Flickr/train/x')
    train_x=[i for i in train_xx]
#     print(len(train_x))
#     c=0
#     for i in train_yy :
#         x=shift_horizontal(i)
#         small_array = cv2.resize(x, (128,128))
#         train_y.append(x)
#         train_x.append(small_array)
#         x=rotate(i)
#         small_array = cv2.resize(x, (128,128))
#         train_y.append(x)
#         train_x.append(small_array)
#         print(c)
#         c=c+1
#     print(len(train_y))
    train_target=np.asarray(train_y)
#     print(train_target.shape)
    train_target=np.moveaxis(train_target,1,-1)
#     print(train_target.shape)
    train_target=np.moveaxis(train_target,1,-1)
#     print(train_target.shape)
    train_target = train_target.astype(np.float32)
#     print(train_target.shape)
    
#     print(len(train_x))
    train_input=np.asarray(train_x)
#     print(train_input.shape)
    train_input=np.moveaxis(train_input,1,-1)
#     print(train_input.shape)
    train_input=np.moveaxis(train_input,1,-1)
#     print(train_input.shape)
    train_input = train_input.astype(np.float32)
#     print(train_input.shape)

    test= load_images_from_folder('/home/harsh.shukla/SRCNN/Flickr/test/x')
    test_input=np.asarray(test)
    test_input=np.moveaxis(test_input,1,-1)
    test_input=np.moveaxis(test_input,1,-1)
    test_input = test_input.astype(np.float32)

    test= load_images_from_folder('/home/harsh.shukla/SRCNN/Flickr/test/y')
    test_target=np.asarray(test)
    test_target=np.moveaxis(test_target,1,-1)
    test_target=np.moveaxis(test_target,1,-1)
    test_target = test_target.astype(np.float32)
    data_train=[]
    data_test_flickr=[]
    data_test_div=[]
    for input, target in zip(train_input, train_target):
        data_train.append([input, target])
    for input, target in zip(test_input, test_target):
        data_test_flickr.append([input, target])
    
    
    
    test= load_images_from_folder('/home/harsh.shukla/SRCNN/Div2K_data/test/x')
    test_input=np.asarray(test)
    test_input=np.moveaxis(test_input,1,-1)
    test_input=np.moveaxis(test_input,1,-1)
    test_input = test_input.astype(np.float32)

    test= load_images_from_folder('/home/harsh.shukla/SRCNN/Div2K_data/test/y')
    test_target=np.asarray(test)
    test_target=np.moveaxis(test_target,1,-1)
    test_target=np.moveaxis(test_target,1,-1)
    test_target = test_target.astype(np.float32)
    for input, target in zip(test_input, test_target):
        data_test_div.append([input, target])
    
    trainloader=torch.utils.data.DataLoader(dataset=data_train, batch_size=48, shuffle=True)
    testloader_flickr=torch.utils.data.DataLoader(dataset=data_test_flickr, batch_size=48, shuffle=True)
    testloader_div=torch.utils.data.DataLoader(dataset=data_test_div, batch_size=48, shuffle=True)
    
    
    return trainloader, testloader_flickr,testloader_div

def calculate_mean_std_dataset(loader):
    mean = 0.
    std = 0.
    nb_samples = 0.
    for data in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    
    
def initialize_train_network(trainloader, testloader_flickr,testloader_div, debug): 
    
    results = "/home/harsh.shukla/SRCNN/SRSN_results"
    
    if not os.path.exists(results):
        os.makedirs(results)
        
    # Initialising Checkpointing directory 
    checkpoints = os.path.join(results,"Checkpoints")
    if not os.path.exists(checkpoints):
        os.makedirs(checkpoints)
    checkpoint_file = os.path.join(checkpoints,"check.pt")  
    
    # Initialising directory for Network Debugging
    net_debug = os.path.join(results,"Debug")
    if not os.path.exists(net_debug):
        os.makedirs(net_debug)

    model = SRSN_RRDB()
    model = nn.DataParallel(model)
    model.to(device)
    print(next(model.parameters()).device)
    model.apply(weight_init) ## Weight initialisation 
    
    
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)
#     optimizer  = optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
    my_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
#     criterion = nn.MSELoss().to(device)
    test_criterion = nn.MSELoss().to(device)
    criterion=nn.L1Loss().to(device)
    
    # load model if exists
    if os.path.exists(checkpoint_file):
        print("Loading from Previous Checkpoint...")
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.train()
    else:
        print("No previous checkpoints exist, initialising network from start...")
        
     ## Parameters in Networks
    print("Number of Parameters in Super Resolution Network")
    count_parameters(model)
    
    best_loss=10000
    train=[]
    test=[]
    psnr_div=[]
    psnr_flickr=[]
    if os.path.exists((os.path.join(results,"Train_loss.txt"))):
        dbfile = open(os.path.join(results,"Train_loss.txt"), 'rb')      
        train = pickle.load(dbfile)
        dbfile = open(os.path.join(results,"Test_loss.txt"), 'rb')      
        test = pickle.load(dbfile)
        dbfile = open(os.path.join(results,"PSNR_flickr.txt"), 'rb')      
        psnr_flickr = pickle.load(dbfile)
        dbfile = open(os.path.join(results,"PSNR_div.txt"), 'rb')      
        psnr_div = pickle.load(dbfile)
    loss1=0
    for epoch in range(80):
        training_loss=[]
        test_loss_flickr=[]
        test_loss_div=[]
        list_no=0
        for input_,target in trainloader:
            if torch.cuda.is_available():
                input_ = input_.to(device)
                target=target.to(device)
            output = model(input_)
            loss=criterion(output, target)
            loss.backward()
            optimizer.step()
            if debug == True: 
                plot_grad_flow(net_debug,model.named_parameters(),"super_resolution_network")
            optimizer.zero_grad()
            training_loss.append(loss.item())
        
        with torch.set_grad_enabled(False):
            for local_batch, local_labels in testloader_div:
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)
                output = model(local_batch).to(device)
                local_labels.require_grad = False
                test_loss_div.append(test_criterion(output, local_labels).item())
                
            for local_batch, local_labels in testloader_flickr:
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)
                output = model(local_batch).to(device)
                local_labels.require_grad = False
                test_loss_flickr.append(test_criterion(output, local_labels).item())
#                 if debug == True: 
#                     visualise_layer_activation(model,local_batch,net_debug)

        my_lr_scheduler.step(test_loss_flickr[-1])
        torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_file)
        
        if(debug == True):
            label=im.fromarray(np.uint8(np.moveaxis((local_labels[0]).cpu().detach().numpy(),0,-1))).convert('RGB')
            output=im.fromarray(np.uint8(np.moveaxis((output[0].cpu()).detach().numpy(),0,-1))).convert('RGB')
            label.save(os.path.join(results,str(epoch) + 'test_target' + '.png'))
            output.save(os.path.join(results,str(epoch) + 'test_output' + '.png'))
  
        train.append(sum(training_loss)/len(training_loss))
        test.append(sum(test_loss_flickr)/len(test_loss_flickr))
        psnr_flickr.append(10*math.log10(255*255/(sum(test_loss_flickr)/len(test_loss_flickr))))
        psnr_div.append(10*math.log10(255*255/(sum(test_loss_div)/len(test_loss_div))))
        with open(os.path.join(results,"Train_loss.txt"), 'wb') as f:
                pickle.dump(train ,f)
        with open(os.path.join(results,"Test_loss.txt"), 'wb') as f:
             pickle.dump(test,f )
        with open(os.path.join(results,"PSNR_flickr.txt"), 'wb') as f:
             pickle.dump(psnr_flickr,f )
        with open(os.path.join(results,"PSNR_div.txt"), 'wb') as f:
             pickle.dump(psnr_div,f )
        print("Epoch :",epoch, flush=True)
        print("Training loss :",sum(training_loss)/len(training_loss),flush=True)
        print("Test loss for Flickr:",sum(test_loss_flickr)/len(test_loss_flickr),flush=True)
        print("Test loss for Div:",sum(test_loss_div)/len(test_loss_div),flush=True)
        print("PSNR for Flickr :", 10*math.log10(255*255/(sum(test_loss_flickr)/len(test_loss_flickr))))
        print("PSNR for Div :", 10*math.log10(255*255/(sum(test_loss_div)/len(test_loss_div))))

        print("-----------------------------------------------------------------------------------------------------------")
    try:
        file = open(os.path.join(results,"SR_train_loss.txt"), 'w+')
        try:
            for i in range(len(test)):
                file.write(str(train[i]) + ","  + str(test[i]))
                file.write('\n')
        finally:
            file.close()
    except IOError:
        print("Unable to create loss file")
    print("---------------------------------------------------------------------------------------------------------------")
    print("Training Completed")    

# def initialize_train_network(trainloader, testloader, debug): 
    
#     results = "/home/harsh.shukla/SRCNN/SRSN_results"
    
#     if not os.path.exists(results):
#         os.makedirs(results)
        
#     # Initialising Checkpointing directory 
#     checkpoints = os.path.join(results,"Checkpoints")
#     if not os.path.exists(checkpoints):
#         os.makedirs(checkpoints)
#     checkpoint_file = os.path.join(checkpoints,"check.pt")  
    
#     # Initialising directory for Network Debugging
#     net_debug = os.path.join(results,"Debug")
#     if not os.path.exists(net_debug):
#         os.makedirs(net_debug)

#     model = SRSN_RRDB()
#     model = nn.DataParallel(model)
#     model.to(device)
#     print(next(model.parameters()).device)
#     model.apply(weight_init) ## Weight initialisation 
    
    
    
#     optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)
# #     optimizer  = optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
#     my_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
# #     criterion = nn.MSELoss().to(device)
#     test_criterion = nn.MSELoss().to(device)
#     criterion=nn.L1Loss().to(device)
    
#     # load model if exists
#     if os.path.exists(checkpoint_file):
#         print("Loading from Previous Checkpoint...")
#         checkpoint = torch.load(checkpoint_file)
#         model.load_state_dict(checkpoint['model_state_dict'])
#         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         model.train()
#     else:
#         print("No previous checkpoints exist, initialising network from start...")
        
#      ## Parameters in Networks
#     print("Number of Parameters in Super Resolution Network")
#     count_parameters(model)
    
#     best_loss=10000
#     train=[]
#     test=[]
#     psnr=[]
#     if os.path.exists((os.path.join(results,"Train_loss.txt"))):
#         dbfile = open(os.path.join(results,"Train_loss.txt"), 'rb')      
#         train = pickle.load(dbfile)
#         dbfile = open(os.path.join(results,"Test_loss.txt"), 'rb')      
#         test = pickle.load(dbfile)
#         dbfile = open(os.path.join(results,"PSNR.txt"), 'rb')      
#         psnr = pickle.load(dbfile)
#     loss1=0
#     for epoch in range(80):
#         training_loss=[]
#         test_loss=[]
#         list_no=0
#         for input_,target in trainloader:
#             if torch.cuda.is_available():
#                 input_ = input_.to(device)
#                 target=target.to(device)
#             output = model(input_)
#             loss=criterion(output, target)
#             loss.backward()
#             optimizer.step()
#             if debug == True: 
#                 plot_grad_flow(net_debug,model.named_parameters(),"super_resolution_network")
#             optimizer.zero_grad()
#             training_loss.append(loss.item())
        
#         with torch.set_grad_enabled(False):
#             for local_batch, local_labels in testloader:
#                 local_batch, local_labels = local_batch.to(device), local_labels.to(device)
#                 output = model(local_batch).to(device)
#                 local_labels.require_grad = False
#                 test_loss.append(test_criterion(output, local_labels).item())
# #                 if debug == True: 
# #                     visualise_layer_activation(model,local_batch,net_debug)

#         my_lr_scheduler.step(test_loss[-1])
#         torch.save({
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 }, checkpoint_file)
        
#         if(debug == True):
#             label=im.fromarray(np.uint8(np.moveaxis((local_labels[0]).cpu().detach().numpy(),0,-1))).convert('RGB')
#             output=im.fromarray(np.uint8(np.moveaxis((output[0].cpu()).detach().numpy(),0,-1))).convert('RGB')
#             label.save(os.path.join(results,str(epoch) + 'test_target' + '.png'))
#             output.save(os.path.join(results,str(epoch) + 'test_output' + '.png'))
  
#         train.append(sum(training_loss)/len(training_loss))
#         test.append(sum(test_loss)/len(test_loss))
#         psnr.append(10*math.log10(255*255/(sum(test_loss)/len(test_loss))))
#         with open(os.path.join(results,"Train_loss.txt"), 'wb') as f:
#                 pickle.dump(train ,f)
#         with open(os.path.join(results,"Test_loss.txt"), 'wb') as f:
#              pickle.dump(test,f )
#         with open(os.path.join(results,"PSNR.txt"), 'wb') as f:
#              pickle.dump(psnr,f )
#         print("Epoch :",epoch, flush=True)
#         print("Training loss :",sum(training_loss)/len(training_loss),flush=True)
#         print("Test loss :",sum(test_loss)/len(test_loss),flush=True)
#         print("PSNR :", 10*math.log10(255*255/(sum(test_loss)/len(test_loss))))

#         print("-----------------------------------------------------------------------------------------------------------")
#     try:
#         file = open(os.path.join(results,"SR_train_loss.txt"), 'w+')
#         try:
#             for i in range(len(test)):
#                 file.write(str(train[i]) + ","  + str(test[i]))
#                 file.write('\n')
#         finally:
#             file.close()
#     except IOError:
#         print("Unable to create loss file")
#     print("---------------------------------------------------------------------------------------------------------------")
#     print("Training Completed")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--debug', help="Mode of Execution here")
    args = parser.parse_args()
    
    grad_flow_flag = False


    if args.debug == "debug": 
        print("Running in Debug Mode.....")
        grad_flow_flag = True

    trainloader, testloader_flickr,testloader_div = process_and_train_load_data()
    
    initialize_train_network(trainloader, testloader_flickr,testloader_div,grad_flow_flag)
    
    
    
    
    
    
    
    