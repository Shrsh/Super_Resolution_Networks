import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
# from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
# from sklearn import svm
# import sklearn.model_selection
# import sklearn.metrics
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import precision_recall_fscore_support
# from sklearn.ensemble import VotingClassifier
import math 
# import tensorflow as tf
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
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device_ids = [i for i in range(torch.cuda.device_count())]
device = 'cuda' if use_cuda else 'cpu'
torch.backends.cudnn.benchmark = True
from prettytable import PrettyTable
import torch.nn.init as init


#custom Initializer
from initializer import kaiming_normal_


####### Initialisation 
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
        kaiming_normal_(m.weight.data, mode='fan_in',nonlinearity='leaky_relu')
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

#########################################################################################################################################################################################################




### Data Preparation ############################################
################################################################

def normalize(data):
    size=data[0].shape[0]*data[0].shape[1]*data[0].shape[2]
    for i in range (len(data)):
        x=data[i].reshape(1,size).tolist()
        data[i]=(data[i]-min(x[0]))/(max(x[0])-min(x[0]))
    return data

def calculate_mean_std_dataset(loader):
    mean_d = 0.
    std_d = 0.
    mean_l = 0. 
    std_l = 0.
    nb_samples = 0.
    for data,label in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean_d += data.mean(2).sum(0)
        std_d += data.std(2).sum(0)
        nb_samples += batch_samples
        
        label = label.view(batch_samples, label.size(1), -1)
        mean_l += label.mean(2).sum(0)
        std_l += label.std(2).sum(0)

    mean_d /= nb_samples
    std_d /= nb_samples
    
    mean_l /= nb_samples 
    std_l /= nb_samples
    print("Data Mean: ",mean_d)
    print("Data Std: ",std_d)
    print("Data Mean: ",mean_l)
    print("Data Std: ",std_l)
    return mean_d, std_d, mean_l, std_l
    
def load_images_from_folder(folder):
    c=0
    images = []
    list_name=[]
#     list_name1=[]
    for filename in os.listdir(folder):
        list_name.append(os.path.join(folder,filename))
#         list_name1.append(filename)
#     list_name1.sort()
    list_name.sort()
#     print(list_name1)
    for filename in list_name:
        img = cv2.imread(filename)
        if img is not None:
            images.append(img)
#             print(c)
#             c=c+1
        # if c==8
    return images

def process_and_train_load_data():
    train_y = []
    train_x = []
    data_train=[]
#     print("Before Train_y")
    train_yy= load_images_from_folder('/home/harsh.shukla/SRCNN/SR_data/train/y')
#     print("After Train_y")
    for i in train_yy:
        train_y.append(i)
#         print("Yes")
#     print("Before Train_x")
    train_xx= load_images_from_folder('/home/harsh.shukla/SRCNN/SR_data/train/x')
#     print("After Train_x")
    for i in train_xx:
        train_x.append(i)

    
#     train_yy= load_images_from_folder('/home/harsh.shukla/SRCNN/training_test_data/Div2K_data/train/y')
#     for i in train_yy:
#         train_y.append(i)
#     train_xx= load_images_from_folder('/home/harsh.shukla/SRCNN/training_test_data/Div2K_data/train/x')
#     for i in train_xx:
#         train_x.append(i)
    
#     train_yy= load_images_from_folder('/home/harsh.shukla/SRCNN/training_test_data/Urban100/train/y')
#     for i in train_yy:
#         train_y.append(i)
#     train_xx= load_images_from_folder('/home/harsh.shukla/SRCNN/training_test_data/Urban100/train/x')
#     for i in train_xx:
#         train_x.append(i)
    
    print("size of training set:", len(train_y))

#     for i in train_yy :
#         x=shift_horizontal(i)
#         small_array = cv2.resize(x, (128,128))
#         train_y.append(x)
#         train_x.append(small_array)
#         x=rotate(i)
#         small_array = cv2.resize(x, (128,128))
#         train_y.append(x)
#         train_x.append(small_array)

    train_target=np.asarray(train_y)
    train_target=np.moveaxis(train_target,1,-1)
    train_target=np.moveaxis(train_target,1,-1)
    train_target = train_target.astype(np.float32)
    train_input=np.asarray(train_x)
    train_input=np.moveaxis(train_input,1,-1)
    train_input=np.moveaxis(train_input,1,-1)
    train_input = train_input.astype(np.float32)
    for input, target in zip(train_input, train_target):
        data_train.append([input, target])

    test= load_images_from_folder('/home/harsh.shukla/SRCNN/SR_data_512/test/x')
    test_input=np.asarray(test)
    test_input=np.moveaxis(test_input,1,-1)
    test_input=np.moveaxis(test_input,1,-1)
    test_input = test_input.astype(np.float32)

    test= load_images_from_folder('/home/harsh.shukla/SRCNN/SR_data_512/test/y')
    test_target=np.asarray(test)
    test_target=np.moveaxis(test_target,1,-1)
    test_target=np.moveaxis(test_target,1,-1)
    test_target = test_target.astype(np.float32)
    data_test_flickr=[]
    data_test_div=[]
    data_test_urban=[]
    for input, target in zip(test_input, test_target):
        data_test_flickr.append([input, target])
    
    
    
    test= load_images_from_folder('/home/harsh.shukla/SRCNN/training_test_data/Div2K_data/test/x')
    test_input=np.asarray(test)
    test_input=np.moveaxis(test_input,1,-1)
    test_input=np.moveaxis(test_input,1,-1)
    test_input = test_input.astype(np.float32)

    test= load_images_from_folder('/home/harsh.shukla/SRCNN/training_test_data/Div2K_data/test/y')
    test_target=np.asarray(test)
    test_target=np.moveaxis(test_target,1,-1)
    test_target=np.moveaxis(test_target,1,-1)
    test_target = test_target.astype(np.float32)

    for input, target in zip(test_input, test_target):
        data_test_div.append([input, target])
        
    test= load_images_from_folder('/home/harsh.shukla/SRCNN/SR_data_256/test/x')
    test_input=np.asarray(test)
    test_input=np.moveaxis(test_input,1,-1)
    test_input=np.moveaxis(test_input,1,-1)
    test_input = test_input.astype(np.float32)

    test= load_images_from_folder('/home/harsh.shukla/SRCNN/SR_data_256/test/y')
    test_target=np.asarray(test)
    test_target=np.moveaxis(test_target,1,-1)
    test_target=np.moveaxis(test_target,1,-1)
    test_target = test_target.astype(np.float32)

    for input, target in zip(test_input, test_target):
        data_test_urban.append([input, target])
        
    
    trainloader=torch.utils.data.DataLoader(dataset=data_train, batch_size=16, shuffle=True)
    testloader_flickr=torch.utils.data.DataLoader(dataset=data_test_flickr, batch_size=16, shuffle=True)
    testloader_div=torch.utils.data.DataLoader(dataset=data_test_div, batch_size=16, shuffle=True)
    testloader_urban=torch.utils.data.DataLoader(dataset=data_test_urban, batch_size=16, shuffle=True)
    
#     calculate_mean_std_dataset(trainloader)
#     calculate_mean_std_dataset(testloader_flickr)
#     calculate_mean_std_dataset(testloader_div)
#     calculate_mean_std_dataset(testloader_urban)
    
    return trainloader, testloader_flickr,testloader_div,testloader_urban

class Normalize:
    """Applies the :class:`~torchvision.transforms.Normalize` transform to a batch of images.
    .. note::
        This transform acts out of place by default, i.e., it does not mutate the input tensor.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.
        dtype (torch.dtype,optional): The data type of tensors to which the transform will be applied.
        device (torch.device,optional): The device of tensors to which the transform will be applied.
    """

    def __init__(self, mean, std, inplace=False, dtype=torch.float, device='cpu'):
        self.mean = torch.as_tensor(mean, dtype=dtype, device=device)[None, :, None, None]
        self.std = torch.as_tensor(std, dtype=dtype, device=device)[None, :, None, None]
        self.inplace = inplace
        
    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor.
        """
        if not self.inplace:
            tensor = tensor.clone()

        tensor.sub_(self.mean).div_(self.std)
        return tensor
    
class ToTensor:
    """Applies the :class:`~torchvision.transforms.ToTensor` transform to a batch of images.
    """

    def __init__(self):
        self.max = 255
        
    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be tensorized.
        Returns:
            Tensor: Tensorized Tensor.
        """
        return tensor.float().div_(self.max)
################################################################

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
                 feature_layer=36,
                 use_bn=False,
                 device=torch.device('cuda')):
        super(VGGFeatureExtractor, self).__init__()
        if use_bn:
            model = torchvision.models.vgg19_bn(pretrained=True)
        else:
            model = torchvision.models.vgg19(pretrained=True)
        self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.features(x)
    
class SRSN_RRDB(nn.Module):
    def __init__(self, input_dim=3, dim=128, scale_factor=4,scale_ratio=0.2):
        super(SRSN_RRDB, self).__init__()
#         self.up = nn.ConvTranspose2d(3,3, 1, stride=1,output_size=(512,512))
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
        
#         self.act1 = nn.PReLU()
#         self.act2 = nn.PReLU()
        
#         self.bn = torch.nn.BatchNorm2d(3)
        
        self.scale_ratio = 1

    def forward(self, LR):
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
        out6 = self.up(out6)
        #LR_feat = self.resnet(out3)
        SR=F.leaky_relu(self.conv3(out6),negative_slope=0.2)
        SR =self.conv4(SR)
        # print(SR.shape)
        return SR   
    
    
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
    

    

def train_discriminator(optimizer, real_data, fake_data,discriminator,b_loss):
    
    optimizer.zero_grad()
    prediction_real = discriminator(real_data.detach())
    prediction_fake = discriminator(fake_data.detach())
    
    # 1. Train on Real Data
    truth_real=Variable(torch.ones(real_data.size(0), 1))-0.1
#     truth_real=Variable(torch.Tensor(real_data.size(0), 1).fill_(0.9).type(dtype))
    error_real = b_loss(prediction_real.to(device), truth_real.to(device))*0 
    error_real.mean().backward(retain_graph=True)

    # 2. Train on Fake Data
    error_fake = b_loss(prediction_fake .to(device), Variable(torch.zeros(real_data.size(0), 1)).to(device))*0
    error_fake.mean().backward()
    optimizer.step()

    return error_real + error_fake, prediction_real, prediction_fake



def train_generator(model, optimizer, fake_data,real_data,discriminator,b_loss,m_loss,vgg_features_high):
    optimizer.zero_grad()
    
    lambda_ = 0.0
    
    ##Reconstruction loss
    loss=m_loss(fake_data, real_data)
    ## Adversarial Loss 
    prediction = discriminator(fake_data)
    error = b_loss(prediction, Variable(torch.ones(real_data.size(0), 1)).to(device))
    ## Perceptual 
    features_gt=vgg_features_high(real_data)
    features_out=vgg_features_high(fake_data)
    loss_perceptual=m_loss(features_gt, features_out)
    total_loss = loss + lambda_*error + loss_perceptual
    total_loss.mean().backward()
    optimizer.step()
    return loss,error,total_loss,loss_perceptual



def train_network(trainloader, testloader_flickr,testloader_div,testloader_urban, debug,num_epochs=200,K=1):
    discriminator = DiscriminativeNet()
    model=SRSN_RRDB()
    model = nn.DataParallel(model, device_ids = device_ids)
    discriminator = nn.DataParallel(discriminator, device_ids= device_ids)
    model = model.to(device)
    model.apply(weight_init)
    discriminator=discriminator.to(device)
    vgg_features_high=nn.DataParallel(VGGFeatureExtractor())
    vgg_features_high.to(device)
       
    d_optimizer = optim.SGD(discriminator.parameters(), lr=0.000001, momentum=0.9)
    g_optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.9, 0.999), eps=1e-8)
    
    Mse_loss = nn.DataParallel(nn.MSELoss(),device_ids = device_ids).to(device)
    Bce_loss = nn.DataParallel(nn.BCEWithLogitsLoss(),device_ids = device_ids).to(device)
    criterion = nn.DataParallel(nn.SmoothL1Loss(),device_ids = device_ids).to(device)

    train_d=[]
    train_g = []
    
    train_g_rec=[]
    train_g_dis=[]
    test=[]
    psnr_div=[]
    psnr_flickr=[]
    psnr_urban=[]
    train_rmse=[]
    train_psnr=[]
    Disriminator_Loss=[]
    Generator_Adver_Loss=[]
    ## Parameters in Networks
    print("Number of Parameters in Generator")
    count_parameters(model)
    print("Number of Parameters in Discriminator")
    count_parameters(discriminator)
    
    results = "/home/harsh.shukla/SRCNN/GAN_results"
    
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
    
   
    # load model if exists
    if os.path.exists(checkpoint_file):
        print("Loading from Previous Checkpoint...")
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        g_optimizer.load_state_dict(checkpoint['g_state_dict'])
        d_optimizer.load_state_dict(checkpoint['d_state_dict'])   
        model.train()
        discriminator.train()
    else:
        print("No previous checkpoints exist, initialising network from start...")
                
                
    if os.path.exists((os.path.join(results,"PSNR_flickr.txt"))):
        dbfile = open(os.path.join(results,"PSNR_flickr.txt"), 'rb')      
        psnr_flickr = pickle.load(dbfile)
        dbfile = open(os.path.join(results,"PSNR_div.txt"), 'rb')      
        psnr_div = pickle.load(dbfile)
        dbfile = open(os.path.join(results,"PSNR_bsd.txt"), 'rb')      
        psnr_urban = pickle.load(dbfile)
        dbfile = open(os.path.join(results,"Train.txt"), 'rb')      
        train_psnr = pickle.load(dbfile)  
        
        dbfile = open(os.path.join(results,"Discriminator.txt"), 'rb')      
        Disriminator_Loss = pickle.load(dbfile)
        dbfile = open(os.path.join(results,"Generator_Adversial.txt"), 'rb')      
        Generator_Adver_Loss = pickle.load(dbfile)
        

    for epoch in range(num_epochs):
        training_loss_d=[]
        training_rec_loss_g=[]
        training_dis_loss_g=[]
        training_loss_g = []
        training_loss_perp = []
        test_loss_flickr=[]
        test_loss_div=[]
        test_loss_urban=[]
        train_rec_rmse=[]
        count = 0
        list_no=0
        for input_,real_data in trainloader:
            if torch.cuda.is_available():
                input_    = input_.to(device)
                real_data = real_data.to(device)
            fake_data = model(input_).to(device)
            if count == K:
                d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer, real_data, fake_data,discriminator,Bce_loss)
                training_loss_d.append(d_error.mean().item())
                count = 0
            g_rec_error,g_dis_error,g_error,l_percp = train_generator(model,g_optimizer, fake_data, real_data, discriminator, Bce_loss, criterion,vgg_features_high)
            training_rec_loss_g.append(g_rec_error.mean().item())
            training_dis_loss_g.append(g_dis_error.mean().item())
            training_loss_g.append(g_error.mean().item())
            training_loss_perp.append(l_percp.mean().item())
            train_rec_rmse.append((Mse_loss(fake_data, real_data)).mean().item())
            count += 1 

        #Plotting Gradient Flow for both the models
        if(debug == True): 
            plot_grad_flow(net_debug,model.named_parameters(),"generator")
            plot_grad_flow(net_debug,discriminator.named_parameters(),"discriminator")
        
        with torch.set_grad_enabled(False):
            for local_batch, local_labels in testloader_urban:
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)
                output = model(local_batch).to(device)
                local_labels.require_grad = False
                test_loss_urban.append(Mse_loss(output, local_labels).mean().item())
                
            for local_batch, local_labels in testloader_div:
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)
                output = model(local_batch).to(device)
                local_labels.require_grad = False
                test_loss_div.append(Mse_loss(output, local_labels).mean().item())
                
            for local_batch, local_labels in testloader_flickr:
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)
                output = model(local_batch).to(device)
                local_labels.require_grad = False
                test_loss_flickr.append(Mse_loss(output, local_labels).mean().item())
                
#         if debug == True:
#             label=im.fromarray(np.uint8(np.moveaxis(local_labels[0].cpu().detach().numpy(),0,-1))).convert('RGB')
#             output=im.fromarray(np.uint8(np.moveaxis(output[0].cpu().detach().numpy(),0,-1))).convert('RGB')
#             label.save(os.path.join(results,str(epoch) + 'test_target' + '.png'))
#             output.save(os.path.join(results,str(epoch) + 'test_output' + '.png'))

        ##Creating Checkpoints
        if epoch % 1 == 0:
            torch.save({
                'generator_state_dict': model.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'g_state_dict': g_optimizer.state_dict(),
                'd_state_dict': d_optimizer.state_dict(),
                }, checkpoint_file)
        print("Epoch :",epoch )
        print("Discriminator Loss :",sum(training_loss_d)/len(training_loss_d))
        print("Generator Reconstruction Loss :",sum(training_rec_loss_g)/len(training_rec_loss_g))
        print("Generator Adversarial Loss :",sum(training_dis_loss_g)/len(training_dis_loss_g))
        print("Perceptual Loss :", sum(training_loss_perp)/len(training_loss_perp))
        print("Total Generator Loss:",sum(training_loss_g)/len(training_loss_g))
        print("D(X) :",d_pred_real.mean(), "D(G(X)) :",d_pred_fake.mean())
        Disriminator_Loss.append(sum(training_loss_d)/len(training_loss_d))
        Generator_Adver_Loss.append(sum(training_dis_loss_g)/len(training_dis_loss_g))
        
        print(len(test_loss_flickr),max(test_loss_flickr))
        psnr_flickr.append(10*math.log10(255*255/(sum(test_loss_flickr)/len(test_loss_flickr))))
        psnr_div.append(10*math.log10(255*255/(sum(test_loss_div)/len(test_loss_div))))
        psnr_urban.append(10*math.log10(255*255/(sum(test_loss_urban)/len(test_loss_urban))))
        train_rmse.append(sum(train_rec_rmse)/len(train_rec_rmse))
        train_psnr.append(10*math.log10(255*255/train_rmse[-1]))
        with open(os.path.join(results,"PSNR_flickr.txt"), 'wb') as f:
             pickle.dump(psnr_flickr,f )
        with open(os.path.join(results,"PSNR_div.txt"), 'wb') as f:
             pickle.dump(psnr_div,f )
        with open(os.path.join(results,"PSNR_bsd.txt"), 'wb') as f:
             pickle.dump(psnr_urban,f )
        with open(os.path.join(results,"Train.txt"), 'wb') as f:
             pickle.dump(train_psnr,f )
       
        with open(os.path.join(results,"Discriminator.txt"), 'wb') as f:
             pickle.dump(Disriminator_Loss,f )
        with open(os.path.join(results,"Generator_Adversial.txt"), 'wb') as f:
             pickle.dump(Generator_Adver_Loss,f )
#         print("Epoch :",epoch, flush=True)

        print("Training Rmse loss : ",train_rmse[-1])
        print("Training PSNR : ",train_psnr[-1])
        print("Test loss for Flickr:",sum(test_loss_flickr)/len(test_loss_flickr),flush=True)
        print("Test loss for Div:",sum(test_loss_div)/len(test_loss_div),flush=True)
        print("Test loss for Urban:",sum(test_loss_urban)/len(test_loss_urban),flush=True)
        print("PSNR for Flickr :", 10*math.log10(255*255/(sum(test_loss_flickr)/len(test_loss_flickr))))
        print("PSNR for Div :", 10*math.log10(255*255/(sum(test_loss_div)/len(test_loss_div))))
        print("PSNR for Urban100 :", 10*math.log10(255*255/(sum(test_loss_urban)/len(test_loss_urban))))  
        print("-----------------------------------------------------------------------------------------------------------")
    try:
        file = open(os.path.join(results,"GAN_train_loss.txt"), 'w+')
        try:
            for i in range(len(test)):
                file.write(str(train_d[i]) + ","  + str(train_g[i]) + "," + str(test[i]))
                file.write('\n')
        finally:
            file.close()
    except IOError:
        print("Unable to create loss file")
    print("---------------------------------------------------------------------------------------------------------------")
    print("Training Completed")
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--debug', help="Mode of Execution here")
    args = parser.parse_args()
    
    grad_flow_flag = False


    if args.debug == "debug": 
        print("Running in Debug Mode.....")
        grad_flow_flag = True
    
    
    
    trainloader, testloader_flickr,testloader_div,testloader_urban = process_and_train_load_data()
    print("Initialised Data Loader ....")
    train_network(trainloader, testloader_flickr,testloader_div,testloader_urban,grad_flow_flag)
    

