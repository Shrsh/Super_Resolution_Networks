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
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
device_ids = [i for i in range(torch.cuda.device_count())]
device = 'cuda' if use_cuda else 'cpu'
torch.backends.cudnn.benchmark = True

trans = transforms.ToPILImage()
trans1 = transforms.ToTensor()
torch.autograd.set_detect_anomaly(True)

# Pre-Computed Data Statistics    
mean_train = [ 97.2139, 111.3607, 123.2277]
std_train = [45.4368, 46.1355, 48.7539]
mean_test = [ 99.2980, 111.7384, 117.2509] 
std_test = [47.4708, 47.9140, 49.1483]


####
####Network 
####
class Generator(nn.Module):
    def __init__(self, num_rrdb_blocks=4):
        r""" This is an esrgan model defined by the author himself.

        We use two settings for our generator – one of them contains 8 residual blocks, with a capacity similar
        to that of SRGAN and the other is a deeper model with 16/23 RRDB blocks.

        Args:
            num_rrdb_blocks (int): How many residual in residual blocks are combined. (Default: 16).

        Notes:
            Use `num_rrdb_blocks` is 16 for TITAN 2080Ti.
            Use `num_rrdb_blocks` is 23 for Tesla A100.
        """
        super(Generator, self).__init__()

        # First layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

        # 16/23 ResidualInResidualDenseBlock layer
        rrdb_blocks = []
        for _ in range(num_rrdb_blocks):
            rrdb_blocks += [ResidualInResidualDenseBlock(64, 32, 0.2)]
        self.Trunk_RRDB = nn.Sequential(*rrdb_blocks)

        # Second conv layer post residual blocks
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # Upsampling layers
        self.up1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.up2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # Next layer after upper sampling
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        # Final output layer
        self.conv4 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out1 = self.conv1(input)
        trunk = self.Trunk_RRDB(out1)
        out2 = self.conv2(trunk)
        out = torch.add(out1, out2)
        out = F.leaky_relu(self.up1(F.interpolate(out, scale_factor=2, mode="nearest")), 0.2, True)
        out = F.leaky_relu(self.up2(F.interpolate(out, scale_factor=2, mode="nearest")), 0.2, True)
        out = self.conv3(out)
        out = self.conv4(out)
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

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias.data, 0.0)

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
        init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
#         init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')        
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
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
        


## Data Augmentation ########################################################

def centre_crop(x): 
    left = int(im.size[0]/2-256/2)
    upper = int(im.size[1]/2-256/2)
    right = left + 256
    lower = upper + 256
    x.crop((left, upper,right,lower))
    

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
  z=[0.8*y[0,:,:][: :-1],0.7*y[1,:,:][: :-1],0.9*y[2,:,:][: :-1]]
  return np.asarray(z)

def random_crop(x):
  z = x[:,random.randint(0,12):random.randint(20,32),random.randint(0,12):random.randint(18,30)]
  z1 = cv2.resize(z[0,:,:], (32,32), interpolation = cv2.INTER_AREA)
  z2 = cv2.resize(z[0,:,:], (32,32), interpolation = cv2.INTER_AREA)
  z3 = cv2.resize(z[0,:,:], (32,32), interpolation = cv2.INTER_AREA)
  z=[z1,z2,z3]
  return np.asarray(z)

def zooming(x):
  z = x[:,8:24,8:24]
  z=np.kron(z, np.ones((1,2,2)))
  return z


def brightness(a):
  a = a.astype(int)
  min=np.min(a)        # result=144
  max=np.max(a)        # result=216
  LUT=np.zeros(256,dtype=np.uint8)
  LUT[min:max+1]=np.linspace(start=0,stop=255,num=(max-min)+1,endpoint=True,dtype=np.uint8)
  return LUT[a]
 

def rotate(x):
  # c, h, w = x.shape
  # x += np.random.randn(c, h, w) * 0.15
  # x = Image.fromarray(x)
  # x= x.rotate(125)
  x=np.rot90(x, k=1, axes=(0, 1))
  return np.rot90(x, k=1, axes=(0, 1))

#############################################################################################
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



def process_and_train_load_data():
    train_y = []
    train_x = []
    train_yy= load_images_from_folder('/home/harsh.shukla/SRCNN/HR_LR_data/train/y')
    train_y=[i for i in train_yy]
    train_xx= load_images_from_folder('/home/harsh.shukla/SRCNN/HR_LR_data/train/x')
    train_x=[i for i in train_xx]
    
#     for i in train_yy :
#         x=gaussian_blur(i)
#         small_array = cv2.resize(x, (64,64))
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

    test= load_images_from_folder('/home/harsh.shukla/SRCNN/HR_LR_data/test/x')
    test_input=np.asarray(test)
    test_input=np.moveaxis(test_input,1,-1)
    test_input=np.moveaxis(test_input,1,-1)
    test_input = test_input.astype(np.float32)

    test= load_images_from_folder('/home/harsh.shukla/SRCNN/HR_LR_data/test/y')
    test_target=np.asarray(test)
    test_target=np.moveaxis(test_target,1,-1)
    test_target=np.moveaxis(test_target,1,-1)
    test_target = test_target.astype(np.float32)
    
    data_train=[]
    data_test=[]
    for input, target in zip(train_input, train_target):
        data_train.append([input, target])
    for input, target in zip(test_input, test_target):
        data_test.append([input, target])

    trainloader=torch.utils.data.DataLoader(dataset=data_train, batch_size=128 , shuffle=True)
    testloader=torch.utils.data.DataLoader(dataset=data_test, batch_size=128  , shuffle=True)
    
    return trainloader, testloader

def calculate_mean_std_dataset(loader):
    mean = 0.
    std = 0.
    nb_samples = 0.
    for images,_ in loader:
        batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(loader.dataset)
    std /= len(loader.dataset)
    return mean, std
    
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
#################################


def initialize_train_network(trainloader, testloader, debug): 
    
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

    model = Generator()
    model = nn.DataParallel(model)
    model.to(device)
    print(next(model.parameters()).device)
#     model.apply(weight_init) ## Weight initialisation 
    
    
    
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-8)
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
    
    ## Initialising Data Normalise 
    train_norm = Normalize(mean_train,std_train)
    test_norm = Normalize(mean_test, std_test)
    
    ##Initialising Unnormalised 
    test_unorm = UnNormalize(mean_test, std_test)

    
    best_loss=10000
    train=[]
    test=[]
    psnr=[]
    if os.path.exists((os.path.join(results,"Train_loss.txt"))):
        dbfile = open(os.path.join(results,"Train_loss.txt"), 'rb')      
        train = pickle.load(dbfile)
        dbfile = open(os.path.join(results,"Test_loss.txt"), 'rb')      
        test = pickle.load(dbfile)
        dbfile = open(os.path.join(results,"PSNR.txt"), 'rb')      
        psnr = pickle.load(dbfile)
    loss1=0
    for epoch in range(48):
        training_loss=[]
        test_loss=[]
        list_no=0
        for input_,target in trainloader:
            if torch.cuda.is_available():
#                 input_ = train_norm(input_)
#                 target= train_norm(target)
                target=target.to(device)
                input_ = input_.to(device)        
            output = model(input_)
            loss=criterion(output, target)
            loss.backward()
            optimizer.step()
            if debug == True: 
                plot_grad_flow(net_debug,model.named_parameters(),"super_resolution_network")
            optimizer.zero_grad()
            training_loss.append(loss.item())
        
        with torch.set_grad_enabled(False):
            for local_batch, local_labels in testloader:
#                 local_batch = test_norm(local_batch)
#                 local_labels = test_norm(local_labels)
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)
                output = model(local_batch).to(device)
                local_labels.require_grad = False
                test_loss.append(test_criterion(output, local_labels).item())
#                 if debug == True: 
#                     visualise_layer_activation(model,local_batch,net_debug)

        my_lr_scheduler.step(test_loss[-1])
        torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_file)
        
        if(debug == True):
#             label = test_unorm(local_labels[0])
#             output = test_unorm(output[0])
            label=im.fromarray(np.uint8(np.moveaxis((local_labels[0].cpu()).detach().numpy(),0,-1))).convert('RGB')
            output=im.fromarray(np.uint8(np.moveaxis((output[0].cpu()).detach().numpy(),0,-1))).convert('RGB')
            
            label.save(os.path.join(results,str(epoch) + 'test_target' + '.png'))
            output.save(os.path.join(results,str(epoch) + 'test_output' + '.png'))
  
        train.append(sum(training_loss)/len(training_loss))
        test.append(sum(test_loss)/len(test_loss))
        psnr.append(10*math.log10(1/(sum(test_loss)/len(test_loss))))
        with open(os.path.join(results,"Train_loss.txt"), 'wb') as f:
                pickle.dump(train ,f)
        with open(os.path.join(results,"Test_loss.txt"), 'wb') as f:
             pickle.dump(test,f )
        with open(os.path.join(results,"PSNR.txt"), 'wb') as f:
             pickle.dump(psnr,f )
                
        print("Epoch :",epoch)
        print("Training loss :",sum(training_loss)/len(training_loss))
        print("Test loss :",sum(test_loss)/len(test_loss))
        print("PSNR :", 10*math.log10(1/(sum(test_loss)/len(test_loss))))

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


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--debug', help="Mode of Execution here")
    args = parser.parse_args()
    
    grad_flow_flag = False


    if args.debug == "debug": 
        print("Running in Debug Mode.....")
        grad_flow_flag = True
        
    trainloader, testloader = process_and_train_load_data()
    initialize_train_network(trainloader, testloader,grad_flow_flag)