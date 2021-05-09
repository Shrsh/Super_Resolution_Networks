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
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device_ids = [i for i in range(torch.cuda.device_count())]
device = 'cuda' if use_cuda else 'cpu'
torch.backends.cudnn.benchmark = True
from prettytable import PrettyTable
import torch.nn.init as init

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
def sobel(window_size):
	assert(window_size%2!=0)
	ind=int(window_size/2)
	matx=[]
	maty=[]
	for j in range(-ind,ind+1):
		row=[]
		for i in range(-ind,ind+1):
			if (i*i+j*j)==0:
				gx_ij=0
			else:
				gx_ij=i/float(i*i+j*j)
			row.append(gx_ij)
		matx.append(row)
	for j in range(-ind,ind+1):
		row=[]
		for i in range(-ind,ind+1):
			if (i*i+j*j)==0:
				gy_ij=0
			else:
				gy_ij=j/float(i*i+j*j)
			row.append(gy_ij)
		maty.append(row)

	# matx=[[-3, 0,+3],
	# 	  [-10, 0 ,+10],
	# 	  [-3, 0,+3]]
	# maty=[[-3, -10,-3],
	# 	  [0, 0 ,0],
	# 	  [3, 10,3]]
	if window_size==3:
		mult=2
	elif window_size==5:
		mult=20
	elif window_size==7:
		mult=780

	matx=np.array(matx)*mult				
	maty=np.array(maty)*mult

	return torch.Tensor(matx), torch.Tensor(maty)


def create_window(window_size, channel):
    windowx,windowy=sobel(window_size)
    windowx,windowy=windowx.unsqueeze(0).unsqueeze(0),windowy.unsqueeze(0).unsqueeze(0)
#     print(windowx.shape)
    windowx=torch.Tensor(windowx.expand(1,1,window_size,window_size))
    windowy=torch.Tensor(windowy.expand(1,1,window_size,window_size))
    return windowx,windowy


def gradient(img, windowx, windowy, window_size, padding, channel):
    if channel>1:
        gradx=torch.ones(img.shape)
        grady=torch.ones(img.shape)
#         print(windowx.shape)
#         print(gradx.shape)
        img=img.unsqueeze(2)
#         img=torch.Tensor(img.expand(16,channel,1,512,512))
#         print(img.shape)
#         print(img[:,0,:,:,:].shape)
        for i in range(channel):
            z=F.conv2d(img[:,i,:,:,:],windowx,padding=padding,groups=1)
            print("z : ",z.shape)
            gradx[:,i,:,:]=F.conv2d(img[:,i,:,:,:],windowx,padding=padding,groups=1).squeeze(1)
            grady[:,i,:,:]=F.conv2d(img[:,i,:,:,:],windowy,padding=padding,groups=1).squeeze(1)
    return gradx,grady



class SobelGrad(torch.nn.Module):
	def __init__(self, window_size = 3, padding= 1):
		super(SobelGrad, self).__init__()
		self.window_size = window_size
		self.padding= padding
		self.channel = 3			# out channel
		self.windowx,self.windowy = create_window(window_size, self.channel)

	def forward(self, pred,label):
		(batch_size, channel, _, _) = pred.size()
		if pred.is_cuda:
			self.windowx = self.windowx.cuda(pred.get_device())
			self.windowx = self.windowx.type_as(pred)
			self.windowy = self.windowy.cuda(pred.get_device())
			self.windowy = self.windowy.type_as(pred)	
            
		pred_gradx,pred_grady=gradient(pred,self.windowx,self.windowy,self.window_size, self.padding,channel)
		label_gradx,label_grady=gradient(label,self.windowx,self.windowy,self.window_size,self.padding, channel)
		return pred_gradx,pred_grady,label_gradx,label_grady
    
    
# https://gist.github.com/sagniklp/a8f7d7b07ef318f339d11379a42f9d4a