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


class SRSN(nn.Module):
    def __init__(self, input_dim=3, dim=32, scale_factor=4):
        super(SRSN, self).__init__()
        self.up = torch.nn.Upsample(scale_factor=4, mode='bicubic')
        self.conv1 = torch.nn.Conv2d(3, 64, 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(64, 32, 1, 1, 0)
        self.resnet = nn.Sequential(
            ResnetBlock(dim, 3, 1, 1, bias=True),
            ResnetBlock(dim, 3, 1, 1, bias=True),
            ResnetBlock(dim, 3, 1, 1, bias=True),
            ResnetBlock(dim, 3, 1, 1, bias=True),)
        self.conv3=torch.nn.Conv2d(32, 16, 1, 1, 0)
        self.conv4=torch.nn.Conv2d(16, 3, 1, 1, 0)

    def forward(self, LR):
        LR_feat = F.relu(self.conv1(self.up(LR)))
        LR_feat = (F.relu(self.conv2(LR_feat)))
        # print(LR_feat.shape)
        LR_feat = self.resnet(LR_feat)
        SR=F.relu(self.conv3(LR_feat))
        SR =self.conv4(SR)
        # print(SR.shape)
        return SR


class ResnetBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=3, stride=1, padding=1, bias=True):
        super(ResnetBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)
        self.conv2 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)

        self.act1 = torch.nn.ReLU(inplace=True)
        self.act2 = torch.nn.ReLU(inplace=True)

    def forward(self, x):

        out = self.act1(x)
        out = self.conv1(out)

        out = self.act2(out)
        out = self.conv2(out)

        out = out + x
        return out
    

def process_and_train_load_data():
    train= load_images_from_folder('/scratch/harsh_cnn/SR_data/train/x')
    train_input=np.asarray(train)
    train_input=np.moveaxis(train_input,1,-1)
    train_input=np.moveaxis(train_input,1,-1)
    train_input = train_input.astype(np.float32)

    train= load_images_from_folder('/scratch/harsh_cnn/SR_data/train/y')
    train_target=np.asarray(train)
    train_target=np.moveaxis(train_target,1,-1)
    train_target=np.moveaxis(train_target,1,-1)
    train_target = train_target.astype(np.float32)

    test= load_images_from_folder('/scratch/harsh_cnn/SR_data/test/x')
    test_input=np.asarray(test)
    test_input=np.moveaxis(test_input,1,-1)
    test_input=np.moveaxis(test_input,1,-1)
    test_input = test_input.astype(np.float32)

    test= load_images_from_folder('/scratch/harsh_cnn/SR_data/test/y')
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

    trainloader=torch.utils.data.DataLoader(dataset=data_train, batch_size=64, shuffle=True)
    testloader=torch.utils.data.DataLoader(dataset=data_test, batch_size=64, shuffle=True)
    return trainloader, testloader


### Network Debugging
#########################################################################

### Creating function for Gradient Visualisation 
def plot_grad_flow(result_directory,named_parameters): 
    
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
    plt.savefig(os.path.join(result_directory,"gradient_flow.png"))
    
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
    model.module.conv1.register_forward_hook(get_activation('conv1'))
    output = model(local_batch[0])
    act = activation['conv1'].squeeze()
    fig, axarr = plt.subplots(act[0].size(0))
    for idx in range(act[0].size(0)):
        axarr[idx].imsave(act[0][idx],result_directory + "1.png")

### Visualising Conv Filters
def visualise_conv_filters():
    kernels = model.conv1.weight.detach()
    fig, axarr = plt.subplots(kernels.size(0))
    for idx in range(kernels.size(0)):
        axarr[idx].imsave(kernels[idx].squeeze(),result_directory + "1.png")
    
    

#########################################################


def initialize_train_network(trainloader, testloader, debug): 

    model = SRSN()
    model = nn.DataParallel(model, device_ids = device_ids)
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)
    criterion = nn.MSELoss().to(device)
    best_loss=10000
    train=[]
    test=[]
    model = model.to(device)
    os.mkdir("/home/harsh.shukla/SRCNN/sr_results")
    results = "/home/harsh.shukla/SRCNN/sr_results"
    activation_results_dir = os.path.join(results,"Activations")
    os.mkdir(activation_results_dir)
    loss1=0
    for epoch in range(5):
        training_loss=[]
        test_loss=[]
        list_no=0
        for input_,target in trainloader:
            if torch.cuda.is_available():
                input_ = input_.to(device)
                target=target.to(device)
            output = model(input_)
            loss=criterion(output, target)
            loss.backward()
            if debug == True:
                plot_grad_flow(results,model.named_parameters())
            optimizer.step()
            optimizer.zero_grad()
            training_loss.append(loss.item()*output.shape[0])
        
        with torch.set_grad_enabled(False):
            for local_batch, local_labels in testloader:
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)
                output = model(local_batch).to(device)
                local_labels.require_grad = False
                test_loss.append(criterion(output, local_labels).item())
        
                visual_information = visualise_layer_activation(model,local_batch,results)
                
        if(debug == True):
            label=im.fromarray(np.uint8(np.moveaxis(local_labels[0].cpu().detach().numpy(),0,-1))).convert('RGB')
            output=im.fromarray(np.uint8(np.moveaxis(output[0].cpu().detach().numpy(),0,-1))).convert('RGB')
            label.save(os.path.join(results,str(epoch) + 'test_target' + '.png'))
            output.save(os.path.join(results,str(epoch) + 'test_output' + '.png'))
  
        train.append(sum(training_loss)/len(training_loss))
        test.append(sum(test_loss)/len(test_loss))
        print("Epoch :",epoch, flush=True)
        print("Training loss :",sum(training_loss)/len(training_loss),flush=True)
        print("Test loss :",sum(test_loss)/len(test_loss),flush=True)

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
    parser.add_argument('-v','--visualise',help="Visualise Layer Activations")
    args = parser.parse_args()
    
    grad_flow_flag = False
    visualise_gradients = False
    

    if args.debug == "debug": 
        print("Running in Debug Mode.....")
        grad_flow_flag = True
    
    if args.visualise == "visualise": 
        print("Running in Debug Mode.....")
        visualise_gradients = True
        
    trainloader, testloader = process_and_train_load_data()
    initialize_train_network(trainloader, testloader,grad_flow_flag)


    
    
