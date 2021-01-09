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


# Ignore warnings
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
if device == "cuda":
    for i in range(len(device_ids)):
        print(torch.cuda.get_device_name(i))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
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
#     print(list_name)
    for filename in list_name:
        img = cv2.imread(filename)
        if img is not None:
            images.append(img)
#             print(c)
#             c=c+1
        # if c==8
    return images

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
    
    
    
model=SRSN()
model = nn.DataParallel(model, device_ids = device_ids)
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)
criterion = nn.MSELoss().to(device)
best_loss=10000
train=[]
test=[]
model = model.to(device)
os.mkdir("/home/harsh.shukla/SRCNN/sr_results")
results = "/home/harsh.shukla/SRCNN/sr_results"
count = 0
loss1=0
for epoch in range(2500):
    training_loss=0
    test_loss=0


    for input_,target in trainloader:
        if torch.cuda.is_available():
            input_ = input_.to(device)
            target=target.to(device)
        # print("cuda")
        output = model(input_)
        loss=criterion(output, target)
        loss.backward()
#         loss1+=loss
        
        ## Gradient Accumulation to fit a larger batch size in the memory. 
        if count % 4 == 0:
#             loss1.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss1=0
        training_loss+=loss.item()
        count+=1

        label=im.fromarray(np.uint8(np.moveaxis(target[0].cpu().detach().numpy(),0,-1))).convert('RGB')
        output=im.fromarray(np.uint8(np.moveaxis(output[0].cpu().detach().numpy(),0,-1))).convert('RGB')
        label.save(os.path.join(results,str(epoch) + 'train_target' + '.png'))
        output.save(os.path.join(results,str(epoch) + 'train_output' + '.png'))
    with torch.set_grad_enabled(False):
        for local_batch, local_labels in testloader:
#             count+=1
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            output = model(local_batch).to(device)
#             print(output[0].type)
#             print(output[0].shape)
            local_labels.require_grad = False
            test_loss += criterion(output, local_labels).item()
#             if count% 10 == 0:

#     print(local_labels[0].cpu().detach().numpy().astype('int').shape)
#     label=local_labels[0].cpu().detach().numpy()
# #     PIL_image = im.fromarray(np.uint8(numpy_image)).convert('RGB')
#     label=np.moveaxis(label,0,-1)
#     label=im.fromarray(np.uint8(label*255)).convert('RGB')
# #     print(label.shape)
    label=im.fromarray(np.uint8(np.moveaxis(local_labels[0].cpu().detach().numpy(),0,-1))).convert('RGB')
    output=im.fromarray(np.uint8(np.moveaxis(output[0].cpu().detach().numpy(),0,-1))).convert('RGB')
    label.save(os.path.join(results,str(epoch) + 'test_target' + '.png'))
    output.save(os.path.join(results,str(epoch) + 'test_output' + '.png'))
#     save_image(local_labels[0].int(),os.path.join(results,str(count) + 'target' + '.png'))
#     save_image(output[0].int(),os.path.join(results,str(count) + 'output' + '.png'))
            # Defining our loss function, comparing the output with the target
            
  
    train.append(training_loss/len(data_train))
    test.append(test_loss/len(data_test))
    if test_loss<best_loss:
        best_loss=test_loss
    print("Epoch :",epoch )
    print("Training loss :",training_loss/len(data_train))
    print("Test loss :",test_loss/len(data_test))

    print("-----------------------------------------------------------------------------------------------------------")
try:
    file = open("home/harsh.shukla/SRCNN/sr_results/SR_train_loss.txt", 'w+')
    for i in range(len(test_loss)):
        file.write(str(training_loss[i]) + ","  + str(test_loss[i]))
        file.write('\n')
finally:
    file.close()
    
    

    



