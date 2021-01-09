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

## Data Loader 
##############
## DataLoader 
class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs,path):
        'Initialization'
        self.list_IDs = list_IDs
        self.dir = path

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        pick_feature = os.path.join(self.dir,os.path.join("input_processed",ID))
        pick_label = os.path.join(self.dir,os.path.join("target_processed",ID))        
        # Load data and get label
        X = trans1(Image.open(pick_feature))
        Y = trans1(Image.open(pick_label))
        return X, Y

def listIDs(path): 
  walker = list(os.walk(path))
  return walker[0][2]

# Parameters
params = {'batch_size': 100,
          'shuffle': True,
          'num_workers':4
         }
max_epochs = 1000

#training data loader 
train_directory = "/home/harsh.shukla/SRCNN/SR_data/train"
trainIDs = listIDs(os.path.join(train_directory,"input_processed"))
training_set = Dataset(trainIDs, train_directory)
training_generator = torch.utils.data.DataLoader(training_set,**params)

#test data loader 
test_directory = "/home/harsh.shukla/SRCNN/SR_data/test"
testIDs = listIDs(os.path.join(test_directory,"input_processed"))
test_set = Dataset(testIDs,test_directory)
test_generator = torch.utils.data.DataLoader(test_set,**params)


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
for epoch in range(2500):
    training_loss=0
    test_loss=0


    for input_,target in training_generator:
        if torch.cuda.is_available():
            input_ = input_.to(device)
            target=target.to(device)
        output = model(input_)
        loss=criterion(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        training_loss+=float(loss.item())
        label=im.fromarray(np.uint8(np.moveaxis(target[0].cpu().detach().numpy(),0,-1))).convert('RGB')
        output=im.fromarray(np.uint8(np.moveaxis(output[0].cpu().detach().numpy(),0,-1))).convert('RGB')
        label.save(os.path.join(results,str(epoch) + 'train_target' + '.png'))
        output.save(os.path.join(results,str(epoch) + 'train_output' + '.png'))
    with torch.set_grad_enabled(False):
        for local_batch, local_labels in test_generator:
#             count+=1
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            output = model(local_batch)
#             print(output[0].type)
#             print(output[0].shape)
            local_labels.require_grad = False
            test_loss += float(criterion(output, local_labels).item())
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