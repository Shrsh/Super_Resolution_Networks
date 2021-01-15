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

use_cuda = torch.cuda.is_available()
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device_ids = [i for i in range(torch.cuda.device_count())]
device = 'cuda' if use_cuda else 'cpu'
torch.backends.cudnn.benchmark = True


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
# if list_name1 == list_name2:
#     print("True")
# else:
#     print("False")

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

trainloader=torch.utils.data.DataLoader(dataset=data_train, batch_size=32, shuffle=True)
testloader=torch.utils.data.DataLoader(dataset=data_test, batch_size=32, shuffle=True)


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
                stride=2, padding=1, bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),
              nn.BatchNorm2d(256)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(512)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(
                in_channels=512, out_channels=256, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(256)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=128, kernel_size=4,
                stride=2, padding=1, bias=False
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
        x = self.conv5(x)
        x = F.relu(self.conv6(x))
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

def train_discriminator(optimizer, real_data, fake_data):
    optimizer.zero_grad()
    
    # 1. Train on Real Data
    prediction_real = discriminator(real_data)
#     print(prediction_real .shape)
#     print(real_data.size(0))
    error_real = Bce_loss(prediction_real, Variable(torch.ones(real_data.size(0), 1)).to(device))
    error_real.backward()

    # 2. Train on Fake Data
    prediction_fake = discriminator(fake_data.detach())
    error_fake = Bce_loss(prediction_fake, Variable(torch.zeros(real_data.size(0), 1)).to(device))
    error_fake.backward()
    optimizer.step()
    return error_real + error_fake, prediction_real, prediction_fake

def train_generator(optimizer, fake_data,real_data):
    optimizer.zero_grad()
    
    ##Reconstruction loss
    loss=Mse_loss(fake_data, real_data)
    loss.backward(retain_graph=True)
    
    prediction = discriminator(fake_data)
    error = Bce_loss(prediction, Variable(torch.ones(real_data.size(0), 1)).to(device))
    error.backward()
    
    
    optimizer.step()
    return error+loss


discriminator = DiscriminativeNet()
model=SRSN()
model = model.to(device)
discriminator=discriminator.to(device)
model = nn.DataParallel(model, device_ids = device_ids)
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
g_optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)
Mse_loss = nn.MSELoss().to(device)
Bce_loss = nn.BCELoss().to(device)

best_loss=10000
train_d=[]
train_g=[]
test=[]

os.mkdir("/home/harsh.shukla/SRCNN/GAN_results")
results = "/home/harsh.shukla/SRCNN/GAN_results"
count = 0
loss1=0
for epoch in range(2500):
    training_loss_d=0
    training_loss_g=0
    test_loss=0

    list_no=0
    for input_,real_data in trainloader:
        if torch.cuda.is_available():
            input_ = input_.to(device)
            real_data=real_data.to(device)

        fake_data = model(input_)
        d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer, real_data, fake_data)
        g_error = train_generator(g_optimizer, fake_data, real_data)
        training_loss_d+=d_error.item()
        training_loss_g+=g_error.item()
        

    lab=im.fromarray(np.uint8(np.moveaxis(real_data[0].cpu().detach().numpy(),0,-1))).convert('RGB')
    out=im.fromarray(np.uint8(np.moveaxis(fake_data[0].cpu().detach().numpy(),0,-1))).convert('RGB')
    lab.save(os.path.join(results,str(epoch) + 'train_target' + '.png'))
    out.save(os.path.join(results,str(epoch) + 'train_output' + '.png'))

        
    with torch.set_grad_enabled(False):
        for local_batch, local_labels in testloader:
#             count+=1
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            output = model(local_batch).to(device)
#             print(output[0].type)
#             print(output[0].shape)
            local_labels.require_grad = False
            test_loss += Mse_loss(output, local_labels).item()

    label=im.fromarray(np.uint8(np.moveaxis(local_labels[0].cpu().detach().numpy(),0,-1))).convert('RGB')
    output=im.fromarray(np.uint8(np.moveaxis(output[0].cpu().detach().numpy(),0,-1))).convert('RGB')
    label.save(os.path.join(results,str(epoch) + 'test_target' + '.png'))
    output.save(os.path.join(results,str(epoch) + 'test_output' + '.png'))

  
    train_d.append(training_loss_d/len(data_train))
    train_g.append(training_loss_g/len(data_train))
    test.append(test_loss/len(data_test))
    if test_loss<best_loss:
        best_loss=test_loss
    print("Epoch :",epoch )
    print("Discriminator Loss :",train_d[-1])
    print("Generator Loss :",train_g[-1])
    
    print("D(X) :",d_pred_real.mean(), "D(G(X)) :",d_pred_fake.mean())
    print("Test loss :",test_loss/len(data_test))

    print("-----------------------------------------------------------------------------------------------------------")
try:
    file = open("home/harsh.shukla/SRCNN/sr_results/SR_train_loss.txt", 'w+')
    for i in range(len(test_loss)):
        file.write(str(training_loss[i]) + ","  + str(test_loss[i]))
        file.write('\n')
finally:
    file.close()
    



