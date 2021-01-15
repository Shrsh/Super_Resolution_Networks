from __future__ import print_function, division


import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models

import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.utils import save_image

from PIL import Image 

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
from IPython.display import clear_output

plt.ion()   # interactive mode

# CUDA for PyTorch
print("Number of GPUs:" + str(torch.cuda.device_count()))
use_cuda = torch.cuda.is_available()
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device_ids = [i for i in range(torch.cuda.device_count())]
device = 'cuda' if use_cuda else 'cpu'
print(device)
if device == "cuda":
    for i in range(len(device_ids)):
        print(torch.cuda.get_device_name(i))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
torch.backends.cudnn.benchmark = True



trans = transforms.ToPILImage()
trans1 = transforms.ToTensor()

# VGG Features Extractor
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
    
## Resnet Block
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
    
## Denoising VAE
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        #Encoder
        self.lin1 = nn.Linear(8192, 4096)
        self.lin2 =nn.Linear(4096, 512)

        # Decoder
        self.fc1 = nn.Linear(512,4096)
        self.fc2 = nn.Linear(4096,8192)
        self.conv1 = torch.nn.Conv2d(128, 128, 1, 1, 0)
        self.conv2 = torch.nn.Conv2d(64, 32, 3, 1, 1)
        self.deconv1=torch.nn.ConvTranspose2d(128, 128, 6, 4, 1)
        self.deconv2=torch.nn.ConvTranspose2d(128, 64, 6, 4, 1)


        self.conv3 = torch.nn.Conv2d(3, 64, 3, 1, 1)
        self.conv4 = torch.nn.Conv2d(64, 64, 3, 1, 1)
        self.conv5 = torch.nn.Conv2d(64, 32, 3, 1, 1)

        self.resnet=ResnetBlock(32, 3, 1, 1)

        self.up_sample = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv=nn.Conv2d(32, 3, 3, 1, 1)
        
  
    def encoder(self,x):
        x=x.view(x.size(0), -1)
        x=F.sigmoid(self.lin1(x))
        mu = self.lin2(x)
        logvar =self.lin2(x)
        return mu ,logvar
  
    def decoder(self,clean_x,noise_x):
        clean_x=self.fc1(clean_x)
        clean_x=F.sigmoid(clean_x)
        clean_x=F.sigmoid(self.fc2(clean_x))
        clean_x = clean_x.view(clean_x.size(0), -1, 8, 8)
        clean_x=F.relu(self.conv1(clean_x))
        clean_x=F.relu(self.deconv1(clean_x))
        clean_x=F.relu(self.deconv2(clean_x))
        clean_x=F.relu(self.conv2(clean_x))

        noise_x=F.relu(self.conv3(noise_x))
        noise_x=F.relu(self.conv4(noise_x))
        noise_x=F.relu(self.conv5(noise_x))

        features=noise_x-clean_x
        features=self.resnet(features)
        features=self.resnet(features)
        features=self.resnet(features)
        features=self.conv(features)
        return features
  
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
  
    def forward(self,x,noise_x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        re_x = self.decoder(z,noise_x)
        denoise=noise_x-re_x
        return denoise,z,mu,logvar

def vae_loss(denoise_x,target,z,mu,log_var):
    reconstruction_loss = L1_criterion(denoise_x, target)
    log_p_z= torch.sum(-0.5 * torch.pow( z , 2 ),dim=1)
    log_q_z=torch.sum(-0.5 * ( log_var + torch.pow( z - mu, 2 ) / torch.exp( log_var ) ),dim=1)
    KL = -(log_p_z - log_q_z)
    KL = torch.sum(KL)
    return (reconstruction_loss + KL)
    
    
    
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
        pick_feature = os.path.join(self.dir,os.path.join("features",ID))
        pick_label = os.path.join(self.dir,os.path.join("label",ID))        
        # Load data and get label
        X = trans1(Image.open(pick_feature))
        Y = trans1(Image.open(pick_label))
        return X, Y

def listIDs(path): 
  walker = list(os.walk(path))
  return walker[0][2]


# Parameters
params = {'batch_size': 256,
          'shuffle': True,
         }
max_epochs = 1000

#training data loader 
train_directory = "/home/harsh.shukla/SRCNN/data/train/processed/"
trainIDs = listIDs(os.path.join(train_directory,"features"))
training_set = Dataset(trainIDs, train_directory)
training_generator = torch.utils.data.DataLoader(training_set,**params)

#test data loader 
test_directory = "/home/harsh.shukla/SRCNN/data/test/processed/"
testIDs = listIDs(os.path.join(test_directory,"features"))
test_set = Dataset(testIDs,test_directory)
test_generator = torch.utils.data.DataLoader(test_set,**params)


#Creating a Fresh Results Directory 
results = "/home/harsh.shukla/SRCNN/results"
os.mkdir(results)

# Initializing Network 
vgg_features=VGGFeatureExtractor().cuda()
model=VAE().cuda()
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(model)
    vgg_features = nn.DataParallel(vgg_features)

model.to(device)
vgg_features.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)
L1_criterion = nn.L1Loss()

training_loss = []
test_loss = []
count = 0
for epoch in range(max_epochs):
    # Training
    running_loss = 0.0
    validation_loss = 0.0
    for local_batch, local_labels in training_generator:
        # Transfer to GPU
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        optimizer.zero_grad()
        features=vgg_features(local_batch)
        denoise_x,z,mu,logvar = model(features,local_batch)
        loss=vae_loss(denoise_x,local_labels,z,mu,logvar)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
    # Validation
    with torch.set_grad_enabled(False):
        for local_batch, local_labels in test_generator:
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            features=vgg_features(local_batch)
            denoise_x,z,mu,logvar = model(features,local_batch)
            local_labels.require_grad = False
            val_loss=vae_loss(denoise_x,local_labels,z,mu,logvar)
            if count% 10 == 0:
                save_image(local_batch[0],os.path.join(results,str(count) + 'input' + '.png'))
                save_image(local_labels[0],os.path.join(results,str(count) + 'target' + '.png'))
                save_image(denoise_x[0],os.path.join(results,str(count) + 'output' + '.png'))
            count +=1
            # Defining our loss function, comparing the output with the target
            validation_loss += val_loss.item()
    print('[%d] Training_loss: %.3f' % (epoch + 1, running_loss/len(training_set)))
    print('[%d] Validation_loss: %.3f' % (epoch + 1, validation_loss/len(test_set)))
    training_loss.append(running_loss)
    test_loss.append(validation_loss)
    running_loss = 0.0
    validation_loss = 0.0
    
    
## saving the plot results 
try:
    file = open("/home/harsh.shukla/SRCNN/results/Loss.txt","w+")
    for i in range(len(test_loss)):
        file.write(str(training_loss[i]) + ","  + str(test_loss[i]))
        file.write('\n')
finally:
    file.close()