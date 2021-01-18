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
from prettytable import PrettyTable


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
    

### Data Preparation ############################################
################################################################
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

def prepare_data(b_size=50):
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

    trainloader=torch.utils.data.DataLoader(dataset=data_train, batch_size=b_size, shuffle=True)
    testloader=torch.utils.data.DataLoader(dataset=data_test, batch_size=b_size, shuffle=True)
    return trainloader,testloader


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

def train_discriminator(optimizer, real_data, fake_data,discriminator,b_loss):
    optimizer.zero_grad()
    
    # 1. Train on Real Data
    prediction_real = discriminator(real_data)
    error_real = b_loss(prediction_real, Variable(torch.ones(real_data.size(0), 1)).to(device))
    error_real.backward()

    # 2. Train on Fake Data
    prediction_fake = discriminator(fake_data.detach())
    error_fake = b_loss(prediction_fake, Variable(torch.zeros(real_data.size(0), 1)).to(device))
    error_fake.backward()
    optimizer.step()
    return error_real + error_fake, prediction_real, prediction_fake

def train_generator(optimizer, fake_data,real_data,discriminator,b_loss,m_loss):
    optimizer.zero_grad()
    
    ##Reconstruction loss
    loss=m_loss(fake_data, real_data)
    loss.backward(retain_graph=True)

    prediction = discriminator(fake_data)
    error = b_loss(prediction, Variable(torch.ones(real_data.size(0), 1)).to(device))
    error.backward()
    
    optimizer.step()
    return (error+loss)



def train_network(trainloader, testloader,num_epochs=200):

    discriminator = DiscriminativeNet()
    model=SRSN()
    model = model.to(device)
    discriminator=discriminator.to(device)

    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    g_optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)

    Mse_loss = nn.MSELoss().to(device)
    Bce_loss = nn.BCELoss().to(device)

    train_d=[]
    train_g=[]
    test=[]
    
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
        
    # load model if exists
    if os.path.exists(checkpoint_file):
        if config.resume:
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
        
    model = nn.DataParallel(model, device_ids = device_ids)
    discriminator = nn.DataParallel(discriminator, device_ids= device_ids)

    for epoch in range(num_epochs):
        training_loss_d=[]
        training_loss_g=[]
        test_loss=[]

        list_no=0
        for input_,real_data in trainloader:
            if torch.cuda.is_available():
                input_ = input_.to(device)
                real_data=real_data.to(device)

            fake_data = model(input_)
            d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer, real_data, fake_data,discriminator,Bce_loss)
            g_error = train_generator(g_optimizer, fake_data, real_data,discriminator,Bce_loss,Mse_loss)
            training_loss_d.append(d_error.item())
            training_loss_g.append(g_error.item())

        with torch.set_grad_enabled(False):
            for local_batch, local_labels in testloader:
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)
                output = model(local_batch).to(device)
                local_labels.require_grad = False
                test_loss.append(Mse_loss(output, local_labels).item())

        label=im.fromarray(np.uint8(np.moveaxis(local_labels[0].cpu().detach().numpy(),0,-1))).convert('RGB')
        output=im.fromarray(np.uint8(np.moveaxis(output[0].cpu().detach().numpy(),0,-1))).convert('RGB')
        label.save(os.path.join(results,str(epoch) + 'test_target' + '.png'))
        output.save(os.path.join(results,str(epoch) + 'test_output' + '.png'))
        
        ##Creating Checkpoints
        if epoch % 2 == 0:
            torch.save({
                'generator_state_dict': model.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'g_state_dict': g_optimizer.state_dict(),
                'd_state_dict': d_optimizer.state_dict(),
                }, checkpoint_file)
        
        #Calculating average loss per epoch
        train_d.append(sum(training_loss_d)/len(training_loss_d))
        train_g.append(sum(training_loss_g)/len(training_loss_g))
        test.append(sum(test_loss)/len(test_loss))

        print("Epoch :",epoch )
        print("Discriminator Loss :",train_d[-1])
        print("Generator Loss :",train_g[-1])

        print("D(X) :",d_pred_real.mean(), "D(G(X)) :",d_pred_fake.mean())
        print("Test loss :",test[-1])

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
    
    trainloader, testloader = prepare_data(b_size=40)
    train_network(trainloader, testloader,num_epochs=3)


