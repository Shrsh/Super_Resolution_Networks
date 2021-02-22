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
import random
from scipy.ndimage.filters import gaussian_filter


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


def normalize(data):
    size=data[0].shape[0]*data[0].shape[1]*data[0].shape[2]
    for i in range (len(data)):
        x=data[i].reshape(1,size).tolist()
        data[i]=(data[i]-min(x[0]))/(max(x[0])-min(x[0]))
    return data
        
def process_and_train_load_data():
    train_xx= load_images_from_folder('/home/harsh.shukla/SRCNN/HR_LR_data/train/x')
    train_x=[i for i in train_xx]
#     print(len(train_xx))
#     for i in train_xx :
#         train_x.append(gaussian_blur(i))
#     print(len(train_x))
    train_x=normalize(train_x)
    train_input=np.asarray(train_x)
    train_input=np.moveaxis(train_input,1,-1)
    train_input=np.moveaxis(train_input,1,-1)
    train_input = train_input.astype(np.float32)

    train_yy= load_images_from_folder('/home/harsh.shukla/SRCNN/HR_LR_data/train/y')
#     print(len(train_yy))
    train_y=[i for i in train_yy]
#     for i in train_yy :
#         train_y.append(gaussian_blur(i))
#     print(len(train_y))
    train_y=normalize(train_y)
    train_target=np.asarray(train_y)
    train_target=np.moveaxis(train_target,1,-1)
    train_target=np.moveaxis(train_target,1,-1)
    train_target = train_target.astype(np.float32)
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

    trainloader=torch.utils.data.DataLoader(dataset=data_train, batch_size=128, shuffle=True)
    testloader=torch.utils.data.DataLoader(dataset=data_test, batch_size=128, shuffle=True)
    return trainloader, testloader

trainloader, testloader = process_and_train_load_data()
