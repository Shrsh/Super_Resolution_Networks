import torch 
import os
from os import listdir
from os.path import isfile, join 
from PIL import Image
import sys
import numpy as np
from random import sample 
import cv2


class pre_process: 
    test_path=""
    train_path=""


    """ 
     Parameters
      ----------
     image : ndarray
     Input image data. Will be converted to float.
     mode : str
     One of the following strings, selecting the type of noise to add:

     'gauss'     Gaussian-distributed additive noise.
     'poisson'   Poisson-distributed noise generated from the data.
     's&p'   Replaces random pixels with 0 or 1.
     'speckle'   Multiplicative noise using out = image + n*image,where
        n is uniform noise with specified mean & variance.
    """

    def __init__(self,test_directory,train_directory,mean=0.,std=1.): 
        self.test_path = test_directory 
        self.train_path = train_directory    

    def process(self,string):
        if string == "train": 
            path = self.train_path 
        else:
            path = self.test_path
    
        os.chdir(path)
        walker = list(os.walk(path))
        filenames = walker[0][2]
        processed_dir = os.path.join(path,"processed")
        features = os.path.join(processed_dir,"features")
        label = os.path.join(processed_dir,"label")
        os.mkdir(processed_dir)
        os.mkdir(features)
        os.mkdir(label)
        
        noise_samples = ['s&p', 'gauss', 'speckle' ]
        for i in filenames:
        
            #center crop the images, 800*800
            im = Image.open(i)
            left = int(im.size[0]/2-800/2)
            upper = int(im.size[1]/2-800/2)
            right = left +800
            lower = upper + 800
            
            im_cropped = im.crop((left, upper,right,lower))
            im = im.resize((128,128))
            im.save(os.path.join(label,i))
            im = np.array(im)
#             trans = sample(noise_samples,2)
#             for sm in trans: 
#                 im = noisy(self,sm,im)
            im = noisy("gauss",im)
            im = Image.fromarray(np.uint8(im)).convert('RGB')
            im = im.save(os.path.join(features, i))
            print(i,end='',flush=True)
            sys.stdout.flush()

def add_noise(inputs):
    noise = torch.randn_like(inputs)*0.2
    return inputs + noise

def noisy(noise_typ,image):
    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 80
        var = 500
        sigma = var ** 0.5
        gaussian = np.random.normal(mean, sigma, (row,col)) #  np.zeros((224, 224), np.float32)
        noisy_image = np.zeros(image.shape, np.float32)
        if len(image.shape) == 2:
            noisy_image = image + 1.5*gaussian
        else:
            noisy_image[:, :, 0] = image[:, :, 0] + gaussian
            noisy_image[:, :, 1] = image[:, :, 1] + gaussian
            noisy_image[:, :, 2] = image[:, :, 2] + gaussian

        cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
        noisy_image = noisy_image.astype(np.uint8)
#         mean = 0
#         var = 4
#         sigma = var**0.5
#         gauss = np.random.normal(mean,sigma,(row,col,ch))
#         gauss = gauss.reshape(row,col,ch)
#         noisy = image + gauss
        return noisy_image
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.003
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out[coords] = 1
        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[coords] = 0
        return out
#     elif noise_typ == "poisson":
#         vals = len(np.unique(image))
#         vals = 2 ** np.ceil(np.log2(vals))
#         noisy = np.random.poisson(image * vals) / float(vals)
#         return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)    
        noisy = image + image * gauss
        return noisy


train_directory = "/home/harsh.shukla/SRCNN/data/train"
test_directory = "/home/harsh.shukla/SRCNN/data/test"

ob = pre_process(test_directory,train_directory)
ob.process("train")
print("Training PreProcessing Done")
ob.process("test")
print("Test PreProcessing Done")

