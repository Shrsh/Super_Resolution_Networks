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
# from vgg import vgg19
from models import arch, DiscriminativeNet, VGGFeatureExtractor
from initializer import kaiming_normal_
from utilities import plot_grad_flow, count_parameters, SobelGrad
import kornia


####### Initialisation 
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
#         init.kaiming_uniform_(m.weight.data, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
#         init.xavier_normal_(m.weight.data)
#         init.xavier_uniform_(m.weight.data, gain=1.0)
        kaiming_normal_(m.weight.data, mode='fan_in',nonlinearity='leaky_relu')
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
#         init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
#         init.xavier_normal_(m.weight.data)
        torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
#         init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
#         init.xavier_normal_(m.weight.data)
        torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
#         init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
#         init.xavier_normal_(m.weight.data)
        torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
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
   
### Data Preparation ############################################
################################################################

def normalize(data):
    size=data[0].shape[0]*data[0].shape[1]*data[0].shape[2]
    for i in range (len(data)):
        x=data[i].reshape(1,size).tolist()
        data[i]=(data[i]-min(x[0]))/(max(x[0])-min(x[0]))
    return data

def calculate_mean_std_dataset(loader):
    mean_d = 0.
    std_d = 0.
    mean_l = 0. 
    std_l = 0.
    nb_samples = 0.
    for data,label in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean_d += data.mean(2).sum(0)
        std_d += data.std(2).sum(0)
        nb_samples += batch_samples
        
        label = label.view(batch_samples, label.size(1), -1)
        mean_l += label.mean(2).sum(0)
        std_l += label.std(2).sum(0)

    mean_d /= nb_samples
    std_d /= nb_samples
    
    mean_l /= nb_samples 
    std_l /= nb_samples
    print("Data Mean: ",mean_d)
    print("Data Std: ",std_d)
    print("Data Mean: ",mean_l)
    print("Data Std: ",std_l)
    return mean_d, std_d, mean_l, std_l
    
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

def process_and_train_load_data():
    train_y = []
    train_x = []
    data_train=[]
#     print("Before Train_y")
    train_yy= load_images_from_folder('/home/harsh.shukla/SRCNN/SR_data/train/y')
#     print("After Train_y")
    for i in train_yy:
        train_y.append(i)
#         print("Yes")
#     print("Before Train_x")
    train_xx= load_images_from_folder('/home/harsh.shukla/SRCNN/SR_data/train/x')
#     print("After Train_x")
    for i in train_xx:
        train_x.append(i)
       
#     train_yy= load_images_from_folder('/home/harsh.shukla/SRCNN/training_test_data/Div2K_data/train/y')
#     for i in train_yy:
#         train_y.append(i)
#     train_xx= load_images_from_folder('/home/harsh.shukla/SRCNN/training_test_data/Div2K_data/train/x')
#     for i in train_xx:
#         train_x.append(i)
    
#     train_yy= load_images_from_folder('/home/harsh.shukla/SRCNN/training_test_data/Urban100/train/y')
#     for i in train_yy:
#         train_y.append(i)
#     train_xx= load_images_from_folder('/home/harsh.shukla/SRCNN/training_test_data/Urban100/train/x')
#     for i in train_xx:
#         train_x.append(i)
    
    print("size of training set:", len(train_y))

#     for i in train_yy :
#         x=shift_horizontal(i)
#         small_array = cv2.resize(x, (128,128))
#         train_y.append(x)
#         train_x.append(small_array)
#         x=rotate(i)
#         small_array = cv2.resize(x, (128,128))
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
    for input, target in zip(train_input, train_target):
        data_train.append([input, target])

    test= load_images_from_folder('/home/harsh.shukla/SRCNN/SR_data_512/test/x')
    test_input=np.asarray(test)
    test_input=np.moveaxis(test_input,1,-1)
    test_input=np.moveaxis(test_input,1,-1)
    test_input = test_input.astype(np.float32)

    test= load_images_from_folder('/home/harsh.shukla/SRCNN/SR_data_512/test/y')
    test_target=np.asarray(test)
    test_target=np.moveaxis(test_target,1,-1)
    test_target=np.moveaxis(test_target,1,-1)
    test_target = test_target.astype(np.float32)
    data_test_flickr=[]
    data_test_div=[]
    data_test_urban=[]
    for input, target in zip(test_input, test_target):
        data_test_flickr.append([input, target])

    test= load_images_from_folder('/home/harsh.shukla/SRCNN/SR_data_512/Urban/x')
    test_input=np.asarray(test)
#     print("Urban:",test_input.shape)
    test_input=np.moveaxis(test_input,1,-1)
#     print("Urban:",test_input.shape)
    test_input=np.moveaxis(test_input,1,-1)
#     print("Urban:",test_input.shape)
    test_input = test_input.astype(np.float32)
#     print("Urban:",test_input.shape)

    test= load_images_from_folder('/home/harsh.shukla/SRCNN/SR_data_512/Urban/y')
    test_target=np.asarray(test)
    test_target=np.moveaxis(test_target,1,-1)
    test_target=np.moveaxis(test_target,1,-1)
    test_target = test_target.astype(np.float32)
    

    for input, target in zip(test_input, test_target):
        data_test_div.append([input, target])
        
    test= load_images_from_folder('/home/harsh.shukla/SRCNN/SR_data_512/Div/x')
    test_input=np.asarray(test)
#     print("Div : ",test_input.shape)
    test_input=np.moveaxis(test_input,1,-1)
#     print("Div : ",test_input.shape)
    test_input=np.moveaxis(test_input,1,-1)
#     print("Div : ",test_input.shape)
    test_input = test_input.astype(np.float32)
#     print("Div : ",test_input.shape)

    test= load_images_from_folder('/home/harsh.shukla/SRCNN/SR_data_512/Div/y')
    test_target=np.asarray(test)
    test_target=np.moveaxis(test_target,1,-1)
    test_target=np.moveaxis(test_target,1,-1)
    test_target = test_target.astype(np.float32)

    for input, target in zip(test_input, test_target):
        data_test_urban.append([input, target])
        
    
    trainloader=torch.utils.data.DataLoader(dataset=data_train, batch_size=16, shuffle=False)
    testloader_flickr=torch.utils.data.DataLoader(dataset=data_test_flickr, batch_size=16, shuffle=False)
    testloader_urban=torch.utils.data.DataLoader(dataset=data_test_div, batch_size=15, shuffle=False)
    testloader_div=torch.utils.data.DataLoader(dataset=data_test_urban, batch_size=8, shuffle=False)
    
#     calculate_mean_std_dataset(trainloader)
#     calculate_mean_std_dataset(testloader_flickr)
#     calculate_mean_std_dataset(testloader_div)
#     calculate_mean_std_dataset(testloader_urban)
    
    return trainloader, testloader_flickr,testloader_div,testloader_urban
    

def train_discriminator(optimizer, real_data, fake_data,discriminator,b_loss):
    
    optimizer.zero_grad()
    prediction_real = discriminator(real_data.detach()).to(device)
    prediction_fake = discriminator(fake_data.detach()).to(device)
    
    # 1. Train on Real Data
    truth_real=Variable(torch.ones(real_data.size(0), 1))-0.1
#     truth_real=Variable(torch.Tensor(real_data.size(0), 1).fill_(0.9).type(dtype))
    error_real = b_loss(prediction_real, truth_real.to(device)) 
    error_real.mean().backward(retain_graph=True)

    # 2. Train on Fake Data
    error_fake = b_loss(prediction_fake, Variable(torch.zeros(real_data.size(0), 1)).to(device))
    error_fake.mean().backward()
    optimizer.step()

    return error_real + error_fake, prediction_real, prediction_fake



def train_generator(model, optimizer, fake_data,real_data,discriminator,b_loss,m_loss,vgg_features_high):
    optimizer.zero_grad()
    loss_perceptual_low=0
    loss_perceptual_high=0
    lambda_ = 0
    
    ##Reconstruction loss
    loss=m_loss(fake_data, real_data)
    ## Adversarial Loss 
    prediction = discriminator(fake_data).to(device)
    error = b_loss(prediction, Variable(torch.ones(real_data.size(0), 1)).to(device))
#     features_gt_low, features_gt_high=vgg_features_high(real_data)
#     features_out_low, features_out_high=vgg_features_high(fake_data)
    
    ## Perceptual High
#     loss_perceptual_low=m_loss(features_gt_low, features_out_low)
#     loss_perceptual_high=m_loss(features_gt_high, features_out_high)
    
    ##Image Gradient Loss

    sobel = SobelGrad()
    pred_gradx,pred_grady,label_gradx,label_grady = sobel(fake_data, real_data)

    pred_g = torch.sqrt(torch.pow(pred_gradx,2) + torch.pow(pred_grady,2))
    label_g =torch.sqrt(torch.pow(label_gradx,2) + torch.pow(label_grady,2))
    img_grad = m_loss(pred_g,label_g)

    
    total_loss = loss + lambda_*error + 0*loss_perceptual_low + 0*loss_perceptual_high + img_grad
    total_loss.mean().backward()
    optimizer.step()
    return loss,error,total_loss,img_grad



def train_network(trainloader, testloader_flickr,testloader_div,testloader_urban, debug,num_epochs=200,K=10):
    discriminator = DiscriminativeNet()
    model=arch()
    model = nn.DataParallel(model, device_ids = device_ids)
    discriminator = nn.DataParallel(discriminator, device_ids= device_ids)
    model = model.to(device)
    model.apply(weight_init)
    discriminator=discriminator.to(device)
#     vgg_features_high=nn.DataParallel(VGGFeatureExtractor())
    vgg_features_high = nn.DataParallel(VGGFeatureExtractor())
    vgg_features_high.to(device)
       
    d_optimizer = optim.SGD(discriminator.parameters(), lr=0.000001, momentum=0.9)
    g_optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(g_optimizer, step_size=60, gamma=0.8,verbose=True)
    
    Mse_loss = nn.DataParallel(nn.MSELoss(),device_ids = device_ids).to(device)
    Bce_loss = nn.DataParallel(nn.BCEWithLogitsLoss(),device_ids = device_ids).to(device)
    criterion = nn.DataParallel(nn.SmoothL1Loss(),device_ids = device_ids).to(device)

    train_d=[]
    train_g = []
    best=0
    train_g_rec=[]
    train_g_dis=[]
    test=[]
    psnr_div=[]
    psnr_flickr=[]
    psnr_urban=[]
    train_rmse=[]
    train_psnr=[]
    Disriminator_Loss=[]
    Generator_Adver_Loss=[]
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
    result_file = os.path.join(checkpoints,"result.pt")
    
    # Initialising directory for Network Debugging
    net_debug = os.path.join(results,"Debug")
    if not os.path.exists(net_debug):
        os.makedirs(net_debug)
    
   
    # load model if exists
    if os.path.exists(result_file):
#         checkpoint_file1='/home/harsh.shukla/SRCNN/Weights/L1_28.52.pt'
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
                
                
    if os.path.exists((os.path.join(results,"PSNR_flickr.txt"))):
        dbfile = open(os.path.join(results,"PSNR_flickr.txt"), 'rb')      
        psnr_flickr = pickle.load(dbfile)
        dbfile = open(os.path.join(results,"PSNR_div.txt"), 'rb')      
        psnr_div = pickle.load(dbfile)
        dbfile = open(os.path.join(results,"PSNR_urban.txt"), 'rb')      
        psnr_urban = pickle.load(dbfile)
        dbfile = open(os.path.join(results,"Train.txt"), 'rb')      
        train_psnr = pickle.load(dbfile)  
        
        dbfile = open(os.path.join(results,"Best.txt"), 'rb')      
        best = pickle.load(dbfile)
        
        dbfile = open(os.path.join(results,"Discriminator.txt"), 'rb')      
        Disriminator_Loss = pickle.load(dbfile)
        dbfile = open(os.path.join(results,"Generator_Adversial.txt"), 'rb')      
        Generator_Adver_Loss = pickle.load(dbfile)
        

    for epoch in range(num_epochs):
        training_loss_d=[]
        training_rec_loss_g=[]
        training_dis_loss_g=[]
        training_loss_g = []
        training_loss_perp = []
        test_loss_flickr=[]
        test_loss_div=[]
        test_loss_urban=[]
        train_rec_rmse=[]
        count = 0
        list_no=0
        for input_,real_data in trainloader:
            if torch.cuda.is_available():
                input_    = input_.to(device)
                real_data = real_data.to(device)
            fake_data,_,__ = model(input_)
            fake_data.to(device)
            if count == K:
                d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer, real_data, fake_data,discriminator,Bce_loss)
                training_loss_d.append(d_error.mean().item())
                count = 0
            g_rec_error,g_dis_error,g_error,l_percp = train_generator(model,g_optimizer, fake_data, real_data, discriminator, Bce_loss, criterion,vgg_features_high)
            training_rec_loss_g.append(g_rec_error.mean().item())
            training_dis_loss_g.append(g_dis_error.mean().item())
            training_loss_g.append(g_error.mean().item())
            training_loss_perp.append(l_percp.mean().item())
            train_rec_rmse.append((Mse_loss(fake_data, real_data)).mean().item())
            count += 1 

        #Plotting Gradient Flow for both the models
        if(debug == True): 
            plot_grad_flow(net_debug,model.named_parameters(),"generator")
            plot_grad_flow(net_debug,discriminator.named_parameters(),"discriminator")
        
        with torch.set_grad_enabled(False):
            for local_batch, local_labels in testloader_urban:
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)
                output,_,__ = model(local_batch)
                output = output.to(device)
                local_labels.require_grad = False
                test_loss_urban.append(Mse_loss(output, local_labels).mean().item())
                
            for local_batch, local_labels in testloader_div:
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)
                output,residual,up = model(local_batch)
                output = output.to(device)
                local_labels.require_grad = False
                test_loss_div.append(Mse_loss(output, local_labels).mean().item())
                
            for local_batch, local_labels in testloader_flickr:
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)
                output,_,__ = model(local_batch)
                output = output.to(device)
                local_labels.require_grad = False
                test_loss_flickr.append(Mse_loss(output, local_labels).mean().item())
                
        if debug == True:
            label=im.fromarray(np.uint8(np.moveaxis(local_labels[0].cpu().detach().numpy(),0,-1))).convert('RGB')
            output=im.fromarray(np.uint8(np.moveaxis(output[0].cpu().detach().numpy(),0,-1))).convert('RGB')
            res = im.fromarray(np.uint8(np.moveaxis(residual[2].cpu().detach().numpy(),0,-1))).convert('RGB')
            up_ = im.fromarray(np.uint8(np.moveaxis(up[2].cpu().detach().numpy(),0,-1))).convert('RGB')
           




        scheduler.step()
        ##Creating Checkpoints
        if epoch % 1 == 0:
            torch.save({
                'generator_state_dict': model.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'g_state_dict': g_optimizer.state_dict(),
                'd_state_dict': d_optimizer.state_dict(),
                }, checkpoint_file)
            
        
        print("Epoch :",epoch )
        print("Discriminator Loss :",sum(training_loss_d)/len(training_loss_d))
        print("Generator Reconstruction Loss :",sum(training_rec_loss_g)/len(training_rec_loss_g))
        print("Generator Adversarial Loss :",sum(training_dis_loss_g)/len(training_dis_loss_g))
        print("Perceptual Loss :", sum(training_loss_perp)/len(training_loss_perp))
        print("Total Generator Loss:",sum(training_loss_g)/len(training_loss_g))
        print("D(X) :",d_pred_real.mean(), "D(G(X)) :",d_pred_fake.mean())
        Disriminator_Loss.append(sum(training_loss_d)/len(training_loss_d))
        Generator_Adver_Loss.append(sum(training_dis_loss_g)/len(training_dis_loss_g))
        
        print(len(test_loss_flickr),max(test_loss_flickr))
        psnr_flickr.append(10*math.log10(255*255/(sum(test_loss_flickr)/len(test_loss_flickr))))
        psnr_div.append(10*math.log10(255*255/(sum(test_loss_div)/len(test_loss_div))))
        psnr_urban.append(10*math.log10(255*255/(sum(test_loss_urban)/len(test_loss_urban))))
        train_rmse.append(sum(train_rec_rmse)/len(train_rec_rmse))
        train_psnr.append(10*math.log10(255*255/train_rmse[-1]))
        with open(os.path.join(results,"PSNR_flickr.txt"), 'wb') as f:
             pickle.dump(psnr_flickr,f )
        with open(os.path.join(results,"PSNR_div.txt"), 'wb') as f:
             pickle.dump(psnr_div,f )
        with open(os.path.join(results,"PSNR_urban.txt"), 'wb') as f:
             pickle.dump(psnr_urban,f )
        with open(os.path.join(results,"Train.txt"), 'wb') as f:
             pickle.dump(train_psnr,f )
                
        if best< psnr_div[-1]:
            best= psnr_flickr[-1]
            with open(os.path.join(results,"Best.txt"), 'wb') as f:
              pickle.dump(best,f )
            torch.save({
                'generator_state_dict': model.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'g_state_dict': g_optimizer.state_dict(),
                'd_state_dict': d_optimizer.state_dict(),
                }, result_file)
            res.save(os.path.join(results,str(epoch) + 'residual' +str(best) +'.png'))
            up_.save(os.path.join(results,str(epoch) + 'upsampled' +str(best) +'.png'))
#             label.save(os.path.join(results,str(epoch) + 'test_target' +str(best) +'.png'))
#             output.save(os.path.join(results,str(epoch) + 'test_output' + str(best)+'.png'))
       
        with open(os.path.join(results,"Discriminator.txt"), 'wb') as f:
             pickle.dump(Disriminator_Loss,f )
        with open(os.path.join(results,"Generator_Adversial.txt"), 'wb') as f:
             pickle.dump(Generator_Adver_Loss,f )
#         print("Epoch :",epoch, flush=True)

        print("Training Rmse loss : ",train_rmse[-1])
        print("Training PSNR : ",train_psnr[-1])
        print("Test loss for Flickr:",sum(test_loss_flickr)/len(test_loss_flickr),flush=True)
        print("Test loss for Div:",sum(test_loss_div)/len(test_loss_div),flush=True)
        print("Test loss for Urban:",sum(test_loss_urban)/len(test_loss_urban),flush=True)
        print("PSNR for Flickr :", 10*math.log10(255*255/(sum(test_loss_flickr)/len(test_loss_flickr))))
        print("PSNR for Div :", 10*math.log10(255*255/(sum(test_loss_div)/len(test_loss_div))))
        print("PSNR for Urban100 :", 10*math.log10(255*255/(sum(test_loss_urban)/len(test_loss_urban))))  
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
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--debug', help="Mode of Execution here")
    args = parser.parse_args()
    
    grad_flow_flag = False


    if args.debug == "debug": 
        print("Running in Debug Mode.....")
        grad_flow_flag = True
    
    
    
    trainloader, testloader_flickr,testloader_div,testloader_urban = process_and_train_load_data()
    print("Initialised Data Loader ....")
    train_network(trainloader, testloader_flickr,testloader_div,testloader_urban,grad_flow_flag)
    
