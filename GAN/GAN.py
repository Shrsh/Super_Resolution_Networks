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
import argparse 
import pickle as pkl

use_cuda = torch.cuda.is_available()
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device_ids = [i for i in range(torch.cuda.device_count())]
device = 'cuda' if use_cuda else 'cpu'
torch.backends.cudnn.benchmark = True
from prettytable import PrettyTable

### Network Debugging
#########################################################################

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

#########################################################################################################################################################################################################




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

def prepare_data(b_size=8):
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

################################################################

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
                stride=4, padding=0, bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),
              nn.BatchNorm2d(256)
        )
#         self.conv5 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=256, out_channels=512, kernel_size=4,
#                 stride=2, padding=1, bias=False
#             ),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.BatchNorm2d(512)
#         )
#         self.conv6 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=512, out_channels=256, kernel_size=4,
#                 stride=2, padding=1, bias=False
#             ),
#             nn.BatchNorm2d(256)
#         )
        self.conv7 = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=128, kernel_size=4,
                stride=4, padding=0, bias=False
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
#         x = self.conv5(x)
#         x = F.relu(self.conv6(x))
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
    def __init__(self, input_dim=3, dim=64, scale_factor=4):
        super(SRSN, self).__init__()
#         self.up = nn.ConvTranspose2d(3,3, 1, stride=1,output_size=(512,512))
        self.conv1 = torch.nn.Conv2d(3, 128, 9, 1, 4)
        self.conv2 = torch.nn.Conv2d(128, 64, 1, 1, 0)
        self.resnet1 = Modified_Resnet_Block(dim, 7, 1, 3, bias=True)
        self.resnet2 = Modified_Resnet_Block(dim, 7, 1, 3, bias=True)
        self.resnet3 = Modified_Resnet_Block(dim, 5, 1, 2, bias=True)
        self.resnet4 = Modified_Resnet_Block(dim, 3, 1, 1, bias=True)
        self.up = torch.nn.Upsample(scale_factor=4, mode='bicubic')
        self.conv3=torch.nn.Conv2d(64, 16, 1, 1, 0)
        self.conv4=torch.nn.Conv2d(16, 3, 1, 1, 0)

    def forward(self, LR):
        LR_feat = F.leaky_relu(self.conv1(LR))
        LR_feat = (F.leaky_relu(self.conv2(LR_feat)))
        
        ##Creating Skip connection between dense blocks 
        out = self.resnet1(LR_feat) 
        out = out + LR_feat
        out1= self.resnet2(out)
        out1= out + out1
        
        out2 = self.resnet3(out1)
        out2 = out + out2
        
        out3 = self.resnet4(out2)
        out3 = out + out1 + out3 
        out3 = self.up(out3)
        #LR_feat = self.resnet(out3)
        SR=F.leaky_relu(self.conv3(out3))
        SR =self.conv4(SR)
        # print(SR.shape)
        return SR

class ResnetBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=3, stride=1, padding=1, bias=True):
        super(ResnetBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)
        self.conv2 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)

        self.act1 = torch.nn.LeakyReLU(inplace=True)
        self.act2 = torch.nn.LeakyReLU(inplace=True)


    def forward(self, x):

        out = self.act1(x)
        out = self.conv1(out)

        out = self.act2(out)
        out = self.conv2(out)

        out = out + x

        return out
    
class Modified_Resnet_Block(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=3, stride=1, padding=1, bias=True):
        super(Modified_Resnet_Block, self).__init__()
        self.conv1 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)
        self.conv2 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)
        self.conv3 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)

        self.act1 = torch.nn.LeakyReLU(inplace=True)
        self.act2 = torch.nn.LeakyReLU(inplace=True)
        self.act3 = torch.nn.LeakyReLU(inplace=True)


    def forward(self, x):
        
        out = self.conv1(self.act1(x))
        out = out + x
        
        out1 = self.conv2(self.act2(out))
        out1 = x + out1 + out
        
        out2 = self.conv3(self.act3(out1))
        out2 = out2 + out1 + out + x

        return out2

def train_discriminator(optimizer, real_data, fake_data,discriminator,b_loss):
    optimizer.zero_grad()
    
    # 1. Train on Real Data
    prediction_real = discriminator(real_data)
    truth_real=Variable(torch.ones(real_data.size(0), 1))-0.1
    error_real = b_loss(prediction_real, truth_real.to(device))
    error_real.backward()

    # 2. Train on Fake Data
    prediction_fake = discriminator(fake_data.detach())
    error_fake = b_loss(prediction_fake, Variable(torch.zeros(real_data.size(0), 1)).to(device))
    error_fake.backward()
    optimizer.step()
    return error_real + error_fake, prediction_real, prediction_fake

def train_generator(model, optimizer, fake_data,real_data,discriminator,b_loss,m_loss):
    optimizer.zero_grad()
    
    lambda_ = 0.00001
    
    ##Reconstruction loss
    loss=m_loss(fake_data, real_data)
    ## Adversarial Loss 
    prediction = discriminator(fake_data)
    error = b_loss(prediction, Variable(torch.ones(real_data.size(0), 1)).to(device))
    total_loss = lambda_*loss + error 
    total_loss.backward()
#     clipping_value = 1 
#     torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)
    optimizer.step()
    return loss,error,total_loss



def train_network(debug,trainloader, testloader,num_epochs=200,K=7):

    discriminator = DiscriminativeNet()
    model=SRSN()
    model = model.to(device)
    discriminator=discriminator.to(device)

    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.00001, betas=(0.5, 0.999))
    g_optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)

    Mse_loss = nn.MSELoss().to(device)
    Bce_loss = nn.BCELoss().to(device)

    train_d=[]
    train_g = []
    train_g_rec=[]
    train_g_dis=[]
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
    
    # Initialising directory for Network Debugging
    net_debug = os.path.join(results,"Debug")
    if not os.path.exists(net_debug):
        os.makedirs(net_debug)
    
    model = nn.DataParallel(model, device_ids = device_ids)
    discriminator = nn.DataParallel(discriminator, device_ids= device_ids)
    # load model if exists
    if os.path.exists(checkpoint_file):
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

    for epoch in range(num_epochs):
        training_loss_d=[]
        training_rec_loss_g=[]
        training_dis_loss_g=[]
        training_loss_g = []
        test_loss=[]
        count = 0
        list_no=0
        for input_,real_data in trainloader:
            if torch.cuda.is_available():
                input_ = input_.to(device)
                real_data=real_data.to(device)

            fake_data = model(input_)
            if count == K:
                d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer, real_data, fake_data,discriminator,Bce_loss)
                training_loss_d.append(d_error.item())
                count = 0
            g_rec_error,g_dis_error,g_error = train_generator(model,g_optimizer, fake_data, real_data,discriminator,Bce_loss,Mse_loss)
            training_rec_loss_g.append(g_rec_error.item())
            training_dis_loss_g.append(g_dis_error.item())
            training_loss_g.append(g_error.item())
            count += 1 
        #Plotting Gradient Flow for both the models
        if(debug == True): 
            plot_grad_flow(net_debug,model.named_parameters(),"generator")
            plot_grad_flow(net_debug,discriminator.named_parameters(),"discriminator")
        
        with torch.set_grad_enabled(False):
            for local_batch, local_labels in testloader:
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)
                output = model(local_batch).to(device)
                local_labels.require_grad = False
                test_loss.append(Mse_loss(output, local_labels).item())
                
        if debug == True:
            label=im.fromarray(np.uint8(np.moveaxis(local_labels[0].cpu().detach().numpy(),0,-1))).convert('RGB')
            output=im.fromarray(np.uint8(np.moveaxis(output[0].cpu().detach().numpy(),0,-1))).convert('RGB')
            label.save(os.path.join(results,str(epoch) + 'test_target' + '.png'))
            output.save(os.path.join(results,str(epoch) + 'test_output' + '.png'))
        
        #Calculating average loss per epoch
        train_d.append(sum(training_loss_d)/len(training_loss_d))
        train_g_rec.append(sum(training_rec_loss_g)/len(training_rec_loss_g))
        train_g_dis.append(sum(training_dis_loss_g)/len(training_dis_loss_g))
        train_g.append(sum(training_loss_g)/len(training_loss_g))
        test.append(sum(test_loss)/len(test_loss))
        ##Creating Checkpoints
        if epoch % 5 == 0:
            torch.save({
                'generator_state_dict': model.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'g_state_dict': g_optimizer.state_dict(),
                'd_state_dict': d_optimizer.state_dict(),
                }, checkpoint_file)
            with open(os.path.join(results,"Generator_loss.txt"), 'wb') as f:
                pkl.dump(train_g ,f)
            with open(os.path.join(results,"Discriminator_loss.txt"), 'wb') as f:
                pkl.dump(train_d ,f)
            with open(os.path.join(results,"Test_loss.txt"), 'wb') as f:
                pkl.dump(test,f )
        print("Epoch :",epoch )
        print("Discriminator Loss :",train_d[-1])
        print("Generator Reconstruction Loss :",train_g_rec[-1])
        print("Generator Adversarial Loss :",train_g_dis[-1])
        print("Total Generator Loss:",train_g[-1])

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
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--debug', help="Mode of Execution here")
    args = parser.parse_args()
    
    grad_flow_flag = False


    if args.debug == "debug": 
        print("Running in Debug Mode.....")
        grad_flow_flag = True
    
    
    
    trainloader, testloader = prepare_data(b_size=23)
    print("Initialised Data Loaders")
    train_network(grad_flow_flag,trainloader, testloader,num_epochs=300)


