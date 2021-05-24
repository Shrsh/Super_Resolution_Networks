import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import os
from os import listdir
from os.path import isfile, join 
from PIL import Image
import cv2
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torchvision import transforms, utils
from torchvision.utils import save_image
import torch.utils.data as data
from PIL import Image 
import pickle
from PIL import Image as im
import warnings
warnings.filterwarnings("ignore")
from IPython.display import clear_output
import argparse 
import pickle as pkl
import math

use_cuda = torch.cuda.is_available()
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device_ids = [i for i in range(torch.cuda.device_count())]
device = 'cuda' if use_cuda else 'cpu'
torch.backends.cudnn.benchmark = True
from prettytable import PrettyTable

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

def process_and_train_load_data():
    test= load_images_from_folder('/home/harsh.shukla/SRCNN/SR_data_512/test/x')
#     test= load_images_from_folder('/home/harsh.shukla/SRCNN/SR_data_512/Urban/x')

    test_input=np.asarray(test)
    test_input=np.moveaxis(test_input,1,-1)
    test_input=np.moveaxis(test_input,1,-1)
    test_input = test_input.astype(np.float32)

    test= load_images_from_folder('/home/harsh.shukla/SRCNN/SR_data_512/test/y')
#     test= load_images_from_folder('/home/harsh.shukla/SRCNN/SR_data_512/Urban/y')
    test_target=np.asarray(test)
    test_target=np.moveaxis(test_target,1,-1)
    test_target=np.moveaxis(test_target,1,-1)
    test_target = test_target.astype(np.float32)
    
    data_test =[]

    for input, target in zip(test_input, test_target):
        data_test.append([input, target])
    testloader=torch.utils.data.DataLoader(dataset=data_test, batch_size=1, shuffle=False)
    
    return testloader


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
        x = F.relu(self.conv7(x))

        x = x.view(-1, 128*4*4)
        x = self.Fc1(x)
        x = self.Fc2(x)
        return x
    
class VGGFeatureExtractor(nn.Module):
    def __init__(self,
                 low_feature_layer=22,
                 high_feature_layer = 36,
                 use_bn=False,
                 device=torch.device('cuda')):
        super(VGGFeatureExtractor, self).__init__()
        if use_bn:
            model = torchvision.models.vgg19_bn(pretrained=True)
        else:
            model = torchvision.models.vgg19(pretrained=True)
        self.features_low = nn.Sequential(*list(model.features.children())[:(low_feature_layer + 1)])
        self.features_high = nn.Sequential(*list(model.features.children())[:(high_feature_layer + 1)])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.features_low(x), self.features_high(x)
    
class SRSN_RRDB(nn.Module):
    def __init__(self, input_dim=3, dim=128, scale_factor=4,scale_ratio=0.2):
        super(SRSN_RRDB, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 128, 9, 1, 4)
        self.conv2 = torch.nn.Conv2d(128, 64, 5, 1,2)
        self.RDB1 = ResidualInResidualDenseBlock(64, 64, 0.2)
        self.RDB2 =ResidualInResidualDenseBlock(64, 64, 0.2)
        self.RDB3 = ResidualDenseBlock(64, 64, 0.2)
        self.RDB4 = ResidualDenseBlock(64, 64, 0.2)
        self.RDB5 = ResidualDenseBlock(64, 64, 0.2)
        self.RDB6 = ResidualDenseBlock(64, 64, 0.2)
        
        self.up = torch.nn.Upsample(scale_factor=4, mode='bicubic')
        self.conv3=torch.nn.Conv2d(64, 16, 3, 1, 1)
        self.conv4=torch.nn.Conv2d(16, 3, 1, 1, 0)
        self.conv5=torch.nn.Conv2d(64*6, 64, 1, 1, 0)
        self.scale_ratio = 1

    def forward(self, LR):
        LR_feat = F.leaky_relu(self.conv1(LR),negative_slope=0.2)
        LR_feat = F.leaky_relu(self.conv2(LR_feat),negative_slope=0.2)
        
        out = self.RDB1(LR_feat) 
        out1= self.RDB2(out)
        
        out2 = self.RDB3(out1)        
        out3 = self.RDB4(out2)
        out3= out3 + LR_feat
        out4 = self.RDB5(out3)
        out5 = self.RDB6(out4)
        out6=torch.cat((out,out1,out2,out3,out4,out5), dim=1)
        out6=self.conv5(out6)
        out6= out6.mul(self.scale_ratio) + LR_feat
        out6 = self.up(out6)
        SR=F.leaky_relu(self.conv3(out6),negative_slope=0.2)
        SR =self.conv4(SR)
        return SR   
    
    
class ResidualDenseBlock(nn.Module):
    r"""The residual block structure of traditional SRGAN and Dense model is defined"""

    def __init__(self, channels: int = 64, growth_channels: int = 48, scale_ratio: float = 0.2):
        """

        Args:
            channels (int): Number of channels in the input image. (Default: 64).
            growth_channels (int): how many filters to add each layer (`k` in paper). (Default: 32).
            scale_ratio (float): Residual channel scaling column. (Default: 0.2)
        """
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels + 0 * growth_channels, growth_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels + 1 * growth_channels, growth_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels + 2 * growth_channels, growth_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(channels + 3 * growth_channels, growth_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv5 = nn.Conv2d(channels + 4 * growth_channels, channels, kernel_size=3, stride=1, padding=1)

        self.scale_ratio = scale_ratio


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        conv1 = self.conv1(input)
        conv2 = self.conv2(torch.cat((input, conv1), 1))
        conv3 = self.conv3(torch.cat((input, conv1, conv2), 1))
        conv4 = self.conv4(torch.cat((input, conv1, conv2, conv3), 1))
        conv5 = self.conv5(torch.cat((input, conv1, conv2, conv3, conv4), 1))

        return conv5.mul(self.scale_ratio) + input
    
class ResidualInResidualDenseBlock(nn.Module):
    r"""The residual block structure of traditional ESRGAN and Dense model is defined"""

    def __init__(self, channels: int = 64, growth_channels: int = 32, scale_ratio: float = 0.2):
        """

        Args:
            channels (int): Number of channels in the input image. (Default: 64).
            growth_channels (int): how many filters to add each layer (`k` in paper). (Default: 32).
            scale_ratio (float): Residual channel scaling column. (Default: 0.2)
        """
        super(ResidualInResidualDenseBlock, self).__init__()
        self.RDB1 = ResidualDenseBlock(channels, growth_channels, scale_ratio)
        self.RDB2 = ResidualDenseBlock(channels, growth_channels, scale_ratio)
        self.RDB3 = ResidualDenseBlock(channels, growth_channels, scale_ratio)

        self.scale_ratio = scale_ratio

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.RDB1(input)
        out = self.RDB2(out)
        out = self.RDB3(out)

        return out.mul(self.scale_ratio) + input
    
def prepare_test_report(testloader): 
    
    cp_addr = ['/home/harsh.shukla/SRCNN/Weights/L1_28.52.pt']
    test_results = "/home/harsh.shukla/SRCNN/results"
    generate_data = "/home/harsh.shukla/SRCNN/SR_data_512/Div"
    generate_data_x = "/home/harsh.shukla/SRCNN/SR_data_512/Div/x"
    generate_data_y = "/home/harsh.shukla/SRCNN/SR_data_512/Div/y"
    
    trainTransform  = torchvision.transforms.Compose([torchvision.transforms.Grayscale(num_output_channels=1)])
    
    discriminator = DiscriminativeNet()
    model=SRSN_RRDB()
    vgg_model = VGGFeatureExtractor()
    
    model = nn.DataParallel(model, device_ids = device_ids)
    discriminator = nn.DataParallel(discriminator, device_ids= device_ids)
    vgg_features_high = nn.DataParallel(vgg_model)

    model = model.to(device)
    discriminator=discriminator.to(device)
    vgg_features_high.to(device)
    
    d_optimizer = optim.SGD(discriminator.parameters(), lr=0.000001, momentum=0.9)
    g_optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(g_optimizer, step_size=60, gamma=0.5,verbose=True)
    
    Mse_loss = nn.DataParallel(nn.MSELoss(),device_ids = device_ids).to(device)
    Bce_loss = nn.DataParallel(nn.BCEWithLogitsLoss(),device_ids = device_ids).to(device)
    criterion = nn.DataParallel(nn.SmoothL1Loss(),device_ids = device_ids).to(device)
    
    up = torch.nn.Upsample(scale_factor=4, mode='bicubic')
    
    if not os.path.exists(test_results):
        os.makedirs(test_results)
        
    if not os.path.exists(generate_data): 
        os.makedirs(generate_data)
        
    if not os.path.exists(generate_data_x): 
        os.makedirs(generate_data_x) 
        
    if not os.path.exists(generate_data_y): 
        os.makedirs(generate_data_y)
    
    for i in range(len(cp_addr)): 
        Loss_list=[]
        bi_Loss_list = []
        results_ = os.path.join(test_results,str(i))
        if not os.path.exists(results_): 
            os.makedirs(results_)
        
        if os.path.exists(cp_addr[i]):
            print("Loading from Previous Checkpoint...")
            checkpoint = torch.load(cp_addr[i])
            model.load_state_dict(checkpoint['generator_state_dict'])
            discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            g_optimizer.load_state_dict(checkpoint['g_state_dict'])
            d_optimizer.load_state_dict(checkpoint['d_state_dict'])   
            model.train()
            discriminator.train()
        else:
            print("No checkpoints exist at specified directory")
        with torch.no_grad():
            for local_batch, local_labels in testloader:
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)
                output = model(local_batch).to(device)       
                bi_output = up(local_batch).to(device)
                local_labels.require_grad = False
                Loss_list.append(Mse_loss(trainTransform(torch.reshape(output, (3,768, 768))), trainTransform(torch.reshape(local_labels, (3,768, 768)))).mean().item())
                mse=Loss_list[-1]
                bi_Loss_list.append(Mse_loss(trainTransform(torch.reshape(bi_output, (3,768, 768))), trainTransform(torch.reshape(local_labels, (3,768, 768)))).mean().item())
#                 print("PSNR Bicubic :", 10*math.log10(255*255/bi_Loss_list[-1]))
#                 print("MSE : ",mse)
#                 print("PSNR :", 10*math.log10(255*255/mse))
                if 10*math.log10(255*255/Loss_list[-1])-10*math.log10(255*255/bi_Loss_list[-1]) > 3: 
                    for j in range(local_batch.size(0)):
                          
#                         local_batch=im.fromarray(np.uint8(np.moveaxis(local_batch[j].cpu().detach().numpy(),0,-1))).convert('RGB')
#                         local_batch.save(os.path.join(generate_data_x,str(10*math.log10(255*255/mse)) + '.png'))
#                         local_labels=im.fromarray(np.uint8(np.moveaxis(local_labels[j].cpu().detach().numpy(),0,-1))).convert('RGB')
#                         local_labels.save(os.path.join(generate_data_y,str(10*math.log10(255*255/mse)) + '.png'))
#                         output_=im.fromarray(np.uint8(np.moveaxis(output[j].cpu().detach().numpy(),0,-1))).convert('RGB')
#                         output_.save(os.path.join(results_,str(10*math.log10(255*255/mse)) +'.png'))
                        
                        
                else:
                    Loss_list.pop()
                    bi_Loss_list.pop()
            print("Average PSNR for bicubic and network ...")
            print("PSNR Bicubic :", 10*math.log10(255*255/(sum(bi_Loss_list)/len(bi_Loss_list))))
            print("MSE : ",sum(Loss_list)/len(Loss_list))
            print("PSNR :", 10*math.log10(255*255/(sum(Loss_list)/len(Loss_list))))



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--debug', help="Mode of Execution here")
    args = parser.parse_args()
    
    grad_flow_flag = False
    if args.debug == "debug": 
        print("Running in Debug Mode.....")
        grad_flow_flag = True
        
    testloader= process_and_train_load_data()
    print("Initialised Data Loader ....")
    print("Starting Networks in Eval Mode")
    prepare_test_report(testloader)
    print("Testing Completed")
