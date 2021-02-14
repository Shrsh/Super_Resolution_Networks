import os
from os import listdir
from os.path import isfile, join 
from PIL import Image

class pre_process: 
    test_path=""
    train_path=""
    
    def __init__(self,test_directory,train_directory): 
        self.test_path = test_directory 
        self.train_path = train_directory
        self.count = 0
    
    def process(self,string):
        self.count = 0
        if string == "train": 
            path = self.train_path 
            processed_input = "/scratch/harsh_cnn/SR_data/train/input"
            processed_target = "/scratch/harsh_cnn/SR_data/train/target"
#             processed_dir = "/home/harsh.shukla/SRCNN/SR_data/train/target"
        else:
            path = self.test_path
            processed_input = "/home/harsh.shukla/SRCNN/HR_LR_data/test/x"
            processed_target = "/home/harsh.shukla/SRCNN/HR_LR_data/test/y"
#             processed_dir = "/home/harsh.shukla/SRCNN/SR_data/test/target"
        
        os.chdir(path)
        walker = list(os.walk(path))
        filenames = walker[0][2]
        
#         os.mkdir(processed_dir)
#         print(filenames)
        for i in filenames:
            print(i)
            #center crop the images, 128*800
            im = Image.open(i)
            left = int(im.size[0]/2-256/2)
            upper = int(im.size[1]/2-256/2)
            right = left + 256
            lower = upper + 256
            im = im.crop((left, upper,right,lower))
            im.save(os.path.join(processed_input, str(self.count) + '.png'))
            im = im.resize((64,64))
            im.save(os.path.join(processed_target, str(self.count)  + '.png'))
            self.count+=1
#             if i.split('.')[0][-1]== path[-1] :
#                 im = im.resize((64,64))
#                 im = im.resize((128,128))
#                 im.save(os.path.join(processed_input,i.split('.')[0].split('_')[0]+i.split('.')[0].split('_')[1]+"_"+path[-1]+".jpg"))
#             else:
# #             print(name[-1])
#                 im.save(os.path.join(processed_target, i.split('.')[0].split('_')[0]+i.split('.')[0].split('_')[1]+"_"+path[-1]+".jpg"))
#             print(os.path.join(processed_dir, i))
            

##Div 2k
train_directory = "/scratch/harsh_cnn/train"
test_directory = "/home/harsh.shukla/SRCNN/test_Data/Urban100"

##Real SR
# train_directory = "/home/harsh.shukla/SRCNN/RealSR (ICCV2019)/Nikon/Train/2"
# test_directory = "/home/harsh.shukla/SRCNN/RealSR (ICCV2019)/Nikon/Test/2"

ob = pre_process(test_directory,train_directory)
# ob.process("train")
ob.process("test")