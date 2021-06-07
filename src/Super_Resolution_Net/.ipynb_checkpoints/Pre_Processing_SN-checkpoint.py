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
            processed_input = "/home/harsh.shukla/SRCNN/SR_data_512/train/input"
            processed_target = "/home/harsh.shukla/SRCNN/SR_data_512/train/target"
            processed_dir = "/home/harsh.shukla/SRCNN/SR_data_512/train"
        else:
            path = self.test_path
            processed_input = "/home/harsh.shukla/SRCNN/training_test_data/Urban/input"
            processed_target = "/home/harsh.shukla/SRCNN/training_test_data/Urban/target"
            processed_dir = "/home/harsh.shukla/SRCNN/training_test_data/Urban/"
        
        os.chdir(path)
        walker = list(os.walk(path))
        filenames = walker[0][2]
        
        os.mkdir(processed_dir)
#         if string = "train":
#             os.mkdir(os.path.join(processed_dir,"train"))
#         else: 
#             os.mkdir(os.path.join(processed_dir,"test"))
        os.mkdir(processed_input)
        os.mkdir(processed_target)
#         print(filenames)
        list_x=[]
        list_y=[]
        c=0
        for i in filenames:
            
            im = Image.open(i)
            print(im.size)
            if im.size[1]>=768 and im.size[0]>=768:
#                 print(im.size)
                print(c)
                c=c+1
                left = int(im.size[0]/2-768/2)
                upper = int(im.size[1]/2-768/2)
                right = left + 768
                lower = upper + 768
                im = im.crop((left, upper,right,lower))
                im.save(os.path.join(processed_target, str(c)  + '.png'))
                im = im.resize((192,192))
                im.save(os.path.join(processed_input, str(c) + '.png'))
                
                

##Div 2k
train_directory = "/scratch/harsh_cnn/train"
test_directory = "/home/harsh.shukla/SRCNN/test_Data/Urban100"

##Real SR
# train_directory = "/home/harsh.shukla/SRCNN/RealSR (ICCV2019)/Nikon/Train/2"
# test_directory = "/home/harsh.shukla/SRCNN/RealSR (ICCV2019)/Nikon/Test/2"

ob = pre_process(test_directory,train_directory)
# ob.process("train")
ob.process("test")