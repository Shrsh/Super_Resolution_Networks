import torch
import torchvision
import cv2
import os
import numpy as np
import math

use_cuda = torch.cuda.is_available()
torch.no_grad()
torch.cuda.empty_cache()
device_ids = [i for i in range(torch.cuda.device_count())]
device = 'cuda' if use_cuda else 'cpu'
torch.backends.cudnn.benchmark = True


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

    test= load_images_from_folder('/home/harsh.shukla/SRCNN/HR_LR_data/test/x_urban')
    test_input=np.asarray(test)
    print(test_input.shape)
    test_input=np.moveaxis(test_input,1,-1)
    test_input=np.moveaxis(test_input,1,-1)
    test_input = test_input.astype(np.float32)

    test= load_images_from_folder('/home/harsh.shukla/SRCNN/HR_LR_data/test/y_urban')
    test_target=np.asarray(test)
    test_target=np.moveaxis(test_target,1,-1)
    test_target=np.moveaxis(test_target,1,-1)
    test_target = test_target.astype(np.float32)
    data_test=[]
    for input, target in zip(test_input, test_target):
        data_test.append([input, target])

    test_loader=torch.utils.data.DataLoader(dataset=data_test, batch_size=32, shuffle=True)
    return test_loader


def test_bicubic_performance(test_loader):
    up = torch.nn.Upsample(scale_factor=4,mode="bicubic")
    criterion = torch.nn.MSELoss().to(device)
    global_loss = []
    with torch.set_grad_enabled(False):
        for local_batch, local_labels in test_loader:
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            output = up(local_batch)
            local_labels.require_grad = False
            loss = criterion(output,local_labels)
            global_loss.append(loss.item())

    print("PSNR : ", 10 * math.log10(255 * 255 / (sum(global_loss) / len(global_loss))))


if __name__ == '__main__':

    test_loader = process_and_train_load_data()
    print("Initialised Data Loaders")
    test_bicubic_performance(test_loader)
    print("=== Done ====")

