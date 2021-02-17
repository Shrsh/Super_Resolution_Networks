import math
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image as im
import argparse
import warnings

warnings.filterwarnings("ignore")
from IPython.display import clear_output
from prettytable import PrettyTable

# CUDA for PyTorch
print("Number of GPUs:" + str(torch.cuda.device_count()))

use_cuda = torch.cuda.is_available()
torch.cuda.empty_cache()
device_ids = [i for i in range(torch.cuda.device_count())]
device = 'cuda' if use_cuda else 'cpu'
torch.backends.cudnn.benchmark = True

trans = transforms.ToPILImage()
trans1 = transforms.ToTensor()
import torch.nn.init as init


### Network Debugging
#########################################################################

### Creating function for Gradient Visualisation
def plot_grad_flow(result_directory, named_parameters, model_name):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.figure(figsize=(12, 12))
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.savefig(os.path.join(result_directory, model_name + "gradient_flow.png"))


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


def visualise_layer_activation(model, local_batch, result_directory):
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
    # plot subplots for different images
    for i in range(act[0].shape[0]):
        output = im.fromarray(np.uint8(np.moveaxis(act[0][i].cpu().detach().numpy(), 0, -1))).convert('RGB')
        output.save(os.path.join(result_directory, str(i) + '.png'))
        #


### Visualising Conv Filters
def visualise_conv_filters(model, result_directory):
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
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

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
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
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
        

#########################################################################################################################################################################################################


def load_images_from_folder(folder):
    c = 0
    images = []
    list_name = []
    for filename in os.listdir(folder):
        list_name.append(os.path.join(folder, filename))
    list_name.sort()
    for filename in list_name:
        img = cv2.imread(filename)
        if img is not None:
            images.append(img)
    return images


class SRSN(nn.Module):
    def __init__(self, input_dim=3, dim=128, scale_factor=4):
        super(SRSN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 128, 9, 1, 4)
        self.conv2 = torch.nn.Conv2d(128, 128, 1, 1, 0)
        self.resnet1 = Modified_Resnet_Block(dim, 13, 1, 6, bias=True)
        self.resnet2 = Modified_Resnet_Block(dim, 7, 1, 3, bias=True)
        self.resnet3 = Modified_Resnet_Block(dim, 5, 1, 2, bias=True)
        self.resnet4 = Modified_Resnet_Block(dim, 3, 1, 1, bias=True)

        # for specifying output size in deconv filter
        #       new_rows = ((rows - 1) * strides[0] + kernel_size[0] - 2 * padding[0] + output_padding[0])
        #       new_cols = ((cols - 1) * strides[1] + kernel_size[1] - 2 * padding[1] +output_padding[1])
        self.up = torch.nn.ConvTranspose2d(64, 64, 4, stride=4)
        #         self.up = torch.nn.Upsample(scale_factor=4, mode='bicubic')
        self.conv3 = torch.nn.Conv2d(64, 32, 1, 1, 0)
        self.conv4 = torch.nn.Conv2d(32,16,1,1,0)
        self.conv5 = torch.nn.Conv2d(16, 3, 1, 1, 0)
        
        self.conv_skip1 = torch.nn.Conv2d(128,64,1,1,0)
        self.conv_skip2 = torch.nn.Conv2d(64,64,3,1,1)
#         self.conv_skip3 = torch.nn.Conv2d(128,)

    def forward(self, LR):
        LR_feat = F.leaky_relu(self.conv1(LR))
        LR_feat = (F.leaky_relu(self.conv2(LR_feat)))
        ##Creating Skip connection between dense blocks
        out = self.resnet1(LR_feat)
        out1 = out + LR_feat
        out1 = self.resnet2(out1)
        out1 = out + out1

        out2 = self.resnet3(out1)
        out2 = out + out2

        out3 = self.resnet4(out2)
        out3 = out+out1+out2 + out3
#         print(out3.shape)
        out3=self.conv_skip1(out3)
        out3=self.conv_skip2(out3)
        out3 = self.up(out3)
        #         LR_feat = self.resnet(out3)
        SR = F.leaky_relu(self.conv3(out3))
        SR = F.leaky_relu(self.conv4(SR))
        
#         print(SR.shape)
        return self.conv5(SR)


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

        self.conv1_bn = nn.BatchNorm2d(num_filter)
        self.conv2_bn = nn.BatchNorm2d(num_filter)
        self.conv3_bn = nn.BatchNorm2d(num_filter)

        self.act1 = torch.nn.LeakyReLU(inplace=True)
        self.act2 = torch.nn.LeakyReLU(inplace=True)
        self.act3 = torch.nn.LeakyReLU(inplace=True)

    def forward(self, x):
        
        out = self.conv1(x)
#         print(x.shape)
#         print(out.shape)
        out = self.conv1_bn(out )
        out=self.act1(out)

        out1 = self.conv2(out+x)
#         out1 = x + out1 + out
        out1 = self.conv2_bn(out1)
        out1=self.act2(out1)

        out2 = self.conv3(out1+out+x)
#         out2 = out2 + out1 + out + x
        out2 = self.conv3_bn(out2)
        out2=self.act3(out2)

        return out2


def process_and_train_load_data():
    train = load_images_from_folder('/home/harsh.shukla/SRCNN/HR_LR_data/train/x')
    train_input = np.asarray(train)
    train_input = np.moveaxis(train_input, 1, -1)
    train_input = np.moveaxis(train_input, 1, -1)
    train_input = train_input.astype(np.float32)

    train = load_images_from_folder('/home/harsh.shukla/SRCNN/HR_LR_data/train/y')
    train_target = np.asarray(train)
    train_target = np.moveaxis(train_target, 1, -1)
    train_target = np.moveaxis(train_target, 1, -1)
    train_target = train_target.astype(np.float32)

    test = load_images_from_folder('/home/harsh.shukla/SRCNN/HR_LR_data/test/x')
    test_input = np.asarray(test)
    test_input = np.moveaxis(test_input, 1, -1)
    test_input = np.moveaxis(test_input, 1, -1)
    test_input = test_input.astype(np.float32)

    test = load_images_from_folder('/home/harsh.shukla/SRCNN/HR_LR_data/test/y')
    test_target = np.asarray(test)
    test_target = np.moveaxis(test_target, 1, -1)
    test_target = np.moveaxis(test_target, 1, -1)
    test_target = test_target.astype(np.float32)
    data_train = []
    data_test = []
    for input, target in zip(train_input, train_target):
        data_train.append([input, target])
    for input, target in zip(test_input, test_target):
        data_test.append([input, target])

    trainloader = torch.utils.data.DataLoader(dataset=data_train, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(dataset=data_test, batch_size=64, shuffle=True)
    return trainloader, testloader


def initialize_train_network(trainloader, testloader, debug):
    results = "/home/harsh.shukla/SRCNN/SRSN_results"

    if not os.path.exists(results):
        os.makedirs(results)

    # Initialising Checkpointing directory
    checkpoints = os.path.join(results, "Checkpoints")
    if not os.path.exists(checkpoints):
        os.makedirs(checkpoints)
    checkpoint_file = os.path.join(checkpoints, "check.pt")

    # Initialising directory for Network Debugging
    net_debug = os.path.join(results, "Debug")
    if not os.path.exists(net_debug):
        os.makedirs(net_debug)

    model = SRSN()
    model = nn.DataParallel(model, device_ids=device_ids)
    model.apply(weight_init)
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)
    criterion = nn.MSELoss().to(device)

    # load model if exists
    if os.path.exists(checkpoint_file):
        print("Loading from Previous Checkpoint...")
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.train()
    else:
        print("No previous checkpoints exist, initialising network from start...")

    ## Parameters in Networks
    print("Number of Parameters in Super Resolution Network")
    count_parameters(model)

    best_loss = 10000
    train = []
    test = []
    model = model.to(device)

    loss1 = 0
    for epoch in range(500):
        training_loss = []
        test_loss = []
        list_no = 0
        for input_, target in trainloader:
            if torch.cuda.is_available():
                input_ = input_.to(device)
                target = target.to(device)
            output = model(input_)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if debug:
                plot_grad_flow(net_debug, model.named_parameters(), "super_resolution_network")
            optimizer.zero_grad()
            training_loss.append(loss.item() * output.shape[0])

        with torch.set_grad_enabled(False):
            for local_batch, local_labels in testloader:
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)
                output = model(local_batch).to(device)
                local_labels.require_grad = False
                test_loss.append(criterion(output, local_labels).item())

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_file)
        if debug:
            label = im.fromarray(np.uint8(np.moveaxis(local_labels[0].cpu().detach().numpy(), 0, -1))).convert('RGB')
            output = im.fromarray(np.uint8(np.moveaxis(output[0].cpu().detach().numpy(), 0, -1))).convert('RGB')
            label.save(os.path.join(results, str(epoch) + 'test_target' + '.png'))
            output.save(os.path.join(results, str(epoch) + 'test_output' + '.png'))

        train.append(sum(training_loss) / len(training_loss))
        test.append(sum(test_loss) / len(test_loss))
        print("Epoch :", epoch, flush=True)
        print("Training loss :", sum(training_loss) / len(training_loss), flush=True)
        print("Test loss :", sum(test_loss) / len(test_loss), flush=True)
        print("PSNR : ", 10*math.log10(255*255/(sum(test_loss)/len(test_loss))))

        print(
            "-----------------------------------------------------------------------------------------------------------")
    try:
        file = open(os.path.join(results, "SR_train_loss.txt"), 'w+')
        try:
            for i in range(len(test)):
                file.write(str(train[i]) + "," + str(test[i]))
                file.write('\n')
        finally:
            file.close()
    except IOError:
        print("Unable to create loss file")
    print(
        "---------------------------------------------------------------------------------------------------------------")
    print("Training Completed")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--debug', help="Mode of Execution here")
    args = parser.parse_args()

    grad_flow_flag = False

    if args.debug == "debug":
        print("Running in Debug Mode.....")
        grad_flow_flag = True

    trainloader, testloader = process_and_train_load_data()
    initialize_train_network(trainloader, testloader, grad_flow_flag)


