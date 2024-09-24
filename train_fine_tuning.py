#!/usr/bin/python3

import argparse
import itertools
import random

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
from net import *
import torch.nn.functional as F
from utils import ReplayBuffer
from utils import LambdaLR
#from utils import Logger
from tqdm import tqdm
#from utils import weights_init_normal
from datasets import ImageDataset
import os
import math
import numpy as np
import cv2
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = torch.device("cuda:0")
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


init_lr = 0.0001
batchSize = 8
n_cpu = 8
netG_B2A = Generator(3,3)
netG_B2A.to(device).train()

netDepth1 = torch.load(r"F:\pythonDoc\wavelet-monodepth-main\wavelet-monodepth-main\KITTI\encoder.pt").eval().to(device)
netDepth2 = torch.load(r"F:\pythonDoc\wavelet-monodepth-main\wavelet-monodepth-main\KITTI\decoder.pt").eval().to(device)


#netG_B2A.load_state_dict(torch.load(r"C:\Users\Admin\Desktop\paper\code\weight\192_2.0019, _.pth"))

criterion_cycle = torch.nn.MSELoss()#MSELoss()
criterion_identity = torch.nn.MSELoss()#MSELoss()
class GaussianBlur(nn.Module):
    def __init__(self, channels, kernel_size=11, sigma=1.5):
        super(GaussianBlur, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.sigma = sigma

        # 创建高斯核
        self.kernel = self.create_gaussian_kernel(kernel_size, sigma)
        # 卷积层
        self.conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=kernel_size // 2, groups=channels, bias=False)

        # 将高斯核赋值给卷积层
        with torch.no_grad():
            self.conv.weight = nn.Parameter(self.kernel)

    def create_gaussian_kernel(self, kernel_size, sigma):
        # 创建高斯核
        kernel = torch.tensor([[(1 / (2 * math.pi * sigma ** 2)) * math.exp(-((x - kernel_size//2) ** 2 + (y - kernel_size//2) ** 2) / (2 * sigma ** 2))
                                for x in range(kernel_size)] for y in range(kernel_size)])
        kernel = kernel / torch.sum(kernel)  # 归一化
        kernel = kernel.view(1, 1, kernel_size, kernel_size)
        kernel = kernel.repeat(self.channels, 1, 1, 1)
        return kernel

    def forward(self, x):
        return self.conv(x)

gaussian_blur = GaussianBlur(channels=1, kernel_size=17, sigma=5).to(device)
def getHazeImage(net1,net2,image):
    with torch.no_grad():
        max_values_1, _ = torch.max(image[:, :3, :, :], dim=1)

        max_values_2, _ = torch.max(max_values_1, dim=1)
        max_values, _ = torch.max(max_values_2, dim=1)
        A = max_values.unsqueeze(1).unsqueeze(2).unsqueeze(3)

        features = net1(image)
        outputs = net2(features)
        depth = outputs[("disp", 0)]

        min_val = torch.min(depth)
        max_val = torch.max(depth)


        depth = torch.div(torch.sub(depth, min_val), torch.sub(max_val, min_val))
        depth = gaussian_blur(depth)
        depth = 1-depth

        # tensor_example2 = depth[:1, :3, :, :].squeeze(0).clamp(0, 1) * 255  # (tensor_example2 - tensor_example2.min()) / (tensor_example2.max() - tensor_example2.min()) * 255
        # tensor_example2 = tensor_example2.byte()
        # image_array2 = np.array(tensor_example2.cpu().permute(1, 2, 0))
        # image_array2 = cv2.cvtColor(image_array2, cv2.COLOR_RGB2BGR)
        #
        # cv2.imshow("dehaze", image_array2)
        # cv2.waitKey(0)

        beta = random.uniform(0.1,2)





    return (torch.exp(-1 * depth * beta) * (image - A) + A)

def getClearImage(net,image):

    # max_values_1, _ = torch.max(image[:, :3, :, :], dim=1)
    #
    # max_values_2, _ = torch.max(max_values_1, dim=1)
    # max_values, _ = torch.max(max_values_2, dim=1)
    # A = max_values.unsqueeze(1).unsqueeze(2).unsqueeze(3).detach()
    # A = 1
    # transmap = net(image)+0.0001
    # clearImage = (image - A*(1-transmap))/transmap

    clearImage = net(image)

    return clearImage


optimizer_G1 = torch.optim.AdamW(netG_B2A.parameters(), lr =init_lr, betas=(0.5, 0.999))
def lr_schedule_cosdecay(t, T, init_lr=init_lr, end_lr=0.0000001):
    lr = init_lr+end_lr-t*init_lr/T#end_lr + 0.5 * (init_lr - end_lr) * (1 + math.cos(t * math.pi / T))
    return lr

dataloader = DataLoader(ImageDataset(),batch_size=batchSize, shuffle=True, num_workers=n_cpu,pin_memory=True,drop_last=True)



if __name__ == "__main__":

        for param_group in optimizer_G1.param_groups:
            print(f"Learning rate: {param_group['lr']}")
            break
        epoch = 0
        lossa = 0
        losscycle = 0
        lossg = 0
        lossd = 0
        losssame = 0
        load = False
        loop = tqdm(enumerate(dataloader), total=len(dataloader))
        threshold = 10

        oneTensor = torch.ones([batchSize,3,256,256],requires_grad=False).to(device)
        NoneNum = 0
        torch.save(netG_B2A.state_dict(), r'C:\Users\Admin\Desktop\paper\code\weight\\latest' + '.pth')
        for i, batch in loop:
            if i%10 == 1:
                clearImage = batch['clearimage'].to(device).detach()
                hazeImage = batch['hazeimage'].to(device).detach()


                optimizer_G1.zero_grad()

                fakeHaze = getHazeImage(netDepth1, netDepth2, clearImage).detach()
                reClear = getClearImage(netG_B2A, fakeHaze)
                loss_cycle_ABA = criterion_cycle(reClear, clearImage)


                sameClear= getClearImage(netG_B2A,clearImage)
                lossSame = criterion_cycle(sameClear, clearImage)

                (loss_cycle_ABA  + lossSame).backward()


                losscycle = losscycle + loss_cycle_ABA.detach()  # +loss_cycle_BAB.detach()
                losssame = losssame + lossSame.detach()
                loop.set_description(f'finalEpoch [{epoch}/{300}]')
                loop.set_postfix(lossa=lossa, loss_gan=lossg, losscycle=losscycle,lossd= lossd,losssame = losssame*5)
            else:
                clearImage = batch['clearimage'].to(device).detach()
                hazeImage = batch['hazeimage'].to(device).detach()

                optimizer_G1.zero_grad()

                fakeHaze = getHazeImage(netDepth1, netDepth2, clearImage).detach()
                reClear = getClearImage(netG_B2A, fakeHaze)
                loss_cycle_ABA = criterion_cycle(reClear, clearImage)

                loss_cycle_ABA.backward()

                losscycle = losscycle + loss_cycle_ABA.detach()  # +loss_cycle_BAB.detach()

                loop.set_description(f'finalEpoch [{epoch}/{300}]')
                loop.set_postfix(lossa=lossa, loss_gan=lossg, losscycle=losscycle,lossd= lossd,losssame = losssame*5)

            if (i % 1000 == 0):

                lr = lr_schedule_cosdecay(i, len(dataloader))
                for param_group in optimizer_G1.param_groups:
                    param_group["lr"] = lr


                epoch = epoch + 1

                for name, param in netG_B2A.named_parameters():
                    print(f"{name}: {optimizer_G1.param_groups[0]['lr']}")
                    break

                torch.save(netG_B2A.state_dict(), r'C:\Users\Admin\Desktop\paper\code\weight\\'+str(epoch)+ "_" + str(losscycle)[7:15]+ "_" + str(lossa)[7:15]+ '.pth')

                lossa = 0
                losscycle = 0
                losssame = 0

