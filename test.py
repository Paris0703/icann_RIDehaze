#!/usr/bin/python3

import argparse
import itertools

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
from datasets import ImageDataset_test
import os
import numpy as np
import cv2
import torch.nn.functional as f

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = torch.device("cuda:0")
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


batchSize = 1
n_cpu = 0


netG_B2A = Generator(3,3)
netG_B2A.load_state_dict(torch.load(r"C:\Users\Admin\Desktop\paper2\paper2Code\weight\141_0.9456, _0.2964, .pth"))

netG_B2A.to(device, non_blocking=True).train()





# Dataset loader

               #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
dataloader = DataLoader(ImageDataset_test(),batch_size=batchSize, shuffle=True, num_workers=n_cpu,pin_memory=True,drop_last=True)


if __name__ == "__main__":

        factor = 4
        loop = tqdm(enumerate(dataloader), total=len(dataloader))


        for i, batch in loop:
            # Set model input

            HazeImage = batch['HazeImage'].to(device)
            ImageName = batch['ImageName']

            h, w = HazeImage.shape[2],HazeImage.shape[3]
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor * factor)
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            fake_Haze = f.pad(HazeImage, (0, padw, 0, padh), 'reflect')









            max_values_1, _ = torch.max(fake_Haze, dim=1)
            max_values_2, _ = torch.max(max_values_1, dim=1)
            max_values, _ = torch.max(max_values_2, dim=1)
            A = max_values.unsqueeze(1).unsqueeze(2).unsqueeze(3).detach()

            transmap = netG_B2A(fake_Haze)
            transmap = (A - fake_Haze + transmap) / A + 0.001
            recovered_Clear = (fake_Haze - A * (1 - transmap)) / transmap
            recovered_Clear= recovered_Clear[:, :, :HazeImage.shape[2], :HazeImage.shape[3]]

            tensor_example2 = recovered_Clear[:1, :3, :, :].squeeze(0) * 255 
            tensor_example2 = tensor_example2.byte()
            image_array2 = np.array(tensor_example2.cpu().permute(1, 2, 0))
            image_array2 = cv2.cvtColor(image_array2, cv2.COLOR_RGB2BGR)
            #cv2.imwrite(r"C:\Users\Admin\Desktop\paper2\paper2Code\reside\\"+ImageName[0],image_array2)
            cv2.imshow("dehaze", image_array2)

            tensor_example2 = fake_Haze[:1, :3, :, :].squeeze(0) * 255
            tensor_example2 = tensor_example2.byte()
            image_array2 = np.array(tensor_example2.cpu().permute(1, 2, 0))
            image_array2 = cv2.cvtColor(image_array2, cv2.COLOR_RGB2BGR)
            cv2.imshow("Haze", image_array2)




            cv2.waitKey(0)







