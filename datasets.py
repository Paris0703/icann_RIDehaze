import glob
import random
import os
import random
import torch
from collections import defaultdict
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from net import *
from tqdm import tqdm
import numpy as np
import cv2
from natsort import ns, natsorted
from torchvision.transforms import ToTensor, RandomCrop, Resize
from PIL import Image, ImageOps
from torchvision.transforms.functional import hflip, rotate, crop

class ImageDataset(Dataset):
    def __init__(self):

        self.transformclear = transforms.Compose([  # transforms.Resize(int(256 * 1.8)),
            transforms.RandomCrop(256, pad_if_needed=1),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),

        ])

        self.transformhaze = transforms.Compose([  # transforms.Resize(int(256 * 1.8)),
            transforms.RandomCrop(256, pad_if_needed=1),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        self.clearimages = natsorted(glob.glob(os.path.join(r"F:\pythonDoc\py\GT") + '/*.*'),alg=ns.PATH)
        self.hazeimages = natsorted(glob.glob(os.path.join(r"F:\UnannotatedHazyImages\UnannotatedHazyImages") + '/*.*'),alg=ns.PATH)


    def __getitem__(self, index):

        clearimage = Image.open(self.clearimages[random.randint(0, len(self.clearimages) - 1)]).convert("RGB")
        hazeimage = Image.open(self.hazeimages[random.randint(0, len(self.hazeimages) - 1)]).convert("RGB")


        clearimage = self.transformclear(clearimage)
        hazeimage = self.transformhaze(hazeimage)

        return {'clearimage': clearimage, 'hazeimage': hazeimage}

    def __len__(self):
        return min(len(self.clearimages), len(self.hazeimages))*300







class ImageDataset_test(Dataset):
    def __init__(self):



        self.filesName = r"F:\pythonDoc\py\Hazy"
        self.filesHaze = os.listdir(self.filesName)



    def pad_to_min_size(self,img, min_width, min_height):
        width, height = img.size
        if width < min_width or height < min_height:
            pad_width = max(0, min_width - width)
            pad_height = max(0, min_height - height)
            padding = (0, 0, pad_width, pad_height)
            img = ImageOps.expand(img, padding)
        return img

    def __getitem__(self, idx):



        ImageName = self.filesHaze[idx]



        HazeImage = Image.open(self.filesName+r"\\"+ ImageName).convert("RGB")


        HazeImage = ToTensor()(HazeImage)



        return {'ImageName': ImageName,"HazeImage":HazeImage}





    def __len__(self):
       return len(self.filesHaze)














