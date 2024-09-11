
import os
import torch
from PIL import Image
import numpy as np
from piqa import PSNR, SSIM
import pyiqa
import torchvision.transforms as transforms
first = True
first2 = True
lpips_metric = None
niqe_metric = None
config = None
device = torch.device("cuda:0")




def read_img(img_path, ref_image=None):
    img = Image.open(img_path).convert('RGB')
    #img = transforms.ToTensor()(img)
    #img = (np.asarray(img)/255.0)
    img = transforms.ToTensor()(img)

    #img = img.permute(2,0,1)
    img = img.to(device).unsqueeze(0)
    return img.contiguous()

def get_NIQE(enhanced_image, gt_path=None):
    niqe_metric = pyiqa.create_metric('niqe', device=enhanced_image.device).to(device)
    return  niqe_metric(enhanced_image)




path1 = r"C:\Users\Admin\Desktop\depthresult"#"F:\pythonDoc\py\GT_480_640"#"F:\pythonDoc\py\GT"


dir1 = os.listdir(path1)
for i in range(len(dir1)):
    dir1[i] = os.path.join(path1,dir1[i])


print(len(dir1))

ssim = 0
ssimList = []

for i in range(len(dir1)):

    image1 = read_img(dir1[i])



    num =get_NIQE(image1)#SSIM(image1, image2) #Brightness(image1,image2)#SSIM(image1, image2)
    print(num)
    ssim = ssim + num
    ssimList.append(num)
    if i % 100 == 0:
        print(i)
    # except:
    #     num=get_NIQE(image1)  # Brightness(image1,image2)#SSIM(image1, image2)
    #     # print("****************************")
    #     ssim = ssim + num
    #     ssimList.append(num)
    #
    #     print("出问题")
    #     print(i)
    #     continue




print("ave:"+str(ssim/len(dir1)))

