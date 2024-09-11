import os
import torch
import cv2 as cv
import torch.nn.functional as F
from multiprocessing.pool import ThreadPool
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def SSIM(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = torch.from_numpy(img1).to(device, dtype=torch.float32).permute(2, 0, 1)  # Permute channels to the front
    img2 = torch.from_numpy(img2).to(device, dtype=torch.float32).permute(2, 0, 1)  # Permute channels to the front
    window_size = 11#101

    window = torch.ones((1, 3, window_size, window_size), device=device, dtype=torch.float32) / (window_size * window_size)
    mu1 = F.conv2d(img1.unsqueeze(0), window, stride=1, padding=window_size//2)
    mu2 = F.conv2d(img2.unsqueeze(0), window, stride=1, padding=window_size//2)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1.unsqueeze(0) ** 2, window, stride=1, padding=window_size//2) - mu1_sq
    sigma2_sq = F.conv2d(img2.unsqueeze(0) ** 2, window, stride=1, padding=window_size//2) - mu2_sq
    sigma12 = F.conv2d(img1.unsqueeze(0) * img2.unsqueeze(0), window, stride=1, padding=window_size//2) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if 0 <= ssim_map.mean().item() <= 0.99999:
        return ssim_map.mean().item()
    else:
        return 1.


path1 = r"F:\pythonDoc\py\GT"#r"F:\Haze4K-T\GT"#"F:\pythonDoc\py\GT_480_640"#"F:\pythonDoc\py\GT"
path2 = r"C:\Users\Admin\Desktop\paper\endnote\PSD\result_reside"

dir1 = os.listdir(path1)
for i in range(len(dir1)):
    dir1[i] = os.path.join(path1,dir1[i])

dir2 = os.listdir(path2)
for i in range(len(dir2)):
    dir2[i] = os.path.join(path2,dir2[i])

print(len(dir1))
print(len(dir2))
ssim = 0
ssimList = []

for i in range(len(dir1)):
    image2 = cv.imread(dir2[i])
    image1 = cv.imread(dir1[i])

    try:

        num =SSIM(image1, image2) #Brightness(image1,image2)#SSIM(image1, image2)
        ssim = ssim + num
        ssimList.append(num)
        if i % 100 == 0:
            print(i)
    except:
        num = SSIM(image1, image2)  # Brightness(image1,image2)#SSIM(image1, image2)
        # print("****************************")
        ssim = ssim + num
        ssimList.append(num)

        print("出问题")
        print(i)
        continue




print("ave:"+str(ssim/len(dir2)))


