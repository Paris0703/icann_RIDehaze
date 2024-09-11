from PIL import Image
import numpy as np
import os
import concurrent.futures  # Importing the concurrent.futures module for multi-threading

def compute_PSNR(img_path1, img_path2):
    img1 = np.array(Image.open(img_path1).convert("RGB"), dtype="float64")
    img2 = np.array(Image.open(img_path2).convert("RGB"), dtype="float64")
    mse = np.mean((img1 - img2) ** 2)

    if mse == 0:
        return 100#float('inf')
    else:
        return 20 * np.log10(255 / np.sqrt(mse))

def compute_average_PSNR(directory1, directory2):
    psnr_sum = 0
    psnr_list = []
    def process_image(i):
        try:
            img1_path = os.path.join(directory1[i])
            img2_path = os.path.join(directory2[i])
            num = compute_PSNR(img1_path, img2_path)
            # print("*********")
            # print(num)
            # print(img1_path)
            # print("*********")
            psnr_list.append(num)
            return num
        except Exception as e:
            print(f"Error processing image {i}: {e}")
            return 0

    with concurrent.futures.ThreadPoolExecutor() as executor:
        psnr_results = list(executor.map(process_image, range(len(directory1))))

    for result in psnr_results:
        psnr_sum += result

    sorted_nums = sorted(enumerate(psnr_list), key=lambda x: x[1])
    print(sorted_nums)

    print("平均PSNR")
    print(psnr_sum / len(directory2))

# Specify your paths
path1 = r"F:\pythonDoc\py\GT"#r"F:\Haze4K-T\GT"#"F:\pythonDoc\py\GT_480_640"#"F:\pythonDoc\py\GT"
path2 = r"C:\Users\Admin\Desktop\paper\endnote\PSD\result_reside"

# List the directories
dir1 = [os.path.join(path1, filename) for filename in os.listdir(path1)]
dir2 = [os.path.join(path2, filename) for filename in os.listdir(path2)]

print(len(dir1))
print(len(dir2))

# Call the function to compute average PSNR using multi-threading
compute_average_PSNR(dir1, dir2)


