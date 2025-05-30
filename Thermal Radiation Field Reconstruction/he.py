import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure as ex
import imageio
import os
from PIL import Image



# def he(img):
#     if len(img.shape) == 2:  # gray
#         outImg = ex.equalize_hist(img) * 255
#     elif len(img.shape) == 3:  # RGB
#         outImg = np.zeros((img.shape[0], img.shape[1], 3))
#         for channel in range(img.shape[2]):
#             outImg[:, :, channel] = ex.equalize_hist(img[:, :, channel]) * 255
#
#     outImg[outImg > 255] = 255
#     outImg[outImg < 0] = 0
#     return outImg.astype(np.uint8)
#
#
# def process_images_in_directory(directory_path, output_directory):
#     # 确保输出目录存在
#     if not os.path.exists(output_directory):
#         os.makedirs(output_directory)
#
#     # 遍历目录中的所有文件
#     for filename in os.listdir(directory_path):
#         if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
#             image_path = os.path.join(directory_path, filename)
#             img = imageio.imread(image_path)
#             result = he(img)
#
#             # 保存处理后的图像
#             output_path = os.path.join(output_directory, filename)
#             imageio.imwrite(output_path, result)
#             print(f"Processed and saved: {output_path}")




def he(img):
    if img.ndim == 2:  # gray
        outImg = ex.equalize_hist(img) * 255
    elif img.ndim == 3:  # RGB
        for channel in range(img.shape[2]):
            img[:, :, channel] = ex.equalize_hist(img[:, :, channel]) * 255
        outImg = img

    outImg = np.clip(outImg, 0, 255)  # 限制值在0-255之间
    return outImg.astype(np.uint8)
# 使用函数处理目录下的图像
# 假设你的图像目录是 'path_to_your_images_directory'
# 假设你的输出目录是 'path_to_your_output_directory'
# process_images_in_directory('path_to_your_images_directory', 'path_to_your_output_directory')