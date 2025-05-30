import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from PIL import Image
from scipy.ndimage import convolve
def gabor_kernel(kernel_size, sigma, frequency, theta):

    ksize = kernel_size[0]
    half_ksize = ksize // 2

    # 创建网格坐标
    y, x = np.mgrid[-half_ksize:half_ksize + 1, -half_ksize:half_ksize + 1].astype(np.float32)

    # 转换为极坐标
    radius = np.sqrt(x ** 2 + y ** 2)
    theta_rad = np.arctan2(y, x)

    # 计算Gabor核
    gabor = np.exp(-(radius ** 2 / (2.0 * sigma ** 2))) * np.cos(2.0 * np.pi * frequency * radius + theta)

    # 确保Gabor核的和为1（归一化）
    gabor /= np.sum(gabor)

    return gabor


def apply_gabor_filter(image, kernel, mode='constant', cval=0.0):
    filtered_image = convolve(image, kernel, mode=mode, cval=cval)
    return filtered_image

def process_image(image,base_name,kernel_size,sigma,frequency,theta):
    save_dir = './datasets2/v1/gabor'
    # 确保保存文件夹存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 将图像转换为NumPy数组
    image = image.convert('L')
    image_np = np.array(image)

    # 定义Gabor滤波器参数
    # kernel_size = (9, 9)  # 核大小
    # sigma = 2  # 高斯函数的标准差
    # frequency = 0.25  # 正弦波的频率
    # theta = np.pi / 9  # 方向（45度）
    kernel_size = kernel_size
    sigma = sigma
    frequency = frequency
    theta = theta
    # 创建Gabor滤波器核
    gabor_kernel_real = gabor_kernel(kernel_size, sigma, frequency, theta)

    # 应用Gabor滤波器到图像
    gabor_filtered_image_np = apply_gabor_filter(image_np, gabor_kernel_real)

    # 确保值在0-255之间，并转换为uint8
    gabor_filtered_image_np = np.clip(gabor_filtered_image_np, 0, 255).astype(np.uint8)


    # 保存处理后的图像
    filename = f'{base_name}_gabor_filtered.jpg'
    save_path = os.path.join(save_dir, filename)
    cv2.imwrite(save_path, gabor_filtered_image_np)


    gabor_filtered_image_pil = Image.fromarray(gabor_filtered_image_np)
    # 返回处理后的图像
    return gabor_filtered_image_pil

