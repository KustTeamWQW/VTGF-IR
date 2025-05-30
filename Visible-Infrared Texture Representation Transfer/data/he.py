import numpy as np
from PIL import Image
from skimage import exposure as ex

def he(img):
    # 将PIL图像对象转换为NumPy数组
    if isinstance(img, Image.Image):
        img_array = np.array(img)
    else:
        raise ValueError("Input must be a PIL Image object")

    # 检查输入是否为NumPy数组
    if not isinstance(img_array, np.ndarray):
        raise ValueError("Input must be a NumPy array")

    if img_array.ndim == 2:  # 灰度图
        outImg = ex.equalize_hist(img_array) * 255
    elif img_array.ndim == 3:  # 彩色图
        for channel in range(img_array.shape[2]):
            img_array[:, :, channel] = ex.equalize_hist(img_array[:, :, channel]) * 255
        outImg = img_array

    outImg = np.clip(outImg, 0, 255)  # 限制值在0-255之间
    outImg = outImg.astype(np.uint8)  # 转换为uint8类型

    # 将处理后的NumPy数组转换回PIL图像对象
    return Image.fromarray(outImg)

# 在paired_dataset.py文件中，你可以直接调用he函数
# J_img = he(J_img)  # J_img在这里是一个PIL图像对象