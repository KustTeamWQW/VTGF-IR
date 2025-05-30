import cv2
import numpy as np
from PIL import Image

def adjust_brightness_mean(low_light_img):
    # 内置的参考图像路径
    normal_img_path = 'datasets/Reference_image/4472.jpg'  # 确保这是正确的路径

    # 读取参考图像，并转换为灰度图
    normal_img = cv2.imread(normal_img_path, cv2.IMREAD_GRAYSCALE)  # 以灰度模式读取

    # 检查图像是否成功读取
    if normal_img is None:
        print(f"Error: Unable to read the image at path: {normal_img_path}")
        print("Please check the file path and ensure the file is a valid image.")
        return None

    # 确保low_light_img是NumPy数组
    if isinstance(low_light_img, Image.Image):
        low_light_img = np.array(low_light_img)
    elif not isinstance(low_light_img, np.ndarray):
        raise ValueError("low_light_img must be a PIL Image or a NumPy array")

    # 获取正常图像的平均亮度
    mean_normal = np.mean(normal_img)

    # 获取低光图像的平均亮度
    mean_low_light = np.mean(low_light_img)

    # 计算调整系数
    if mean_low_light == 0:
        brightness_factor = 1
    else:
        brightness_factor = mean_normal / mean_low_light

    # 应用调整
    adjusted_img = np.clip(low_light_img * brightness_factor, 0, 255).astype(np.uint8)

    # 将调整后的NumPy数组转换回PIL图像对象
    adjusted_img_pil = Image.fromarray(adjusted_img)

    return adjusted_img_pil

# 假设你已经有了一个低光图像的PIL对象 J_img
# J_img = adjust_brightness_mean(J_img)
# 保存调整后的图像
# J_img.save('adjusted_image.jpg')