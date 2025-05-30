import cv2
import numpy as np


def ag(image):

    # 计算图像梯度
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # 计算梯度幅值
    grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # 计算平均梯度
    average_gradient = np.mean(grad_magnitude)

    return average_gradient

#
# # 示例用法
# image_path = 'path/to/your/image.jpg'
# contrast_score = compute_contrast(image_path)
# print(f"Contrast Score (Average Gradient): {contrast_score}")
