import os
from PIL import Image
import pandas as pd
import numpy as np
import cv2
from skimage import exposure, filters


def enhance_contour_texture(
        image,
        filter_size=50,
        highpass_gain=1,
        sharpen_mode='laplacian',
        blend_ratio=0.01,
        clahe_clip=0.001

):
    # 确保输入为浮点型并归一化到[0,1]
    if image.dtype == np.uint8:
        img = image.astype(np.float32) / 255.0
    else:
        img = image.copy()
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)

    # 步骤1: 频域高通滤波增强纹理
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    y, x = np.ogrid[:rows, :cols]
    dist = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)
    sigma = filter_size / 2
    highpass = 1 - np.exp(-(dist ** 2) / (2 * sigma ** 2))

    fshift *= highpass * highpass_gain
    f_ishift = np.fft.ifftshift(fshift)
    img_highpass = np.abs(np.fft.ifft2(f_ishift))

    # 步骤2: 空间域锐化增强轮廓
    if sharpen_mode == 'laplacian':
        laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
        sharpened = cv2.filter2D(img_highpass, -1, laplacian)
    elif sharpen_mode == 'unsharp':
        blurred = cv2.GaussianBlur(img_highpass, (5, 5), 2.0)
        sharpened = cv2.addWeighted(img_highpass, 1.5, blurred, -0.5, 0)

    # 步骤3: 与原图混合保留自然感
    enhanced = cv2.addWeighted(img, 1 - blend_ratio, sharpened, blend_ratio, 0)

    # 修复：裁剪到合法范围 [0, 1]
    enhanced = np.clip(enhanced, 0, 1)

    # 步骤4: 对比度限制直方图均衡化 (CLAHE)
    enhanced = exposure.equalize_adapthist(enhanced, clip_limit=clahe_clip)

    return (enhanced * 255).astype(np.uint8)

def get_weight_for_category(category):
    """根据类别返回权重"""

    if category == 0:
        return 1.15   # 建筑楼房
    elif category == 1:
        return 1.05    # road
    elif category == 2:
        return 0.95   # car
    else:
        return 0.85



def apply_coordinates_to_image(temp_img, data_segments, background_weight=1):
    """根据数据段中的坐标信息调整图像的权重矩阵并返回处理后的像素图"""
    th = 5.67032 * 10 ** (-8)
    weights = np.full(temp_img.shape, background_weight, dtype=float)

    for segment in data_segments:
        user_id = segment['user_id']
        coordinates = segment['coordinates']
        weight = get_weight_for_category(user_id)

        for i in range(0, len(coordinates)-1, 2):
            x_coord = float(coordinates[i])
            y_coord = float(coordinates[i + 1])

            x = int(x_coord * temp_img.shape[1])
            y = int(y_coord * temp_img.shape[0])

            x = min(max(x, 0), temp_img.shape[1] - 1)
            y = min(max(y, 0), temp_img.shape[0] - 1)

            weights[y, x] = weight

    processed_img = ((temp_img + 273.15) ** 4) * weights * th

    processed_img = (processed_img - np.min(processed_img)) / (np.max(processed_img) - np.min(processed_img)) * 255

    return processed_img.astype(float)

def read_label_data(label_file_path):
    """从标签文件读取数据，每个数据段由用户ID和坐标列表组成"""
    data_segments = []
    current_segment = None

    with open(label_file_path, 'r') as file:
        for line in file:
            if line.strip():
                parts = line.strip().split()
                if parts[0].isdigit():
                    if current_segment:
                        data_segments.append(current_segment)
                    current_segment = {'user_id': int(parts[0]), 'coordinates': [float(x) for x in parts[1:]]}
                elif current_segment:
                    current_segment['coordinates'].extend([float(x) for x in parts])

    if current_segment:
        data_segments.append(current_segment)

    return data_segments

def read_temperature_data(csv_file_path):

    temperature_data = pd.read_csv(csv_file_path, header=None).values.astype(float)
    return temperature_data

def process_images(temp_folder_path, label_folder_path, path_output_deviation):
    """处理指定目录下的所有图像，应用从标签文件中读取的数据"""
    for temp_img_filename in os.listdir(temp_folder_path):
        csv_file_path = os.path.join(temp_folder_path, temp_img_filename.replace('.jpg', '.csv'))
        label_file_path = os.path.join(label_folder_path, os.path.splitext(temp_img_filename)[0] + '.txt')

        if os.path.isfile(csv_file_path):
            temp_img = read_temperature_data(csv_file_path)

            temp_img = enhance_contour_texture(temp_img)

            if os.path.isfile(label_file_path):
                data_segments = read_label_data(label_file_path)
                temperature_value = apply_coordinates_to_image(temp_img, data_segments)


                output_filename = os.path.splitext(temp_img_filename)[0] + '.jpg'
                output_path = os.path.join(path_output_deviation, output_filename)

                enhance_temperature_value = np.clip(temperature_value, 0, 255).astype(np.uint8)
                denoised2_temperature_image = Image.fromarray(enhance_temperature_value)
                denoised2_temperature_image.save(output_path)

