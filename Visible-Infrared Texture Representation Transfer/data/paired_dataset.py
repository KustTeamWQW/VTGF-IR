import os.path
from data.base_dataset import BaseDataset, get_transform, get_params
from data.image_folder import make_dataset
from PIL import Image
import random
from models.get_vtex import process_image
import numpy as np
from data.adjust_contrast import adjust_brightness_mean
from data.clahe import apply_clahe_to_image


class pairedDataset(BaseDataset):
    def __init__(self, opt):

        BaseDataset.__init__(self, opt)
        self.dir_I = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_J = os.path.join(opt.dataroot, opt.phase + 'B')
        self.I_paths = sorted(make_dataset(self.dir_I, opt.max_dataset_size))
        self.J_paths = sorted(make_dataset(self.dir_J, opt.max_dataset_size))
        self.I_size = len(self.I_paths)
        self.J_size = len(self.J_paths)

    def __getitem__(self, index):
        # 如果我们想要强制连续索引返回成对的图像
        if (index + 1) < len(self):  # 确保下一个索引在数据集范围内
            I_path = self.I_paths[index]
            J_path = self.J_paths[index]  # 假设I_paths和J_paths有相同的长度或索引有意义
        else:
            # 如果索引超出范围（例如，在最后一个元素时），则随机选择
            I_path = self.I_paths[index % self.I_size]
            if self.opt.serial_batches:
                J_path = self.J_paths[index % self.J_size]
            else:
                J_path = self.J_paths[random.randint(0, self.J_size - 1)]

        # 将灰度图转换为伪RGB图
        def to_pseudo_rgb(image):
            # 确保图像是灰度图
            if image.mode != 'L':
                image = image.convert('L')

                # 将图像数据转换为numpy数组
            gray_array = np.array(image)

            # 复制灰度值到RGB三个通道
            rgb_array = np.dstack((gray_array, gray_array, gray_array))

            # 将numpy数组转换回PIL图像
            rgb_image = Image.fromarray(rgb_array.astype('uint8'), 'RGB')

            return rgb_image

        base_name_I = os.path.splitext(os.path.basename(I_path))[0]
        base_name_J = os.path.splitext(os.path.basename(J_path))[0]
        I_img = Image.open(I_path)  # 假设I_img是灰度图或任何类型的图
        J_img = Image.open(J_path)  # 假设J_img也是灰度图或任何类型的图

        I_img = apply_clahe_to_image(I_img,clip_limit=1.3)
        I_img = process_image(I_img, base_name_I, (11, 11), 1.6, 0.25, np.pi / 9)

        J_img = apply_clahe_to_image(J_img,clip_limit=1.3)
        J_img = adjust_brightness_mean(J_img)
        J_img = process_image(J_img, base_name_J, (11, 11), 1.6, 0.25, np.pi / 9)

        I_img = process_image(I_img, base_name_I, (11, 11), 1.6, 0.25, np.pi / 9)


        I_img = to_pseudo_rgb(I_img)
        J_img = to_pseudo_rgb(J_img)

        # 随机裁剪图像
        params = {}  # 确保params是一个空字典

        self.crop_size = (300, 300)  # 设置裁剪尺寸为256x256
        self.do_crop = True

        if self.do_crop and self.crop_size:
            # 从两个图像中裁剪相同区域
            crop_width, crop_height = self.crop_size
            # 生成一个随机的裁剪位置
            left = np.random.randint(0, I_img.width - crop_width)
            top = np.random.randint(0, I_img.height - crop_height)
            # 使用相同的裁剪位置裁剪两张图像
            I_img = I_img.crop((left, top, left + crop_width, top + crop_height))
            J_img = J_img.crop((left, top, left + crop_width, top + crop_height))

        transform_I = get_transform(self.opt, params=params, grayscale=(self.opt.input_nc == 1))
        transform_J = get_transform(self.opt, params=params, grayscale=(self.opt.output_nc == 1))

        # apply image transformation
        real_I = transform_I(I_img)
        real_J = transform_J(J_img)

        return {'infrared': real_I, 'visible': real_J, 'paths': I_path, 'J_paths': J_path}

    def __len__(self):
        return max(self.I_size, self.J_size)
