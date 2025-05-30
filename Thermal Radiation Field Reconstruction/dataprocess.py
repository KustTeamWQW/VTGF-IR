from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
class IRDataset(Dataset):
    def __init__(self, segmentation_dir, temperature_dir, ir_dir, transform=None, temp_transform=None, text_transform=None):
        """
        Args:
            segmentation_dir (string): 路径到语义分割图像的文件夹。
            temperature_dir (string): 路径到温度图像的文件夹。
            ir_dir (string): 路径到实际红外图像的文件夹。
            segmentation_text_dir (string): 路径到语义分割文本数据的文件夹。
            transform (callable, optional): 可选的变换来应用于三通道样本。
            temp_transform (callable, optional): 可选的变换来应用于单通道样本。
            text_transform (callable, optional): 可选的变换来应用于语义分割文本数据。
        """
        self.segmentation_dir = segmentation_dir
        self.temperature_dir = temperature_dir
        self.ir_dir = ir_dir

        self.transform = transform
        self.temp_transform = temp_transform


        # 假设所有文件夹中的文件数量和命名都是对应的
        self.filenames = [f for f in os.listdir(segmentation_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        seg_path = os.path.join(self.segmentation_dir, self.filenames[idx])
        temp_path = os.path.join(self.temperature_dir, self.filenames[idx])
        ir_path = os.path.join(self.ir_dir, self.filenames[idx])


        seg_image = Image.open(seg_path).convert('RGB')  # 三通道模式
        temp_image = Image.open(temp_path).convert('L')  # 灰度模式
        ir_image = Image.open(ir_path).convert('RGB')  # 三通道模式


        if self.transform:
            seg_image = self.transform(seg_image)
            ir_image = self.transform(ir_image)
        if self.temp_transform:
            temp_image = self.temp_transform(temp_image)


        return seg_image, temp_image, ir_image
# 转换图像尺寸和归一化，分别适用于三通道和单通道图像
# 定义转换
transform = transforms.Compose([
    transforms.CenterCrop((128, 128)),  # 中心裁剪到 256x256
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

temp_transform = transforms.Compose([
    transforms.CenterCrop((128, 128)),  # 中心裁剪到 256x256
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
# 修改 transform 和 temp_transform 以适用于语义分割文本数据
text_transform = transforms.Compose([
    # 可以添加适当的文本预处理和转换步骤
])

def use_ir_dataset(segmentation_dir, temperature_dir, ir_dir):
    # 创建数据集实例
    dataset = IRDataset(segmentation_dir=segmentation_dir,
                        temperature_dir=temperature_dir,
                        ir_dir=ir_dir,
                        transform=transform,
                        temp_transform=temp_transform,)

    return dataset

# 实例化数据集
# dataset = IRDataset(segmentation_dir='path_to_segmentation_images',
#                     temperature_dir='path_to_temperature_images',
#                     ir_dir='path_to_ir_images',
#                     transform=transform,
#                     temp_transform=temp_transform)
