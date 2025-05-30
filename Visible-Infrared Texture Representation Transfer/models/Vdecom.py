import torch
import torch.nn as nn
from models.architecture import *
from torchvision import transforms
from PIL import Image
import numpy as np
import os
class VDecom(nn.Module):
    def __init__(self):
        super().__init__()
        self.decom = nn.Sequential(
            get_conv2d_layer(in_c=3, out_c=32, k=3, s=1, p=1),
            nn.LeakyReLU(0.2, inplace=True),
            get_conv2d_layer(in_c=32, out_c=32, k=3, s=1, p=1),
            nn.LeakyReLU(0.2, inplace=True),
            get_conv2d_layer(in_c=32, out_c=32, k=3, s=1, p=1),
            nn.LeakyReLU(0.2, inplace=True),
            get_conv2d_layer(in_c=32, out_c=4, k=3, s=1, p=1),
            nn.ReLU()
        )
    #
    # def forward(self, input):
    #     output = self.decom(input)
    #     R = output[:, 0:3, :, :]
    #     L = output[:, 3:4, :, :]
    #     return R, L

    def forward(self, input):

        output = self.decom(input)
        R = output[:, 0:3, :, :]
        L = output[:, 3:4, :, :]  # 注意这里L只取了一个通道，通常可能需要更多
        return R, L











