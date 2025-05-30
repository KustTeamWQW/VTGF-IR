import numpy as np
import cv2
# import pywt
from util.Contrast_Evaluation import CE
from util.piqe import piqe
from util.niqe import niqe
from util.ag import ag
from data.he import he

def convert_to_grayscale(img):
    # 将图像转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def fuse_images(img1, img2):
    score_img1piqe = 1/piqe(img1)
    score_img2piqe = 1/piqe(img2)
    fuseWeightpiqe = score_img1piqe / (score_img1piqe + score_img2piqe)

    score_img1ag = ag(img1)
    score_img2ag = ag(img2)
    fuseWeightag = score_img1ag / (score_img1ag + score_img2ag)

    score_img1niqe = 1/niqe(img1)
    score_img2niqe = 1/niqe(img2)
    fuseWeightniqe = score_img1niqe / (score_img1niqe + score_img2niqe)

    score_img1CE = CE(img1)
    score_img2CE = CE(img2)
    fuseWeightCE = score_img1CE / (score_img1CE + score_img2CE)

    fuseWeight = (fuseWeightpiqe + fuseWeightag+ fuseWeightniqe + fuseWeightCE) / 4
    # fuseWeight = (fuseWeightag  + fuseWeightCE) / 3

    return img1*fuseWeight + img2*(1-fuseWeight)

#
# def evaluate_image_quality(img):
#     # 计算图像质量分数
#     # score_piqe = 1 - piqe(img)
#     score_ag = ag(img)
#     # score_niqe = 1 - niqe(img)
#     score_CE = CE(img)
#
#     # return score_piqe, score_ag, score_niqe, score_CE
#     return  score_ag, score_CE
#
# def fuse_block(img1, img2):
#     # 计算每个图像块的质量分数
#     scores_img1 = evaluate_image_quality(img1)
#     scores_img2 = evaluate_image_quality(img2)
#
#     # 计算融合权重
#     fuseWeights = [(s1 - s2) / (s1 + s2) for s1, s2 in zip(scores_img1, scores_img2)]
#     fuseWeight = np.mean(fuseWeights)
#
#     # 根据权重融合图像块
#     return (img1 * fuseWeight) + (img2 * (1 - fuseWeight))
#
#
# def fuse_images(img1, img2):
#     block_size = 27
#     # 将图像转换为灰度图像
#     # gray1 = convert_to_grayscale(img1)
#     # gray2 = convert_to_grayscale(img2)
#
#     # 获取图像的宽度和高度
#     height, width = img1.shape[:2]
#
#     # 创建一个数组来存储融合后的图像
#     fused_image = np.zeros_like(img1)
#
#     # 逐块进行图像融合
#     for i in range(0, height, block_size):
#         for j in range(0, width, block_size):
#             # 计算当前块的边界
#             top, bottom = min(i + block_size, height), height
#             left, right = min(j + block_size, width), width
#
#             # 提取当前块
#             block1 = img1[top:bottom, left:right]
#             block2 = img2[top:bottom, left:right]
#             # gray_block1 = gray1[top:bottom, left:right]
#             # gray_block2 = gray2[top:bottom, left:right]
#
#             # 融合当前块
#             fused_block = fuse_block(block1, block2)
#
#             # 将融合后的块存储回fused_image中
#             fused_image[top:bottom, left:right] = fused_block
#
#     return fused_image