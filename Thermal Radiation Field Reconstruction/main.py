import os
from fushejisuan import process_images
from args_options import args

def apply_coordinates_to_image( ):
    # 可见光和红外图像的文件夹路径
    path_dataset_temp = args.dataset_temp
    path_dataset_segtxt = args.dataset_segtxt
    path_output_deviation = args.output_deviation
    if not os.path.exists(path_output_deviation):
        os.makedirs(path_output_deviation)

    process_images(path_dataset_temp, path_dataset_segtxt,path_output_deviation)

# 如果这个脚本是作为主程序运行的（而不是被导入为模块），则调用main函数
if __name__ == "__main__":
    apply_coordinates_to_image()