import os, time
import ntpath
import numpy as np
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util import util
from util.fusion_3 import fuse_images


if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1
    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)

    if opt.save_image:
        curSaveFolder = os.path.join(opt.dataroot, opt.method_name)
        if not os.path.exists(curSaveFolder):
            os.makedirs(curSaveFolder, mode=0o777)
    if opt.eval:
        model.eval()

    time_total = 0
    for i, data in enumerate(dataset):
        # if i <= 627:
        #     continue

        img_path = data['paths']
        short_path = ntpath.basename(img_path[0])
        name = os.path.splitext(short_path)[0]
        print('%s [%d]' % (short_path, i + 1))
        # print(data['B_paths'])


        t0 = time.time()
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference
        time_total += time.time() - t0

        visuals = model.get_current_visuals()  # get image results

        rec_J = util.tensor2im(visuals['rec_J'], np.float64) / 255.  # [0, 1]
        rec_I = util.tensor2im(visuals['rec_I'], np.float64) / 255.  # [0, 1]
        refine_J = util.tensor2im(visuals['refine_J'], np.float64) / 255.  # [0, 1]
        real_I = util.tensor2im(data['infrared'], np.float64)  # [0, 255], np.float

        fused_J = fuse_images(rec_J * 255., refine_J * 255.) / 255.
        # fused_J = image_fusion2(rec_J * 255., refine_J * 255.) / 255.
        fusedImg = (fused_J * 255).astype(np.uint8)



        import os
        import numpy as np

        # 假设您有这四个目录的名称
        dir_fuse = os.path.join(curSaveFolder, 'fuse_images')
        dir_refine = os.path.join(curSaveFolder, 'refine_images')
        dir_I = os.path.join(curSaveFolder, 'rec_I_images')
        dir_retinex = os.path.join(curSaveFolder, 'retinex_images')

        # 确保这些目录存在
        for dir_name in [dir_fuse, dir_refine, dir_I, dir_retinex]:
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

                # 保存图像
        util.save_image(fusedImg, os.path.join(dir_fuse, '%s_fuse.png' % (name)))
        util.save_image((refine_J * 255).astype(np.uint8), os.path.join(dir_refine, '%s_refine.png' % (name)))
        util.save_image((rec_I * 255).astype(np.uint8), os.path.join(dir_I, '%s_I.png' % (name)))
        util.save_image((rec_J * 255).astype(np.uint8), os.path.join(dir_retinex, '%s_retinex.png' % (name)))
        # save result images
        # if opt.save_image:

    print('num: %d' % len(dataset))
    print('average time: %f' % (time_total / len(dataset)))