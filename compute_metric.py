import os, cv2
from os.path import join
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from torchvision import transforms

gt_dir_SIR2_Postcard = '/data/doublebin/SIRR/test/generalization_SIR2_Postcard/real_T'
input_dir_SIR2_Postcard = '/data/doublebin/SIRR/test/generalization_SIR2_Postcard/real_I'
result_dir_SIR2_Postcard = './results/IRLR/test_50/images'

gt_dir_SIR2_SolidObject = '/data/doublebin/SIRR/test/generalization_SIR2_SolidObject/real_T'
input_dir_SIR2_SolidObject = '/data/doublebin/SIRR/test/generalization_SIR2_SolidObject/real_I'
result_dir_SIR2_SolidObject = './results/IRLR/test_50/images'

gt_dir_SIR2_In_the_wild = '/data/doublebin/SIRR/test/generalization_SIR2_In_the_wild/real_T'
input_dir_SIR2_In_the_wild = '/data/doublebin/SIRR/test/generalization_SIR2_In_the_wild/real_I'
result_dir_SIR2_In_the_wild = './results/IRLR/test_50/images'

gt_dir_SIR2_Kaggle = '/data/doublebin/SIRR/test/generalization_Kaggle/real_T'
input_dir_SIR2_Kaggle = '/data/doublebin/SIRR/test/generalization_Kaggle/real_I'
result_dir_SIR2_Kaggle = './results/IRLR/test_50/images'

gt_dir_real20 = '/data/doublebin/SIRR/test/real20/real_T'
input_dir_real20 = '/data/doublebin/SIRR/test/real20/real_I'
result_dir_real20 = './results/IRLR/test_final/images'

gt_dir_Nature = '/data/doublebin/SIRR/test/Nature/real_T'
input_dir_Nature = '/data/doublebin/SIRR/test/Nature/real_I'
result_dir_Nature = './results/IRLR/test_final/images'


gt_dir = gt_dir_Nature
input_dir = input_dir_Nature
result_dir = result_dir_Nature
img_lsit = os.listdir(gt_dir)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


psnr_total_before = AverageMeter()
ssim_total_before = AverageMeter()
psnr_total_after = AverageMeter()
ssim_total_after = AverageMeter()

for img in img_lsit:
    img_gt = Image.open(join(gt_dir, img))
    img_input = Image.open(join(input_dir, img))
    img_result = Image.open(join(result_dir, img[:-4] + '_fake_Ts_03.png'))
    w, h = img_result.size
    resize = transforms.Resize([h, w])

    img_gt_np = np.asarray(resize(img_gt), np.float32)
    img_input_np = np.asarray(resize(img_input), np.float32)
    img_result_np = np.asarray(img_result, np.float32)

    ssim_before = structural_similarity(img_gt_np, img_input_np, gaussian_weights=True, data_range=255,
                                        multichannel=True, channel_axis=2)
    ssim_after = structural_similarity(img_gt_np, img_result_np, gaussian_weights=True, data_range=255,
                                       multichannel=True, channel_axis=2)

    psnr_before = peak_signal_noise_ratio(img_gt_np, img_input_np, data_range=255)
    psnr_after = peak_signal_noise_ratio(img_gt_np, img_result_np, data_range=255)

    psnr_total_before.update(psnr_before)
    ssim_total_before.update(ssim_before)
    psnr_total_after.update(psnr_after)
    ssim_total_after.update(ssim_after)

    print('[INF] quantitative result: PSNR_bafore:{}, SSIM_bafore:{}, PSNR_after:{}, SSIM_after:{}'.format(
        psnr_before, ssim_before, psnr_after, ssim_after))
print('[AVE] quantitative result: PSNR_bafore:{}, SSIM_bafore:{}, PSNR_after:{}, SSIM_after:{}'.format(
            psnr_total_before.avg, ssim_total_before.avg, psnr_total_after.avg, ssim_total_after.avg))
