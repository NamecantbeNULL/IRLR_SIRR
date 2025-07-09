import time
import os
import torch
import cv2
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from options.train_options import TrainOptions
from data import create_dataset
from data.sirr_dataset import TestDataset
from models import create_model
from util.visualizer import Visualizer
from util import util
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import copy
import torch.nn.functional as F


class TestDataLoader():
    def __init__(self, opt):
        self.opt = opt
        self.dataset = TestDataset(opt)

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
            num_workers=int(opt.num_threads))

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i >= self.opt.max_dataset_size:
                break
            yield data


def PSNR_pt(target, output):
    mse_loss = F.mse_loss(output, target, reduction='none').mean((1, 2, 3))
    psnr = 10 * torch.log10(1 / mse_loss).mean()
    return psnr.item()


def ssim_pt(img, img2):
    """Calculate SSIM (structural similarity) (PyTorch version).

    It is called by func:`calculate_ssim_pt`.

    Args:
        img (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        img2 (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).

    Returns:
        float: SSIM result.
    """
    c1 = (0.01 * 255)**2
    c2 = (0.03 * 255)**2

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    window = torch.from_numpy(window).view(1, 1, 11, 11).expand(img.size(1), 1, 11, 11).to(img.dtype).to(img.device)

    mu1 = F.conv2d(img, window, stride=1, padding=0, groups=img.shape[1])  # valid mode
    mu2 = F.conv2d(img2, window, stride=1, padding=0, groups=img2.shape[1])  # valid mode
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img * img, window, stride=1, padding=0, groups=img.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, stride=1, padding=0, groups=img.shape[1]) - mu2_sq
    sigma12 = F.conv2d(img * img2, window, stride=1, padding=0, groups=img.shape[1]) - mu1_mu2

    cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    ssim_map = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map
    return ssim_map.mean([1, 2, 3])


def compare_save(metrics, visuals):
    fake_Ts = visuals['fake_Ts']
    real_T = visuals['real_T']
    fake_T0 = torch.clamp(fake_Ts[0], min=0, max=1)
    fake_T1 = torch.clamp(fake_Ts[1], min=0, max=1)
    fake_T2 = torch.clamp(fake_Ts[2], min=0, max=1)
    fake_T3 = torch.clamp(fake_Ts[3], min=0, max=1)
    real_T = torch.clamp(real_T, min=0, max=1)

    # metrics["ssim0"] += structural_similarity(real_T, fake_T0, gaussian_weights=True, data_range=255, multichannel=True, channel_axis=2)
    metrics["ssim0"] += ssim_pt(fake_T0.to(torch.float64) * 255., real_T.to(torch.float64) * 255.)
    # metrics["psnr0"] += peak_signal_noise_ratio(real_T, fake_T0, data_range=255)
    metrics["psnr0"] += PSNR_pt(real_T, fake_T0)
    metrics["ssim1"] += 0.
    metrics["psnr1"] += 0.
    metrics["ssim2"] += 0.
    metrics["psnr2"] += 0.
    # metrics["ssim3"] += structural_similarity(real_T, fake_T3, gaussian_weights=True, data_range=255, multichannel=True, channel_axis=2)
    metrics["ssim3"] += ssim_pt(fake_T3.to(torch.float64) * 255., real_T.to(torch.float64) * 255.)
    # metrics["psnr3"] += peak_signal_noise_ratio(real_T, fake_T3, data_range=255)
    metrics["psnr3"] += PSNR_pt(real_T, fake_T3)

    return metrics

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_test = TestDataLoader(opt)
    event_dir = os.path.join(opt.checkpoints_dir, opt.name)
    timestr = datetime.now().strftime('%Y%m%d-%H%M%S')
    writer = SummaryWriter(event_dir + '/' + timestr)

    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)
    print('The number of test images = %d' % len(dataset_test))

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        for i, data in enumerate(dataset):  # inner loop within one epoch
            torch.cuda.empty_cache()
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            # if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
            #     save_result = total_iters % opt.update_html_freq == 0
            #     model.compute_visuals()
            #     visualizer.display_current_results(model.get_current_visuals(), total_iters, save_result)

            if total_iters % opt.print_freq == 0:   # print training losses and save logging information to the disk
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                losses = model.get_current_losses()
                visualizer.print_current_losses(epoch, epoch_iter, model.get_current_losses(), t_comp, t_data)
                writer.add_scalars('losses', {'total_T': losses['T'], 'idt_T': losses['idt_T'], 'ssim': losses['ssim'],
                                              'total_R': losses['R'], 'MP': losses['MP'],
                                              'Composition':losses['comp'], 'CLIP': losses['CLIP'],
                                              'CLIP_MSE': losses['CLIP_MSE'], 'Margin':losses['margin']}, total_iters)
                # if opt.display_id > 0:
                #     visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, model.get_current_losses())
            iter_data_time = time.time()

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            metrics = {"ssim0": 0, "psnr0": 0, "ssim1": 0, "psnr1": 0, "ssim2": 0, "psnr2": 0, "ssim3": 0, "psnr3": 0}
            for i, data in enumerate(dataset_test):
                model.set_input(data)  # unpack data from data loader
                model.test()  # run inference
                visuals = model.get_current_visuals()  # get image results
                img_path = model.get_image_paths()  # get image paths
                metrics = compare_save(metrics, visuals)
            result = open(os.path.join(opt.checkpoints_dir, opt.name, 'score.txt'), 'a')
            result.write(
                "Epc: %03d | SSIM0: %.3f PSNR0: %.2f | SSIM1: %.3f PSNR1: %.2f | SSIM2: %.3f PSNR2: %.2f | SSIM3: %.3f PSNR3: %.2f \n"
                % (epoch, metrics["ssim0"] / len(dataset_test), metrics["psnr0"] / len(dataset_test),
                   metrics["ssim1"] / len(dataset_test), metrics["psnr1"] / len(dataset_test),
                   metrics["ssim2"] / len(dataset_test), metrics["psnr2"] / len(dataset_test),
                   metrics["ssim3"] / len(dataset_test), metrics["psnr3"] / len(dataset_test)))
            writer.add_scalars('psnr', {'input': metrics["psnr0"] / len(dataset_test),
                                        'output': metrics["psnr3"] / len(dataset_test)}, epoch)
            writer.add_scalars('ssim', {'input': metrics["ssim0"] / len(dataset_test),
                                        'output': metrics["ssim3"] / len(dataset_test)}, epoch)
            result.close()
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
