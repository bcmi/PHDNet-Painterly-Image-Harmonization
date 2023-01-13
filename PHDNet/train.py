import time
from options.train_options import TrainOptions
from data import CustomDataset
from models import create_model
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

import os
from util import util
import numpy as np
import torch
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from tqdm import tqdm

from util.visualizer import Visualizer
from PIL import Image, ImageFile


Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
# Disable OSError: image file is truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
def calculateMean(vars):
    return sum(vars) / len(vars)

def evaluateModel(model, opt, test_dataloader, epoch, visualizer):
    model.netG.eval()
    total_iters = 0
    model.test()  # inference
    for i, data in tqdm(enumerate(test_dataloader)):  # inner loop within one epoch
        iter_start_time = time.time()  # timer for computation per iteration
        total_iters += 1
        model.set_input(data)         # unpack data from dataset
        model.forward()   # calculate loss functions, get gradients, update network weights
        if total_iters % opt.display_freq == 0:
            visual_dict = model.get_current_visuals()
            visualizer.display_current_results(visual_dict, epoch)
            # visualizer.display_current_results(visual_dict, total_iters)


def resolveResults(results):
    interval_metrics = {}
    mask, mse, psnr = np.array(results['mask']), np.array(results['mse']), np.array(results['psnr'])
    interval_metrics['0.00-0.05'] = [np.mean(mse[np.logical_and(mask <= 0.05, mask > 0.0)]),
                                     np.mean(psnr[np.logical_and(mask <= 0.05, mask > 0.0)])]
    interval_metrics['0.05-0.15'] = [np.mean(mse[np.logical_and(mask <= 0.15, mask > 0.05)]),
                                     np.mean(psnr[np.logical_and(mask <= 0.15, mask > 0.05)])]
    interval_metrics['0.15-0.25'] = [np.mean(mse[np.logical_and(mask <= 0.25, mask > 0.15)]),
                                     np.mean(psnr[np.logical_and(mask <= 0.25, mask > 0.15)])]
    interval_metrics['0.25-0.50'] = [np.mean(mse[np.logical_and(mask <= 0.5, mask > 0.25)]),
                                     np.mean(psnr[np.logical_and(mask <= 0.5, mask > 0.25)])]
    interval_metrics['0.50-1.00'] = [np.mean(mse[mask > 0.5]), np.mean(psnr[mask > 0.5])]
    return interval_metrics

def updateWriterInterval(writer, metrics, epoch):
    for k, v in metrics.items():
        writer.add_scalar('interval/{}-MSE'.format(k), v[0], epoch)
        writer.add_scalar('interval/{}-PSNR'.format(k), v[1], epoch)

if __name__ == '__main__':
    # setup_seed(6)

    opt = TrainOptions().parse()   # get training 
    visualizer = Visualizer(opt,'train')
    visualizer_test = Visualizer(opt,'test')

    train_dataset = CustomDataset(opt, is_for_train=True)
    test_dataset = CustomDataset(opt, is_for_train=False)

    train_dataset_size = len(train_dataset)    # get the number of images in the dataset.
    test_dataset_size = len(test_dataset)
    print('The number of training images = %d' % train_dataset_size)
    print('The number of testing images = %d' % test_dataset_size)
    
    train_dataloader = train_dataset.load_data()
    test_dataloader = test_dataset.load_data()
    print('The total batches of training images = %d' % len(train_dataset.dataloader))

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    total_iters = 0                # the total number of training iterations
    writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name))

    for epoch in tqdm(range(int(opt.load_iter)+1, opt.niter + opt.niter_decay + 1)):
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        for i, data in enumerate(tqdm(train_dataloader)):  # inner loop within one epoch
            if i > len(train_dataset.dataloader) -2:
                continue
            # print('epoch {} iter {}'.format(epoch, i))
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_iters += 1
            epoch_iter += 1
            model.set_input(data)         # unpack data from dataset
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                writer.add_scalar('./loss/loss_content', losses['c'], i + 1)
                writer.add_scalar('./loss/loss_style', losses['s'], i + 1)
                writer.add_scalar('./loss/loss_tv', losses['tv'], i + 1)
                writer.add_scalar('./loss/loss_mask',losses['mask'], i + 1)
                writer.add_scalar('./loss/loss_GAN',losses['G_GAN'], i + 1)
                writer.add_scalar('./loss/loss_D',losses['D'], i + 1)
            
            if total_iters % opt.display_freq == 0:
                visual_dict = model.get_current_visuals()
                visualizer.display_current_results(visual_dict, epoch)
                # visualizer.display_current_results(visual_dict, total_iters)

                
            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        # evaluate for every epoch
        evaluateModel(model, opt, test_dataloader, epoch,visualizer_test)

        torch.cuda.empty_cache()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks('%d' % epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
        for scheduler in model.schedulers:
            print('Current learning rate: {}'.format(scheduler.get_lr()))

    writer.close()
