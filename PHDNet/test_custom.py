import os
from os.path import realpath
from options.test_options import TestOptions
import torch
import numpy as np
from util import util
from util.visualizer import Visualizer
from PIL import Image, ImageFile
from data import CustomDataset
from models import create_model
import os
from tqdm import tqdm
import time


opt = TestOptions().parse()   # get training 
opt.isTrain = False
visualizer = Visualizer(opt,'test')
test_dataset = CustomDataset(opt, is_for_train=False)
test_dataset_size = len(test_dataset)
print('The number of testing images = %d' % test_dataset_size)
test_dataloader = test_dataset.load_data()

model = create_model(opt)      # create a model given opt.model and other options
model.setup(opt)               # regular setup: load and print networks; create schedulers
model.netG.eval()              # inference
total_iters = 0
iter_data_time = time.time()
save_dir = opt.save_dir
for i, data in enumerate(tqdm(test_dataloader)):  # inner loop within one epoch
    img_name = data['img_path'][0].split('/')[-1]
    total_iters += 1
    #if i > 70:
    #   break

    iter_start_time = time.time()  # timer for computation per iteration
    if total_iters % opt.print_freq == 0:
        t_data = iter_start_time - iter_data_time
    model.set_input(data)         # unpack data from dataset
    model.forward()   # calculate loss functions, get gradients, update network weights
    visual_dict = model.get_current_visuals()
    print('saving iteration {}'.format(i))
    visualizer.save_images(visual_dict, save_dir, img_name)

