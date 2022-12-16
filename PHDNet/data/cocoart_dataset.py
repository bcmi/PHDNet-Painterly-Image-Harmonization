import os.path
import torch
import random
import torchvision.transforms.functional as tf
from data.base_dataset import BaseDataset, get_transform
import PIL

from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from pathlib import Path

# ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image, ImageFile
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True
import time
import random

import cv2
random.seed(1)
#random.seed(2)


def mask_bboxregion(mask):
    w,h = np.shape(mask)[:2]
    valid_index = np.argwhere(mask==255) # [length,2]
    if np.shape(valid_index)[0] < 1:
        x_left = 0
        x_right = 0
        y_bottom = 0
        y_top = 0
    else:
        x_left = np.min(valid_index[:,0])
        x_right = np.max(valid_index[:,0])
        y_bottom = np.min(valid_index[:,1])
        y_top = np.max(valid_index[:,1])
    region = mask[x_left:x_right,y_bottom:y_top]
    return region
    # return [x_left, y_top, x_right, y_bottom]

def findContours(im):
    """
    Wraps cv2.findContours to maintain compatiblity between versions
    3 and 4
    Returns:
        contours, hierarchy
    """
    img = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)  # PIL转cv2
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # if cv2.__version__.startswith('4'):
    #     contours, hierarchy = cv2.findContours(*args, **kwargs)
    # elif cv2.__version__.startswith('3'):
    #     _, contours, hierarchy = cv2.findContours(*args, **kwargs)
    # else:
    #     raise AssertionError(
    #         'cv2 must be either version 3 or 4 to call this method')
    return contours, hierarchy


class COCOARTDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    def __init__(self, opt, is_for_train):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        ## content and mask
        self.path_img = []
        self.path_mask = []
        ## style
        self.path_style = []
        self.isTrain = is_for_train
        self.opt = opt
        self._load_images_paths()
        self.transform = get_transform(opt)
        


    def _load_images_paths(self,):
        if self.isTrain:
            path = self.opt.content_dir + '/train2014/'
        else:
            path = self.opt.content_dir + '/val2014/'
        
        self.paths_all = list(Path(path).glob('*'))

        if self.isTrain:
            print('loading training set')
            for path in self.paths_all:
                path_mask = str(path).replace('train2014','SegmentationClass_select',1).replace('jpg','png')
                if not os.path.exists(path_mask):
                    # print('train not exist',path_mask)
                    continue
                else:
                    self.path_img.append(str(path))
                    self.path_mask.append(path_mask)
            ## style
            for f in open(self.opt.style_dir + 'WikiArt_Split/style_train.csv'):
                self.path_style.append(self.opt.style_dir + f.split(',')[0])
        else:
            print('loading testing set')
            for path in self.paths_all:
                path_mask = str(path).replace('val2014','SegmentationClass_select',1).replace('jpg','png')
                if not os.path.exists(path_mask):
                    continue
                else:
                    self.path_img.append(str(path))
                    self.path_mask.append(path_mask)
            ## style
            for f in open(self.opt.style_dir + 'WikiArt_Split/style_val.csv'):
                self.path_style.append(self.opt.style_dir + f.strip().split(',')[0])

        print('foreground number',len(self.path_img))
        print('background number',len(self.path_style))
        
    def select_mask(self,index):
        """select one foreground to generate the composite image"""
        if self.isTrain:
            mask_all = Image.open(self.path_img[index].replace('train2014','SegmentationClass_select',1).replace('jpg','png')).convert('L')
        else:
            mask_all = Image.open(self.path_img[index].replace('val2014','SegmentationClass_select',1).replace('jpg','png')).convert('L')

        mask_array = np.array(mask_all)
        mask_value = np.unique(np.sort(mask_array[mask_array>0]))
        object_num = len(mask_value)
        whole_area = np.ones(np.shape(mask_array))

        if self.isTrain:
            random_pixel = random.choice(mask_value)
        else:
            random_pixel = mask_value[0]
        
        if random_pixel!=255:
            mask_array[mask_array==255] = 0
        mask_array[mask_array==random_pixel]=255
        mask_array[mask_array!=255]=0
        return mask_array

    def get_small_scale_mask(self, mask, number):
        """generate n*n patch to supervise discriminator"""
        mask = np.asarray(mask)
        mask = np.uint8(mask / 255.)
        mask_small = np.zeros([number, number],dtype=np.float32)
        split_size = self.opt.load_size // number
        for i in range(number):
            for j in range(number):
                mask_split = mask[i*split_size: (i+1)*split_size, j*split_size: (j+1)*split_size]
                mask_small[i, j] = (np.sum(mask_split) > 0) * 255
                #mask_small[i, j] = (np.sum(mask_split) / (split_size * split_size)) * 255
        mask_small = np.uint8(mask_small)
        return Image.fromarray(mask_small,mode='L')


    def __getitem__(self, index):
        style = Image.open(self.path_style[index]).convert('RGB')
        c_index = index % len(self.path_img)
        content = Image.open(self.path_img[c_index]).convert('RGB')
        select_mask = self.select_mask(c_index)
        np_mask = np.uint8(select_mask)
        mask = Image.fromarray(np_mask,mode='L')
        
        content = tf.resize(content, [self.opt.load_size, self.opt.load_size])
        style = tf.resize(style, [self.opt.load_size, self.opt.load_size])
        mask = tf.resize(mask, [self.opt.load_size, self.opt.load_size])
        mask_small = self.get_small_scale_mask(mask, self.opt.patch_number)

        #apply the same transform to composite and real images
        content = self.transform(content)
        style = self.transform(style)

        mask = tf.to_tensor(mask)
        mask = mask*2 -1
        mask_small = tf.to_tensor(mask_small)
        comp = self._compose(content, mask, style)

        return {'comp': comp, 'mask': mask, 'mask_small': mask_small, 'style': style,'content':content, 'img_path':self.path_style[index]}

    def __len__(self):
        return len(self.path_style)
        # return 100

    def _compose(self, foreground_img, foreground_mask, background_img):
        foreground_img = foreground_img/2 + 0.5
        background_img = background_img/2 + 0.5
        foreground_mask = foreground_mask/2 + 0.5
        comp = foreground_img * foreground_mask + background_img * (1 - foreground_mask)
        comp = comp*2-1
        return comp
