import torch
from .base_model import BaseModel
from collections import OrderedDict
from . import networks
import torch.nn.functional as F
from torch import nn, cuda
from torch.autograd import Variable
import time
import numpy as np
from util import util
import os
import itertools



class PHDNetModel(BaseModel):
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.opt = opt
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G', 'c', 's', 'tv', 'mask', 'G_GAN', 'D', 'D_fake', 'D_real']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['comp', 'content', 'style', 'mask_vis', 'output', 'coarse', 'attention_map']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netvgg = networks.vgg
        self.netvgg.load_state_dict(torch.load(opt.vgg))
        self.netvgg = nn.Sequential(*list(self.netvgg.children())[:31])

        if opt.is_skip:
            self.netDecoder = networks.decoder_cat
        else:
            self.netDecoder = networks.decoder
        if opt.netG == 'phd':
            self.netG = networks.PHDNet(self.netvgg, self.netDecoder, is_matting=opt.is_matting, is_skip=opt.is_skip, is_fft=opt.is_fft, fft_num=opt.fft_num)
        else:
            raise NotImplementedError(f'{opt.netD} not implemented')
        if len(self.gpu_ids) > 0:
            assert(torch.cuda.is_available())
            self.netG.to(self.gpu_ids[0])
            self.netG = torch.nn.DataParallel(self.netG, self.gpu_ids)
        if self.isTrain:
            # define loss functions
            self.criterionGAN = nn.MSELoss().to(self.device)
            if opt.netD == 'conv':
                netD = networks.ConvDiscriminator(depth=8, patch_number=opt.patch_number, batchnorm_from=0, use_fft=True)
            else:
                raise NotImplementedError(f'{opt.netD} not implemented')
            self.netD = networks.init_net(netD, opt.init_type, opt.init_gain, self.gpu_ids)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(filter(lambda p: p.requires_grad, self.netG.parameters()), lr=opt.lr*opt.g_lr_ratio, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr*opt.d_lr_ratio, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        self.content = input['content'].to(self.device)
        self.style = input['style'].to(self.device)
        self.comp = input['comp'].to(self.device)
        self.mask_vis = input['mask'].to(self.device)
        self.mask = self.mask_vis/2+0.5
        if self.isTrain:
            self.mask_small = input['mask_small'].to(self.device)

    def forward(self):
        """Employ generator to generate the output, and calculate the losses for generator"""
        self.output, self.coarse, self.loss_c, self.loss_s, self.loss_tv, self.loss_mask, self.attention_map = self.netG(self.comp, self.content, self.style, self.mask)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake;
        fake_AB = self.output
        self.pred_fake = self.netD(fake_AB.detach())
        self.pred_comp = self.netD(self.comp)
        output_fake = self.criterionGAN(self.pred_fake, self.mask_small)
        composite_fake = self.criterionGAN(self.pred_comp, self.mask_small)
        self.loss_D_fake = output_fake + composite_fake

        # Real
        real_AB = self.style
        self.pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(self.pred_real, torch.zeros(self.mask_small.size()).cuda())

        # combine loss and calculate gradients
        self.loss_D = self.loss_D_fake + self.loss_D_real
        self.loss_D.backward(retain_graph=True)

    def backward_G(self):
        """Calculate GAN and other losses for the generator"""
        # GAN loss
        fake_AB = self.output
        self.pred_fake_G = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(self.pred_fake_G, torch.zeros(self.mask_small.size()).cuda())

        # content/style loss, smooth loss, mask blending loss
        self.loss_G = self.opt.lambda_content * self.loss_c + self.opt.lambda_style * self.loss_s + self.loss_tv*self.opt.lambda_tv + self.opt.lambda_mask * self.loss_mask + self.opt.lambda_g * self.loss_G_GAN
        print('loss: c {}_s {}_tv {}_m {}_g {}'.format(self.loss_c,self.loss_s.item(),self.loss_tv,self.loss_mask,self.loss_G_GAN))
        self.loss_G.backward(retain_graph=True)

    def optimize_parameters(self):
        """optimize both G and D, only run this in training phase"""
        self.forward()
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        self.backward_D()  # calculate gradients for D
        self.optimizer_D.step()  # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights


    def get_current_visuals(self):
        num = self.style.size(0)
        visual_ret = OrderedDict()
        all =[]
        for i in range(0,num):
            row=[]
            for name in self.visual_names:
                if isinstance(name, str):
                    if hasattr(self,name):
                        im = util.tensor2im(getattr(self, name).data[i:i+1,:,:,:])
                        row.append(im)
            row=tuple(row)
            all.append(np.hstack(row))
        all = tuple(all)

        allim = np.vstack(all)
        return OrderedDict([(self.opt.name,allim)])
    

    

