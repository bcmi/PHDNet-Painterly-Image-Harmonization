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



class unetModel(BaseModel):
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.opt = opt
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        #self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake', 'D_gp', 'D_global', 'D_local', 'G_global', 'G_local', 'c', 's']
        self.loss_names = ['G_L1', 'c', 's', 'tv']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        # self.visual_names = ['comp', 'content','style', 'output', 'mask_vis','mask_all_vis', 'real_f', 'fake_f', 'bg', 'attentioned']
        # self.visual_names = ['comp', 'content','style', 'output', 'mask_vis', 'mask_all_vis', 'real_f', 'fake_f']

        self.visual_names = ['comp', 'content','style', 'mask_vis', 'output', 'coarse', 'attention_map', 'real_f', 'fake_f']
        #self.visual_names = ['comp', 'output', 'attention_map']
        #self.visual_names = ['output']

        
        # self.visual_names = ['comp', 'content','style', 'mask_vis', 'output']



        # self.visual_names = ['comp', 'content','style','mask_vis','mask_all_vis']


        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            # self.model_names = ['G', 'D','Decoder']
            self.model_names = ['G']
        else:
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        # self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG,
        #                               not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        
        #self.netDecoder = networks.decoder
        self.netvgg = networks.vgg
        self.netvgg.load_state_dict(torch.load(opt.vgg))
        self.netvgg = nn.Sequential(*list(self.netvgg.children())[:31])
        '''
        if opt.is_update == 1:
            self.netEncoder = networks.encoder
            self.netEncoder = nn.Sequential(*list(self.netEncoder.children())[:4])
            self.netG = networks.FUNet_update_first(self.netEncoder, self.netDecoder,self.netvgg , is_matting=opt.is_matting)#cuda()
        elif opt.is_update == 2:
            self.netEncoder = networks.encoder
            self.netEncoder = nn.Sequential(*list(self.netEncoder.children())[:31])
            self.netG = networks.FUNet_update_all(self.netEncoder, self.netDecoder, self.netvgg, is_matting=opt.is_matting)#cuda()
        '''
        if opt.is_skip:
            self.netDecoder = networks.decoder_cat
        else:
            self.netDecoder = networks.decoder
        #self.netDecoder = networks.decoder
        #self.netDecoder = networks.decoder_unet
        #self.netG = networks.FUNet_fixed_fft(self.netvgg, self.netDecoder, is_matting=opt.is_matting)#cuda()
        self.netG = networks.FUNet_fixed(self.netvgg, self.netDecoder, is_matting=opt.is_matting, is_skip=opt.is_skip, 
                                         is_fft=opt.is_fft, fft_num=opt.fft_num, transfer=opt.transfer, style_loss=opt.style_loss, fft_mode=opt.fft_mode)#cuda()
        if len(self.gpu_ids) > 0:
            assert(torch.cuda.is_available())
            self.netG.to(self.gpu_ids[0])
            self.netG = torch.nn.DataParallel(self.netG, self.gpu_ids)

        # self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt, opt.normG,
        #                               not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, vgg, self.netDecoder)
                                
        
        # decoder = net.decoder
        # vgg = net.vgg
        # vgg.load_state_dict(torch.load(args.vgg))
        # vgg = nn.Sequential(*list(vgg.children())[:31])
        # network = net.Net(vgg, decoder)
        # network.train()
        #print(self.netG)
        self.cnt = 0
        

        
        self.relu = nn.ReLU()

        #if self.isTrain: 
        #    self.gan_mode = opt.gan_mode
        #    netD = networks.NLayerDiscriminator(opt.output_nc, opt.ndf, opt.n_layers_D, networks.get_norm_layer(opt.normD))
        #    self.netD = networks.init_net(netD, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(filter(lambda p: p.requires_grad, self.netG.parameters()), lr=opt.lr*opt.g_lr_ratio, betas=(opt.beta1, 0.999))
            #if opt.is_update == 0:
            #    self.optimizer_G = torch.optim.Adam(self.netDecoder.parameters(), lr=opt.lr*opt.g_lr_ratio, betas=(opt.beta1, 0.999))
            #else:
            #    self.optimizer_G = torch.optim.Adam(itertools.chain(self.netEncoder.parameters(), self.netDecoder.parameters()), lr=opt.lr*opt.g_lr_ratio, betas=(opt.beta1, 0.999))
            #self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr*opt.d_lr_ratio, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            #self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        self.content = input['content'].to(self.device)
        self.style = input['style'].to(self.device)
        self.comp = input['comp'].to(self.device)
        # self.real = input['real'].to(self.device)
        self.mask_vis = input['mask'].to(self.device)
        self.mask_blur = input['mask_blur'].to(self.device)
        self.mask = self.mask_vis/2 + 0.5
        self.mask_blur = self.mask_blur/2 + 0.5
        self.inputs = self.comp
        if self.opt.input_nc == 4:
            self.inputs = torch.cat([self.inputs, self.mask], 1)  # channel-wise concatenation
        self.real_f = self.content * self.mask
        self.real_f[self.real_f==0] = -1
        self.bg = self.style * (1 - self.mask)
        self.bg[self.bg==0] = -1
        self.mask_all_vis = input['mask_all'].to(self.device)
        #print(self.comp.shape)
        #print('content', self.content)
        #print('style', self.style)
        #print('comp', self.comp)
        #print('mask', self.mask)
        #exit(0)

    def forward(self):
        self.cnt += 1
        # self.output = self.netG(self.inputs, self.mask)
        self.output, self.coarse,  self.loss_c, self.loss_s, self.loss_tv, self.loss_map, self.attention_map = self.netG(self.comp, self.content, self.style, self.mask, self.mask_blur, self.cnt)
        # print(self.output.max(),self.output.min())
        self.fake_f = self.output * self.mask
        self.fake_f[self.fake_f==0] = -1
        self.attentioned = self.output * self.mask + self.inputs[:,:3,:,:] * (1 - self.mask)
        self.harmonized = self.attentioned

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake;
        fake_AB = self.harmonized
        pred_fake, ver_fake = self.netD(fake_AB.detach(), self.mask)
        if self.gan_mode == 'wgangp':
            global_fake = self.relu(1 + pred_fake).mean()
            local_fake = self.relu(1 + ver_fake).mean()
        else:
            global_fake = self.criterionGAN(pred_fake, False)
            local_fake = self.criterionGAN(ver_fake, False)
        self.loss_D_fake = global_fake + local_fake

        # Real
        real_AB = self.style
        pred_real, ver_real = self.netD(real_AB, self.mask)
        if self.gan_mode == 'wgangp':
            global_real = self.relu(1 - pred_real).mean()
            local_real = self.relu(1 - ver_real).mean()
        else:
            global_real = self.criterionGAN(pred_real, True)
            local_real = self.criterionGAN(ver_real, True)
        self.loss_D_real = global_real + local_real

        self.loss_D_global = global_fake + global_real
        self.loss_D_local = local_fake + local_real
        
        gradient_penalty, gradients = networks.cal_gradient_penalty(self.netD, real_AB.detach(), fake_AB.detach(),
                                                                    'cuda', mask=self.mask)
        self.loss_D_gp = gradient_penalty

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real + self.opt.gp_ratio * gradient_penalty)
        self.loss_D.backward(retain_graph=True)

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        #fake_AB = self.harmonized
        #pred_fake, ver_fake, featg_fake, featl_fake = self.netD(fake_AB, self.mask, feat_loss=True)
        #self.loss_G_global = self.criterionGAN(pred_fake, True)
        #self.loss_G_local = self.criterionGAN(ver_fake, True)
        
        #self.loss_G_GAN =self.opt.lambda_a * self.loss_G_global + self.opt.lambda_v * self.loss_G_local

        self.loss_G_L1 = torch.zeros(1).cuda()

        # self.loss_G_L1 = self.criterionL1(self.attentioned, self.real) * self.opt.lambda_L1
        # self.loss_G = self.loss_G_GAN + self.loss_G_L1
        # content/style loss
        self.loss_G = self.opt.content_weight * self.loss_c + self.opt.style_weight * self.loss_s + self.loss_tv*self.opt.lamda_tv + self.opt.lamda_mask * self.loss_map # + self.loss_G_GAN
        print('loss: c{}_s{}_tv{}'.format(self.loss_c,self.loss_s.item(),self.loss_tv))
        self.loss_G.backward(retain_graph=True)

    def optimize_parameters(self):
        self.forward()

        # if self.opt.lambda_a>0:
        # update D
        #self.set_requires_grad(self.netD, True)  # enable backprop for D
        #self.optimizer_D.zero_grad()  # set D's gradients to zero
        #self.backward_D()  # calculate gradients for D
        #self.optimizer_D.step()  # update D's weights
        # update G
        #self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights
    

    def get_current_visuals(self):
        t= time.time()
        nim = self.style.size(0)
        #print(nim)
        visual_ret = OrderedDict()
        all =[]
        if nim>1:
            num = min(nim-1,5)
        else:
            num=1

        for i in range(0,num):
        # for i in range(0,1):
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
    

    

