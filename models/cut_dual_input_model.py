import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from monai.visualize import img2tensorboard
from torch.utils.tensorboard import SummaryWriter

import util.util as util
from util import tensorboard
from util.image_pool import ImagePool
from . import networks
from . import networks3d
from .base_model import BaseModel
from .cut3d_model import CUT3dModel
from .patchnce import PatchNCELoss

os.environ['VXM_BACKEND'] = 'pytorch'

#
class CUTDualInputModel(CUT3dModel):
    """ This class implements Dual input cut model
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """

        parser = CUT3dModel.modify_commandline_options(parser, is_train)

        # add dual_mode = True to parser defaults
        parser.set_defaults(dual_mode=True)

        return parser

    def __init__(self, opt):
        """Initialize the CUTDualInputModel class.
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        self.visual_names = ['real_A_center_sag', 'real_A_center_cor', 'real_A_center_axi']
        self.visual_names += ['fake_B_center_sag', 'fake_B_center_cor', 'fake_B_center_axi']
        self.visual_names += ['real_B_center_sag', 'real_B_center_cor', 'real_B_center_axi']

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B_center_sag', 'idt_B_center_cor', 'idt_B_center_axi']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = networks3d.define_G(opt.input_nc, opt.output_nc, opt.ngf, "resnet_cat", opt.normG, not opt.no_dropout,
                                        opt.init_type, opt.init_gain, no_antialias=opt.no_antialias,
                                        no_antialias_up=opt.no_antialias_up, gpu_ids=self.gpu_ids, opt=opt)
        self.netF = networks3d.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type,
                                        opt.init_gain, opt.no_antialias, gpu_ids=self.gpu_ids, opt=opt)

        if self.isTrain:
            self.netD = networks3d.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type,
                                            opt.init_gain, no_antialias=opt.no_antialias, gpu_ids=self.gpu_ids, opt=opt)

            # define loss functions
            self.criterionGAN_syn = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.fake_pool = ImagePool(opt.pool_size)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # Both real_B and real_A if we also use the loss from the identity mapping: NCE(G(Y), Y)) in NCE loss

        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt else self.real_A
        if self.opt.load_second_sample:
            second_real = torch.cat((self.real_B_correspinding, self.real_B_correspinding), dim=0) if self.opt.nce_idt else self.real_B_correspinding
        else:
            second_real = torch.cat((self.real_B, self.real_B), dim=0) if self.opt.nce_idt else self.real_B

        # Inspired by GcGAN, FastCUT is trained with flip-equivariance augmentation, where
        # the input image to the generator is horizontally flipped, and the output features
        # are flipped back before computing the PatchNCE loss
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        self.fake = self.netG(self.real, second_image=second_real)

        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]


