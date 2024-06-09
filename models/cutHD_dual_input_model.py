from collections import OrderedDict
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from monai.visualize import img2tensorboard
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from . import networks3d
from .patchnce import PatchNCELoss
import util.util as util
from util import tensorboard
from .cutHD3d_model import CUTHD3dModel

os.environ['VXM_BACKEND'] = 'pytorch'
from voxelmorph import voxelmorph as vxm


#
class CUTHDDualInputModel(CUTHD3dModel):
    """ This class implements CUT and FastCUT model, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020

    The code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser = CUTHD3dModel.modify_commandline_options(parser, is_train)

        # add dual_mode = True to parser defaults
        parser.set_defaults(dual_mode=True, netG='noise_aware')
        parser.add_argument('--dual_global', action='store_true', help='use dual input for global generator')

        return parser


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B_unfiltered'].to(self.device)
        self.real_B = input['B_unfiltered' if AtoB else 'A'].to(self.device)
        self.zoomed_B = input['B_zoomed'].to(self.device)
        self.real_B_dn = input['B_denoised'].to(self.device)
        self.mask_A = input['A_mask'].to(self.device).type(self.real_A.dtype)

        if input['B_mask_available'][0]:
            self.mask_B = input['B_mask'].to(self.device).type(self.real_A.dtype)
        else:
            self.mask_B = None

        self.patient = input['Patient']
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

        if self.opt.load_second_sample:
            self.real_B_non_corr = input['B2_B'].to(self.device)
            self.real_B_dn_non_corr = input['B2_B_denoised'].to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # Both real_B and real_A if we also use the loss from the identity mapping: NCE(G(Y), Y)) in NCE loss

        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt else self.real_A
        if self.opt.load_second_sample:
            second_real = torch.cat((self.real_B_non_corr, self.real_B_non_corr), dim=0) if self.opt.nce_idt else self.real_B_non_corr
        else:
            second_real = torch.cat((self.zoomed_B, self.zoomed_B), dim=0) if self.opt.nce_idt else self.zoomed_B

        self.fake, self.fake_dn = self.netG(self.real, second_image=second_real)
        if self.opt.coarse_only:
            self.fake = self.fake.detach()

        self.fake_B = self.fake[:self.real_A.size(0)]
        self.fake_B_dn = self.fake_dn[:self.real_A.size(0)]

        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]
            self.idt_B_dn = self.fake_dn[self.real_A.size(0):]


    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        # First, G(A) should fake the discriminator
        pred_fake = self.netD(torch.cat((self.real_A, self.fake_B), dim=1))
        # The adversarial loss for the algorithm to also change the domain
        self.loss_G_GAN = self.criterionGAN_syn(pred_fake, True).mean() * self.opt.lambda_GAN

        pred_fake_dn = self.netD_DN(torch.cat((self.real_A, self.fake_B_dn), dim=1))
        # The adversarial loss for the algorithm to also change the domain
        self.loss_G_GAN_dn = self.criterionGAN_syn(pred_fake_dn, True).mean() * self.opt.lambda_GAN
        self.loss_G_GAN += self.loss_G_GAN_dn

        pred_fake_n = self.discriminate(self.real_A, (self.fake_B - self.fake_B_dn.detach()), netD=self.netD_N)
        self.loss_G_GAN_n = self.criterionGAN_syn(pred_fake_n, True).mean()
        self.loss_G_GAN += self.loss_G_GAN_n

        # Lambda = 1 by default
        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B_dn, input_is_denoised=True,
                                                    mask=self.mask_A if self.opt.mask_contrastive_loss else None)
            if not self.opt.dual_global:
                self.loss_NCE += self.calculate_NCE_loss(self.real_A, self.fake_B, input_is_denoised=False,
                                                         mask=self.mask_A if self.opt.mask_contrastive_loss else None)
        else:
            self.loss_NCE= 0.0

        # For contrastive loss between Y and G(Y) but nce_idt is by default 0.0
        # Lambda = 1 by default
        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B_dn, self.idt_B_dn, input_is_denoised=True,
                                                       mask=self.mask_B if self.opt.mask_contrastive_loss else None)
            if not self.opt.dual_global:
                self.loss_NCE_Y += self.calculate_NCE_loss(self.real_B, self.idt_B, input_is_denoised=False,
                                                          mask=self.mask_B if self.opt.mask_contrastive_loss else None)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            self.loss_NCE_Y = 0.0
            loss_NCE_both = self.loss_NCE

        pred_real = self.discriminate(self.real_A, self.real_B, netD=self.netD)
        pred_real_dn = self.discriminate(self.real_A, self.real_B_dn, netD=self.netD_DN)

        # feature matching loss
        # GAN feature matching loss

        self.loss_G_GAN_Feat, self.loss_G_GAN_Feat_dn = self.feature_matching_loss(pred_fake, pred_fake_dn, pred_real,
                                                                         pred_real_dn)

        self.loss_G = self.loss_G_GAN + loss_NCE_both

        if not self.opt.no_ganFeat_loss:
            self.loss_G += self.loss_G_GAN_Feat + self.loss_G_GAN_Feat_dn
        return self.loss_G