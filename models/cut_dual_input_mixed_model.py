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
from .cut_dual_input_model import CUTDualInputModel
from .patchnce import PatchNCELoss

os.environ['VXM_BACKEND'] = 'pytorch'

#
class CUTDualInputMixedModel(CUTDualInputModel):
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
        parser.set_defaults(dual_mode=True, load_second_sample=True)

        return parser

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A : torch.Tensor = input['A' if AtoB else 'B'].to(self.device)
        self.real_B : torch.Tensor = input['B' if AtoB else 'A'].to(self.device)

        self.patient = input['Patient']
        self.mask_A = input['A_mask'].to(self.device).type(self.real_A.dtype)
        if input['B_mask_available'][0]:  # TODO fix it for batch size > 1
            self.mask_B = input['B_mask'].to(self.device).type(self.real_A.dtype)
        else:
            self.mask_B = None

        if self.opt.load_second_sample:
            self.real_B_non_corr = input['B2_B'].to(self.device)
            self.mask_B_non_corr = input['B2_B_mask'].to(self.device).type(self.real_A.dtype)

        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # Both real_B and real_A if we also use the loss from the identity mapping: NCE(G(Y), Y)) in NCE loss

        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt else self.real_A
        second_real = torch.cat((self.real_B_non_corr, self.real_B_non_corr), dim=0) if self.opt.nce_idt else self.real_B_non_corr

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


