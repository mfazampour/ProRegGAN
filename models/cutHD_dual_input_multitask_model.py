import argparse
from collections import OrderedDict
import util.util as util

import torch
from torch.utils.tensorboard import SummaryWriter

from .cutHD_dual_input_model import CUTHDDualInputModel
from .multitask_parent import Multitask


class CUTHDDualInputMultiTaskModel(CUTHDDualInputModel, Multitask):

    @staticmethod
    def modify_commandline_options(parser: argparse.ArgumentParser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        Parameters
        ----------
        parser
        """
        parser = CUTHDDualInputModel.modify_commandline_options(parser, is_train)
        parser = Multitask.modify_commandline_options(parser, is_train)

        parser.add_argument('--reg_coarse_image', action='store_true', help='if coarse image should be used for registration')

        return parser

    def __init__(self, opt: argparse.Namespace):
        """Initialize the class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        For visualization, we set 'visual_names' as 'real_A' (input real image),
        'real_B_rgb' (ground truth RGB image), and 'fake_B_rgb' (predicted RGB image)
        We convert the Lab image 'real_B' (inherited from Pix2pixModel) to a RGB image 'real_B_rgb'.
        we convert the Lab image 'fake_B' (inherited from Pix2pixModel) to a RGB image 'fake_B_rgb'.
        """
        super(CUTHDDualInputMultiTaskModel, self).__init__(opt)

        self.add_visdom_names(self.loss_names, self.visual_names)

        self.loss_names.append('G_adv')

        self.loss_functions = ['backward_G', 'compute_D_loss']

        self.add_networks(opt, self.model_names, self.loss_functions, self.gpu_ids)

        if self.isTrain:
            self.add_optimizers(self.optimizers)

    def set_input(self, input):
        self.clean_tensors()
        super(CUTHDDualInputMultiTaskModel, self).set_input(input)
        self.set_mt_input(input, real_B=self.real_B, shape=self.real_B.shape,
                          dtype=self.real_B.dtype, device=self.real_B.device)
        self.init_loss_tensors()
        self.loss_G_adv = torch.tensor([0.0])

    def forward(self):
        super().forward()
        if self.opt.reg_coarse_image:
            fixed = self.idt_B_dn.detach()
            moving = self.fake_B_dn
        else:
            moving = self.fake_B
            if self.opt.reg_idt_B:
                fixed = self.idt_B.detach()
            else:
                fixed = self.real_B
        self.mt_forward(moving, fixed, self.real_B, self.real_A)

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        self.loss_G_adv = self.compute_G_loss()
        self.loss_G = self.loss_G_adv
        self.loss_G += self.mt_g_backward()

        if torch.is_grad_enabled():
            self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A), rigid registration params, DVF and segmentation mask
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()  # update D's weights

        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        # self.set_requires_grad(self.netRigidReg, False)
        self.set_requires_grad(self.netDefReg, False)
        self.set_requires_grad(self.netSeg, False)
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        if self.opt.lambda_NCE > 0.0 and self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        self.backward_G()  # calculate gradients for G
        self.optimizer_G.step()  # update G's weights
        if self.opt.lambda_NCE > 0.0 and self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()

        # # update rigid registration network
        # if self.opt.use_rigid_branch:
        #     # self.set_requires_grad(self.netRigidReg, True)
        #     self.optimizer_RigidReg.zero_grad()
        #     self.backward_RigidReg()
        #     self.optimizer_RigidReg.step()

        # update deformable registration and segmentation network
        if (1 - self.first_phase_coeff) == 0:
            return
        # for _ in range(self.opt.vxm_iteration_steps):
        self.set_requires_grad(self.netDefReg, True)
        self.set_requires_grad(self.netSeg, True)
        self.optimizer_DefReg.zero_grad()
        self.optimizer_Seg.zero_grad()
        self.backward_DefReg_Seg()  # only back propagate through fake_B once
        self.optimizer_DefReg.step()
        self.optimizer_Seg.step()

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        super().compute_visuals()
        self.compute_mt_visuals(self.real_B, self.real_A.shape)

    def update_learning_rate(self, epoch=0):
        super().update_learning_rate(epoch=epoch)
        self.set_coeff_multitask_loss(epoch)

