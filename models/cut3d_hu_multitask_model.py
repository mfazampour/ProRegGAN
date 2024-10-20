import argparse
from collections import OrderedDict
import util.util as util

import torch
from torch.utils.tensorboard import SummaryWriter

from .cut3d_model import CUT3dModel
from .multitask_parent import Multitask
from .networks import DiceLoss


class CUT3DHuMultiTaskModel(CUT3dModel, Multitask):

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
        parser = CUT3dModel.modify_commandline_options(parser, is_train)
        parser = Multitask.modify_commandline_options(parser, is_train)

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
        super(CUT3DHuMultiTaskModel, self).__init__(opt)

        self.add_visdom_names(self.loss_names, self.visual_names)
        # prune loss names
        self.loss_names = [loss for loss in self.loss_names if 'Rigid' not in loss and 'Seg' not in loss]
        self.visual_names = []

        self.loss_functions = []

        self.add_networks(opt, self.model_names, self.loss_functions, self.gpu_ids)

        self.criterionDefReg = DiceLoss(include_background=False, to_onehot_y=True)

        if self.isTrain:
            self.add_optimizers(self.optimizers)
            self.loss_functions = ['backward_G', 'compute_D_loss', 'backward_DefReg_Seg']

    def set_input(self, input):
        self.clean_tensors()
        super(CUT3DHuMultiTaskModel, self).set_input(input)
        self.set_mt_input(input, real_B=self.real_B, shape=self.real_B.shape,
                          dtype=self.real_B.dtype, device=self.real_B.device)
        self.init_loss_tensors()
        self.loss_G_adv = torch.tensor([0.0])

    def forward(self):
        super().forward()
        fixed = self.idt_B.detach() if self.opt.reg_idt_B else self.real_B
        self.mt_forward(self.fake_B, fixed, self.real_B, self.real_A)

    def get_current_landmark_distances(self):
        return torch.tensor([0]), torch.tensor([0]), torch.tensor([0])

    def mt_forward(self, moving: torch.Tensor, fixed: torch.Tensor, real_B: torch.Tensor, real_A: torch.Tensor):
        """
        Calls forward of the multitask part of the code, which is deformable registration, segmentation and
        if needed rigid registration
        Parameters
        ----------
        moving
        fixed
        real_B
        real_A

        Returns
        -------

        """
        # storing params
        self.moving = moving * 0.5 + 0.5
        self.fixed = fixed * 0.5 + 0.5

        # just for compatibility
        self.real_A = real_A
        self.real_B = real_B

        # deformable registration branch
        def_reg_output = self.netDefReg(self.moving.detach(), self.fixed, registration=not self.isTrain)

        if self.opt.bidir and self.isTrain:
            (self.deformed_moving, self.deformed_fixed, self.dvf) = def_reg_output
        else:
            (self.deformed_moving, self.dvf) = def_reg_output

        if self.isTrain:
            self.dvf = self.resizer(self.dvf)
        self.deformed_moving_real = self.transformer_intensity(real_A, self.dvf.detach())
        self.mask_moving_deformed = self.transformer_label(self.mask_A, self.dvf.detach())
        self.mask_moving_deformed_w_grad = self.transformer_intensity(self.mask_A, self.dvf)

        self.compute_gt_dice()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        self.loss_G_adv = self.compute_G_loss()
        self.loss_G = self.loss_G_adv
        # self.loss_G += self.mt_g_backward()

        if torch.is_grad_enabled():
            self.loss_G.backward()

    def backward_DefReg_Seg(self):
        if self.mask_B is not None:
            self.loss_DefReg = self.criterionDefReg(self.mask_moving_deformed_w_grad, self.mask_B)
        else:
            self.loss_DefReg = torch.tensor([0.0], device=self.moving.device)

        self.loss_DefRgl = self.criterionDefRgl(self.dvf, None)
        self.loss_DefReg += self.loss_DefRgl * self.opt.grad_loss_weight

        if torch.is_grad_enabled():
            self.loss_DefReg.backward()

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
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        self.backward_G()  # calculate gradients for G
        self.optimizer_G.step()  # update G's weights

        # update deformable registration and segmentation network
        if (1 - self.first_phase_coeff) == 0:
            return
        self.set_requires_grad(self.netDefReg, True)
        self.optimizer_DefReg.zero_grad()
        self.backward_DefReg_Seg()  # only back propagate through fake_B once
        self.optimizer_DefReg.step()

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        super().compute_visuals()

    def compute_landmark_loss(self):
        pass

    def update_learning_rate(self, epoch=0):
        super().update_learning_rate(epoch=epoch)
        self.set_coeff_multitask_loss(epoch)

    def log_mt_tensorboard(self, real_A, real_B, fake_B, writer: SummaryWriter, global_step: int = 0,
                           use_image_name=False, mode=''):
        self.add_deformable_figures(mode, real_A *0.5 + 0.5, self.mask_A, self.fixed, self.mask_B, self.deformed_moving_real,
                                    self.mask_moving_deformed, global_step, writer, use_image_name)
        # self.add_deformable_figures(mode=mode, global_step=global_step, writer=writer, use_image_name=use_image_name)
        self.add_losses(mode, global_step, writer)

