import argparse
import datetime
import os
import sys
from collections import OrderedDict
from typing import Tuple

import torch
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

import util.util as util
from util import affine_transform
from util import distance_landmarks
from . import networks
from . import networks3d
from util import tensorboard

os.environ['VXM_BACKEND'] = 'pytorch'
from voxelmorph import voxelmorph as vxm

from monai.metrics import compute_meandice
from monai.metrics import compute_hausdorff_distance
from monai.metrics import compute_average_surface_distance

class Multitask:

    @staticmethod
    def modify_commandline_options(parser: argparse.ArgumentParser, is_train=True):
        # rigid and segmentation
        # parser.add_argument('--netReg', type=str, default='NormalNet', help='Type of network used for registration')
        parser.add_argument('--netSeg', type=str, default='unet_small', help='Type of network used for segmentation')
        parser.add_argument('--num-classes', type=int, default=2, help='num of classes for segmentation')

        # voxelmorph params
        parser.add_argument('--cudnn-nondet', action='store_true', help='disable cudnn determinism - might slow down training')
        # network architecture parameters
        parser.add_argument('--enc', type=int, nargs='+', help='list of unet encoder filters (default: 16 32 32 32)')
        parser.add_argument('--dec', type=int, nargs='+', help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
        parser.add_argument('--int-steps', type=int, default=7, help='number of integration steps (default: 7)')
        parser.add_argument('--int-downsize', type=int, default=2, help='flow downsample factor for integration (default: 2)')
        parser.add_argument('--bidir', action='store_true', help='enable bidirectional cost function')
        parser.add_argument('--use-probs', action='store_true', help='enable probabilities')
        parser.add_argument('--kl-lambda', type=float, default=10, help='prior lambda regularization for KL loss (default: 10)')
        parser.add_argument('--flow-logsigma-bias', type=float, default=-10, help='negative value for initialization of the logsigma layer bias value')

        # others
        parser.add_argument('--reg_idt_B', action='store_true', help='use idt_B from CUT model instead of real B')
        parser.add_argument('--show_volumes', type=bool, default=False, help='visualize transformed volumes w napari')
        parser.add_argument('--augment_segmentation', action='store_true', help='Augment data before segmenting')

        if is_train:
            parser.add_argument('--lambda_Reg', type=float, default=0.5, help='weight for the registration loss')
            parser.add_argument('--lr_Reg', type=float, default=0.0001, help='learning rate for the reg. network opt.')
            parser.add_argument('--lambda_Seg', type=float, default=0.5, help='weight for the segmentation loss')
            parser.add_argument('--lr_Seg', type=float, default=0.0001, help='learning rate for the reg. network opt.')
            parser.add_argument('--lambda_Def', type=float, default=10.0, help='weight for the def. reg. network loss')
            parser.add_argument('--lr_Def', type=float, default=0.0001, help='learning rate for the reg. network opt.')
            parser.add_argument('--vxm_iteration_steps', type=int, default=5, help='number of steps to train the registration network for each simulated US')
            parser.add_argument('--similarity', type=str, default='NCC', choices=['NCC', 'MIND'], help='type of the similarity used for training voxelmorph')
            parser.add_argument('--epochs_before_reg', type=int, default=0, help='number of epochs to train the network before reg loss is used')
            parser.add_argument('--image-loss', default='mse', help='image reconstruction loss - can be mse or ncc (default: mse)')
            parser.add_argument('--grad_loss_weight', type=float, default=1.0, help='weight of gradient loss (lambda) (default: 1.0)')
        return parser

    def __init__(self, opt):
        self.isTrain = opt.isTrain
        self.opt = opt

    @staticmethod
    def get_model_names():
        return ['DefReg', 'Seg']

    def add_networks(self, opt, model_names, loss_functions, gpu_ids):
        # specify the models you want to save to the disk. The training/test scripts will call
        # <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            model_names += self.get_model_names()
        else:  # during test time, only load G
            model_names += self.get_model_names()

        loss_functions += ['backward_DefReg_Seg', 'compute_landmark_loss']

        self.criterionDefReg = getattr(sys.modules['models.networks'], self.opt.similarity + 'Loss')()
        self.criterionDefRgl = networks.GradLoss('l2', loss_mult=self.opt.int_downsize)
        self.first_phase_coeff = 0.0

        self.criterionSeg = networks.DiceLoss()
        self.criterionSeg_wo_bg = networks.DiceLoss(ignore_index=0)
        self.transformer = vxm.layers.SpatialTransformer((1, 1, 80, 80, 80))

        # extract shape from sampled input
        inshape = opt.inshape
        # device handling
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in opt.gpu_ids])
        # enabling cudnn determinism appears to speed up training by a lot
        torch.backends.cudnn.deterministic = not opt.cudnn_nondet
        # unet architecture
        enc_nf = opt.enc if opt.enc else [16, 32, 32, 32]
        dec_nf = opt.dec if opt.dec else [32, 32, 32, 32, 32, 16, 16]
        # configure new model
        self.netDefReg = vxm.networks.VxmDense(
            inshape=inshape,
            nb_unet_features=[enc_nf, dec_nf],
            bidir=opt.bidir,
            int_steps=opt.int_steps,
            int_downsize=opt.int_downsize,
            use_probs=opt.use_probs,
            flow_logsigma_bias=opt.flow_logsigma_bias
        )
        self.netDefReg = networks3d.init_net(self.netDefReg, gpu_ids=gpu_ids)

        self.netSeg = networks3d.define_G(opt.input_nc, opt.num_classes, opt.ngf,
                                          opt.netSeg, opt.norm, use_dropout=not opt.no_dropout,
                                          gpu_ids=gpu_ids, is_seg_net=True)
        self.transformer_label = networks3d.init_net(vxm.layers.SpatialTransformer(size=opt.inshape, mode='nearest'),
                                                     gpu_ids=gpu_ids)
        self.transformer_intensity = networks3d.init_net(vxm.layers.SpatialTransformer(size=opt.inshape),
                                                         gpu_ids=gpu_ids)
        self.resizer = networks3d.init_net(vxm.layers.ResizeTransform(vel_resize=0.5, ndims=3), gpu_ids=gpu_ids)

    def add_visdom_names(self, loss_names, visual_names):
        # specify the training losses you want to print out. The training/test scripts will call
        # <BaseModel.get_current_losses>
        loss_names += ['DefReg_real_moving', 'DefReg_real', 'DefReg_fake', 'Seg_real', 'Seg_fake', 'DefRgl']
        loss_names += ['Seg_real_inverse']
        loss_names += ['diff_dice', 'moving_dice', 'warped_dice']
        loss_names += ['warped_HD', 'moving_HD', 'diff_HD']
        loss_names += ['warped_ASD', 'moving_ASD', 'diff_ASD']
        # specify the images you want to save/display. The training/test scripts will call
        # <BaseModel.get_current_visuals>

        # segmentation
        visual_names += ['mask_A_center_sag', 'mask_A_center_cor', 'mask_A_center_axi']
        visual_names += ['seg_A_center_sag', 'seg_A_center_cor', 'seg_A_center_axi']
        visual_names += ['seg_B_center_sag', 'seg_B_center_cor', 'seg_B_center_axi']

        # deformable registration
        visual_names += ['dvf_center_sag', 'dvf_center_cor', 'dvf_center_axi']
        visual_names += ['deformed_B_center_sag', 'deformed_B_center_cor', 'deformed_B_center_axi']

    def add_optimizers(self, optimizers):

        # self.criterionDefReg = getattr(sys.modules['models.networks'], self.opt.similarity + 'Loss')()
        # self.criterionDefRgl = networks.GradLoss('l2', loss_mult=self.opt.int_downsize)
        self.optimizer_DefReg = torch.optim.Adam(self.netDefReg.parameters(), lr=self.opt.lr_Def)
        optimizers.append(self.optimizer_DefReg)

        self.optimizer_Seg = torch.optim.Adam(self.netSeg.parameters(), lr=self.opt.lr_Seg,
                                              betas=(self.opt.beta1, 0.999))
        optimizers.append(self.optimizer_Seg)
        # resize = vxm.layers.ResizeTransform(0.5, 1)

        self.first_phase_coeff = 1 if self.opt.epochs_before_reg > 0 else 0

    def clean_tensors(self):
        pass
        # all_members = self.__dict__.keys()
        # # print(f'{all_members}')
        # # GPUtil.showUtilization()
        # for item in all_members:
        #     if isinstance(self.__dict__[item], torch.Tensor):
        #         self.__dict__[item] = None
        # torch.cuda.empty_cache()
        # # GPUtil.showUtilization()

    def set_mt_input(self, input, real_B, shape, device, dtype=torch.float32):
        self.patient = input['Patient']
        self.landmarks_A = input['A_landmark'].to(device)
        self.landmarks_B = input['B_landmark'].to(device)

        self.deformed_fixed = None #affine_transform.transform_image(real_B, affine, device)
        # self.deformed_LM_B = affine_transform.transform_image(self.landmarks_B, affine, self.landmarks_B.device)

        # if self.opt.show_volumes:
        #     affine_transform.show_volumes([real_B, self.deformed_fixed])

        self.mask_A = input['A_mask'].to(device).type(dtype)
        if input['B_mask_available'][0]:  # TODO in this way it only works with batch size 1
            self.mask_B = input['B_mask'].to(device).type(dtype)
        else:
            self.mask_B = None

    def init_loss_tensors(self):
        self.loss_DefReg = torch.tensor([0.0])
        self.loss_DefReg_fake = torch.tensor([0.0])
        self.loss_DefRgl = torch.tensor([0.0])

        self.loss_Seg_real = torch.tensor([0.0])
        self.loss_Seg_fake = torch.tensor([0.0])
        self.loss_Seg = torch.tensor([0.0])

        self.loss_DefReg_real_moving = torch.tensor([0.0])
        self.loss_landmarks_beginning = torch.tensor([0.0])

        self.loss_warped_dice, self.loss_moving_dice, self.loss_diff_dice= None, None, None
        self.loss_warped_HD, self.loss_moving_HD, self.loss_diff_HD = None, None, None
        self.loss_warped_ASD, self.loss_moving_ASD, self.loss_diff_ASD = None, None, None

        self.loss_landmarks_def, self.loss_landmarks_def_diff = None, None

    def mt_forward(self, moving: torch.Tensor, fixed: torch.Tensor, real_B: torch.Tensor, real_A: torch.Tensor):
        """
        Calls forward of the multitask part of the code, which is deformable registration, segmentation
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
        self.landmark_A_dvf = self.transformer_label(self.landmarks_A, self.dvf.detach())

        # segmentation branch
        if self.opt.augment_segmentation:
            self.augmented_mask, affine = affine_transform.apply_random_affine(
                torch.cat([self.mask_A, self.mask_moving_deformed_w_grad], dim=0), rotation=0.5,
                translation=0.1, batchsize=2 * self.opt.batch_size, interp_mode='nearest')
            self.augmented_fake, _ = affine_transform.apply_random_affine(self.moving, affine=affine[self.opt.batch_size:, ...])
            self.augmented_real, _ = affine_transform.apply_random_affine(self.fixed, affine=affine[:self.opt.batch_size, ...])
            if self.mask_B is not None:
                self.augmented_mask_B, _ = affine_transform.apply_random_affine(self.mask_B, affine=affine[:self.opt.batch_size, ...], interp_mode='nearest')
            self.seg_B = self.netSeg(self.augmented_real)
            self.seg_fake_B = self.netSeg(self.augmented_fake)
        else:
            self.seg_B = self.netSeg(self.fixed)
            self.seg_fake_B = self.netSeg(self.moving)

        self.compute_gt_dice()

    def compute_gt_dice(self):
        """
        calculate the dice score between the deformed mask and the ground truth if we have it
        Returns
        -------

        """
        if self.mask_B is not None:
            shape = list(self.mask_moving_deformed.shape)
            n = self.opt.num_classes  # number of classes
            shape[1] = n
            one_hot_fixed = torch.zeros(shape, device=self.mask_B.device)
            one_hot_deformed = torch.zeros(shape, device=self.mask_B.device)
            one_hot_moving = torch.zeros(shape, device=self.mask_B.device)
            for i in range(n):
                one_hot_fixed[:, i, self.mask_B[0, 0, ...] == i] = 1
                one_hot_deformed[:, i, self.mask_moving_deformed[0, 0, ...] == i] = 1
                one_hot_moving[:, i, self.mask_A[0, 0, ...] == i] = 1

            self.loss_warped_dice = compute_meandice(one_hot_deformed, one_hot_fixed, include_background=False).mean()
            self.loss_moving_dice = compute_meandice(one_hot_moving, one_hot_fixed, include_background=False).mean()
            self.loss_diff_dice = 100 * (self.loss_warped_dice - self.loss_moving_dice)/self.loss_moving_dice

            self.loss_warped_HD = compute_hausdorff_distance(one_hot_deformed, one_hot_fixed, percentile=95).mean()
            self.loss_moving_HD = compute_hausdorff_distance(one_hot_moving, one_hot_fixed, percentile=95).mean()
            self.loss_diff_HD = 100 * (self.loss_warped_HD - self.loss_moving_HD)/self.loss_moving_HD

            self.loss_warped_ASD = compute_average_surface_distance(one_hot_deformed, one_hot_fixed).mean()
            self.loss_moving_ASD = compute_average_surface_distance(one_hot_moving, one_hot_fixed).mean()
            self.loss_diff_ASD = 100 * (self.loss_warped_ASD - self.loss_moving_ASD) / self.loss_moving_ASD

    def compute_landmark_loss(self):
        # Calc landmark difference from original landmarks and deformed

        self.loss_landmarks_beginning = distance_landmarks.get_distance_lmark(self.landmarks_A, self.landmarks_B,
                                                                              self.landmarks_B.device)

        self.loss_landmarks_def = distance_landmarks.get_distance_lmark(self.landmark_A_dvf, self.landmarks_B,
                                                                        self.landmarks_B.device)

        self.loss_landmarks_def_diff = self.loss_landmarks_beginning - self.loss_landmarks_def

    def mt_g_backward(self):

        # # Def registration:
        # if self.opt.bidir:
        #     loss_DefReg_fake = self.criterionDefReg(self.deformed_fixed.detach(), fake_B)
        # else:
        #     loss_DefReg_fake = 0

        # segmentation
        loss_G_Seg = self.criterionSeg(self.seg_fake_B, self.mask_A) * self.opt.lambda_Seg

        # combine loss and calculate gradients
        loss = loss_G_Seg * (1 - self.first_phase_coeff)  # + loss_DefReg_fake * (1 - self.first_phase_coeff)

        return loss


    def backward_DefReg_Seg(self):
        self.loss_DefReg_real_moving = self.criterionDefReg(self.moving, self.fixed)
        self.loss_DefReg_real = self.criterionDefReg(self.deformed_moving, self.fixed)  # TODO add weights same as vxm!
        self.loss_DefReg = self.criterionDefReg(self.deformed_moving, self.fixed)
        if self.opt.bidir:
            self.loss_DefReg_fake = self.criterionDefReg(self.deformed_fixed, self.moving.detach())
            self.loss_DefReg += self.loss_DefReg_fake
        else:
            self.loss_DefReg_fake = torch.tensor([0.0])
        self.loss_DefRgl = self.criterionDefRgl(self.dvf, None)
        self.loss_DefReg += self.loss_DefRgl * self.opt.grad_loss_weight
        self.loss_DefReg *= (1 - self.first_phase_coeff)

        if self.opt.augment_segmentation:
            seg_B = self.netSeg(self.augmented_real)
            # load the warped mask from the augmented batch by taking the second half of the batch
            mask = self.augmented_mask[self.opt.batch_size:, ...]
        else:
            seg_B = self.netSeg(self.fixed)
            mask = self.mask_moving_deformed_w_grad
        self.loss_Seg_real = self.criterionSeg(seg_B, torch.round(mask).detach())
        # self.loss_DefReg += self.criterionSeg(torch.stack([1-mask, mask], dim=1).squeeze(dim=2),
        #                                       torch.round(seg_B).detach()) * 1
        ####
        # calculate the segmentation loss taking the output of the segmentation network as target and the warped mask as input
        # the goal is to use this loss to update the registration network based on the segmentation output
        # do this part only if the segmentation output is of enough quality

        seg_B_argmax = torch.argmax(self.seg_B, dim=1)
        seg_B_argmax = seg_B_argmax.unsqueeze(1)
        seg_B_argmax = seg_B_argmax.float()
        # check if it has enough quality by comparing the volume of the segmentation output with the volume of the mask
        if (seg_B.sum() > 0.7 * mask.sum()) and (seg_B_argmax - torch.round(mask)).abs().sum() < 0.5 * torch.round(mask).sum():
            # add a channel for the background
            mask = torch.cat([1 - mask, mask], dim=1)
            self.loss_Seg_real_inverse = self.criterionSeg_wo_bg(mask, seg_B_argmax)
            self.loss_Seg_real += self.loss_Seg_real_inverse * 10
        else:
            self.loss_Seg_real_inverse = torch.tensor([0.0])
        ####

        if self.opt.augment_segmentation:
            seg_fake_B = self.netSeg(self.augmented_fake.detach())
            mask = self.augmented_mask[:self.opt.batch_size, ...].detach()
        else:
            seg_fake_B = self.netSeg(self.moving.detach())
            mask = self.mask_A
        self.loss_Seg_fake = self.criterionSeg(seg_fake_B, torch.round(mask).detach())

        self.loss_Seg = (self.loss_Seg_real + self.loss_Seg_fake) * (1 - self.first_phase_coeff)
        # self.loss_DefReg.backward()
        if torch.is_grad_enabled():
            (self.loss_DefReg * self.opt.lambda_Def + self.loss_Seg).backward()

    def compute_mt_visuals(self, real_B, shape):

        seg_fake_B_img = torch.argmax(self.seg_fake_B, dim=1, keepdim=True)
        seg_B_img = torch.argmax(self.seg_B, dim=1, keepdim=True)

        n_c = shape[2]
        # average over channel to get the real and fake image
        self.mask_A_center_sag = self.mask_A[:, :, int(n_c / 2), ...]
        self.seg_A_center_sag = seg_fake_B_img[:, :, int(n_c / 2), ...]
        self.seg_B_center_sag = seg_B_img[:, :, int(n_c / 2), ...]

        n_c = shape[3]
        self.mask_A_center_cor = self.mask_A[:, :, :, int(n_c / 2), ...]
        self.seg_A_center_cor = seg_fake_B_img[:, :, :, int(n_c / 2), ...]
        self.seg_B_center_cor = seg_B_img[:, :, :, int(n_c / 2), ...]

        n_c = shape[4]
        self.mask_A_center_axi = self.mask_A[..., int(n_c / 2)]
        self.seg_A_center_axi = seg_fake_B_img[..., int(n_c / 2)]
        self.seg_B_center_axi = seg_B_img[..., int(n_c / 2)]

        n_c = int(shape[2] / 2)
        self.dvf_center_sag = self.dvf[:, :, n_c, ...]
        self.deformed_B_center_sag = self.deformed_moving_real[:, :, n_c, ...]

        n_c = int(shape[3] / 2)
        self.dvf_center_cor = self.dvf[..., n_c, :]
        self.deformed_B_center_cor = self.deformed_moving_real[..., n_c, :]

        n_c = int(shape[4] / 2)
        self.dvf_center_axi = self.dvf[:, :, ..., n_c]
        self.deformed_B_center_axi = self.deformed_moving_real[..., n_c]


    def log_mt_tensorboard(self, real_A, real_B, fake_B, writer: SummaryWriter, global_step: int = 0,
                           use_image_name=False, mode=''):

        wandb_dict = {}
        fig_seg, tag = self.add_segmentation_figures(mode, fake_B, real_B, global_step, writer, use_image_name=use_image_name)
        wandb_dict[tag] = {}
        wandb_dict[tag]['Segmentation'] = fig_seg

        fig_def, _ = self.add_deformable_figures(mode, real_A, self.mask_A, real_B, self.mask_B, self.deformed_moving_real.detach(),
                                    self.mask_moving_deformed, global_step, writer, use_image_name=use_image_name)
        wandb_dict[tag]['Deformable'] = fig_def

        loss_dict = self.add_losses(tag, global_step, writer, use_image_name=use_image_name)
        wandb_dict[tag].update(loss_dict[tag])
        return wandb_dict, tag

    def add_deformable_figures(self, mode, moving_real, mask_moving, fixed, mask_fixed, deformed, mask_deformed, global_step, writer, use_image_name=False):
        n = 10 if mask_fixed is None and mask_moving is not None else 12
        axs, fig = tensorboard.init_figure(3, n)
        tensorboard.set_axs_attribute(axs)
        tensorboard.fill_subplots(self.dvf.cpu()[:, 0:1, ...], axs=axs[0, :], img_name='Def. X', cmap='RdBu',
                                      fig=fig, show_colorbar=True)
        tensorboard.fill_subplots(self.dvf.cpu()[:, 1:2, ...], axs=axs[1, :], img_name='Def. Y', cmap='RdBu',
                                      fig=fig, show_colorbar=True)
        tensorboard.fill_subplots(self.dvf.cpu()[:, 2:3, ...], axs=axs[2, :], img_name='Def. Z', cmap='RdBu',
                                      fig=fig, show_colorbar=True)
        tensorboard.fill_subplots(fixed.detach().cpu(), axs=axs[3, :], img_name='Fixed')
        tensorboard.fill_subplots(moving_real.detach().cpu(), axs=axs[4, :], img_name='Moving')
        tensorboard.fill_subplots(deformed.detach().cpu(), axs=axs[5, :], img_name='Deformed')
        tensorboard.fill_subplots((deformed.detach() - moving_real).abs().cpu(),
                                  axs=axs[6, :], img_name='Deformed - moving')

        overlay = tensorboard.image_over_image(moving_real, fixed)
        tensorboard.fill_subplots(overlay.cpu(), axs=axs[7, :], img_name='Moving overlay', cmap=None)

        overlay = tensorboard.image_over_image(deformed, fixed)
        tensorboard.fill_subplots(overlay.cpu(), axs=axs[8, :], img_name='Deformed overlay', cmap=None)

        overlay = tensorboard.mask_over_mask(mask_moving, mask_deformed)
        tensorboard.fill_subplots(overlay.cpu(), axs=axs[9, :], img_name='Def. mask overlay', cmap=None)

        if mask_fixed is not None and mask_moving is not None:
            overlay = tensorboard.mask_over_mask(mask_fixed, mask_moving)
            dice = self.loss_moving_dice.item() if self.loss_moving_dice is not None else -1
            tensorboard.fill_subplots(overlay.cpu(), axs=axs[10, :],
                                      img_name=f'mask moving on US\nDice {dice:.3f}', cmap=None)

            overlay = tensorboard.mask_over_mask(mask_fixed, mask_deformed)
            dice = self.loss_warped_dice.item() if self.loss_warped_dice is not None else -1
            tensorboard.fill_subplots(overlay.cpu(), axs=axs[11, :],
                                      img_name=f'mask warped on US\nDice {dice:.3f}', cmap=None)
        #
        fig.suptitle(f'ID {self.patient}')
        tag = f'{self.patient[0]}' + mode if use_image_name else mode
        # tag = mode + f'{self.patient}/Deformable' if use_image_name else mode
        writer.add_figure(tag=tag + '/Deformable', figure=fig, global_step=global_step, close=False)

        return fig, tag


    def add_segmentation_figures(self, mode, fake_B, real_B, global_step, writer, use_image_name=False):
        n_rows = 9 if self.mask_B is not None else 7
        axs, fig = tensorboard.init_figure(3, n_rows)
        tensorboard.set_axs_attribute(axs)
        prostate_vol = self.mask_A.sum().item()/1e3
        tensorboard.fill_subplots(self.mask_A.cpu(), axs=axs[0, :], img_name=f'Mask MR\nVolume: {prostate_vol:0.1f}')
        with torch.no_grad():
            seg_fake_B = self.netSeg(self.moving.detach())
            seg_fake_B = torch.argmax(seg_fake_B, dim=1, keepdim=True)
            tensorboard.fill_subplots(seg_fake_B.cpu(), axs=axs[1, :], img_name='Seg fake US')
        idx = 2
        if self.opt.augment_segmentation:
            img = self.augmented_fake.detach()
        else:
            img = fake_B.detach()
        overlay = tensorboard.mask_over_image(self.seg_fake_B, img)
        tensorboard.fill_subplots(overlay.cpu(), axs=axs[idx, :], img_name='Fake mask overlay', cmap=None)
        idx += 1
        tensorboard.fill_subplots(self.mask_moving_deformed.detach().cpu(), axs=axs[idx, :], img_name='Deformed mask')
        idx += 1

        if self.opt.augment_segmentation:
            img = self.augmented_real.detach()
            mask = torch.round(self.augmented_mask[self.opt.batch_size:, ...]).detach()
        else:
            img = real_B.detach()
            mask = self.mask_moving_deformed.detach()
        overlay = tensorboard.mask_over_image(mask, img, one_hot=True)
        tensorboard.fill_subplots(overlay.cpu(), axs=axs[idx, :], img_name='Def. mask overlay', cmap=None)
        idx += 1

        seg_B_img = torch.argmax(self.seg_B, dim=1, keepdim=True)
        tensorboard.fill_subplots(seg_B_img.detach().cpu(), axs=axs[idx, :], img_name='Seg. US')
        idx += 1

        overlay = tensorboard.mask_over_image(seg_B_img, img, one_hot=True)
        tensorboard.fill_subplots(overlay.cpu(), axs=axs[idx, :], img_name='Seg. US overlay', cmap=None)
        idx += 1

        if self.mask_B is not None:
            if self.opt.augment_segmentation:
                mask_gt = self.augmented_mask_B.detach()
                mask_pre = torch.round(self.augmented_mask[:self.opt.batch_size, ...]).detach()
            else:
                mask_gt = self.mask_B
                mask_pre = self.mask_A
            mask = seg_B_img.detach()
            dice_score_pos = compute_meandice(mask, mask_gt, include_background=False).mean()
            dice_score_pre = compute_meandice(mask_pre, mask_gt, include_background=False).mean()
            # print(f'Seg. US on GT, DSC:{dice_score_pos.item():0.2f}/{dice_score_pre.item():0.2f}')
            overlay = tensorboard.mask_over_mask(mask_gt, mask)
            tensorboard.fill_subplots(overlay.cpu(), axs=axs[idx, :],
                                      img_name=f'Seg. US on GT\n'
                                               f'DSC:{dice_score_pos.item():0.2f}/{dice_score_pre.item():0.2f}', cmap=None)
            idx += 1

            prostate_vol = self.mask_B.sum().item() / 1e3
            tensorboard.fill_subplots(self.mask_B.cpu(), axs=axs[idx, :], img_name=f'Mask US\nVolume: {prostate_vol:0.1f}')
            idx += 1

        tag = mode + f'{self.patient[0]}' if use_image_name else mode
        # if use_image_name:
        #     tag = mode + f'{self.patient[0]}/Segmentation'
        # else:
        #     tag = mode + '/Segmentation'
        writer.add_figure(tag=tag + '/Segmentation', figure=fig, global_step=global_step, close=False)

        return fig, tag


    def add_losses(self, mode, global_step, writer, use_image_name=False):

        wandb_dict = {}
        wandb_dict[mode] = {}
        if self.loss_landmarks_def is not None:
            writer.add_scalar(mode + '/landmarks/def', scalar_value=self.loss_landmarks_def,
                              global_step=global_step)
            wandb_dict[mode]['landmarks def'] = self.loss_landmarks_def
        if self.loss_landmarks_def_diff is not None:
            writer.add_scalar(mode + '/landmarks/difference_def', scalar_value=self.loss_landmarks_def_diff,
                              global_step=global_step)
            wandb_dict[mode]['landmarks difference_def'] = self.loss_landmarks_def_diff
        if self.loss_diff_dice is not None:
            writer.add_scalar(mode + '/DICE/difference', scalar_value=self.loss_diff_dice, global_step=global_step)
            writer.add_scalar(mode + '/DICE/deformed', scalar_value=self.loss_warped_dice,
                              global_step=global_step)
            writer.add_scalar(mode + '/DICE/moving', scalar_value=self.loss_moving_dice,
                              global_step=global_step)
            wandb_dict[mode]['DICE difference'] = self.loss_diff_dice
            wandb_dict[mode]['DICE deformed'] = self.loss_warped_dice
            wandb_dict[mode]['DICE moving'] = self.loss_moving_dice

        jacob = vxm.py.utils.jacobian_determinant(
            self.dvf[0, ...].permute(*range(1, len(self.dvf.shape) - 1), 0).cpu().numpy())
        c = jacob[jacob < 0].size
        r = c / jacob.size
        writer.add_scalar(mode + '/Jacobian/count', scalar_value=c, global_step=global_step)
        writer.add_scalar(mode + '/Jacobian/ratio', scalar_value=r, global_step=global_step)
        wandb_dict[mode]['Jacobian count'] = c
        wandb_dict[mode]['Jacobian ratio'] = r
        return wandb_dict

    def get_current_landmark_distances(self):
        return self.loss_landmarks_beginning, self.loss_landmarks_def

    def set_coeff_multitask_loss(self, epoch):
        if epoch >= self.opt.epochs_before_reg:
            self.first_phase_coeff = 1 / (epoch + 1 - self.opt.epochs_before_reg)