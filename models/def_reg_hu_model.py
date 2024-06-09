import os
import sys
from collections import OrderedDict

import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from monai.metrics import compute_meandice, compute_hausdorff_distance, compute_average_surface_distance
from monai.losses import DiceLoss

from .base_model import BaseModel
from .multitask_parent import Multitask
from . import networks3d
from torch.utils.tensorboard import SummaryWriter
from util import tensorboard, distance_landmarks
from monai.visualize import img2tensorboard
from util import affine_transform

os.environ['VXM_BACKEND'] = 'pytorch'
from voxelmorph import voxelmorph as vxm


from . import networks


class DefRegHuModel(BaseModel, Multitask):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        parser.set_defaults(norm='batch', netG='unet_128', dataset_mode='volume', batch_size=1, mask_B_required=True, dual_mode=False)

        # voxelmorph params
        parser.add_argument('--cudnn-nondet', action='store_true',
                            help='disable cudnn determinism - might slow down training')
        # network architecture parameters
        parser.add_argument('--enc', type=int, nargs='+', help='list of unet encoder filters (default: 16 32 32 32)')
        parser.add_argument('--dec', type=int, nargs='+',
                            help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
        parser.add_argument('--int-steps', type=int, default=7, help='number of integration steps (default: 7)')
        parser.add_argument('--int-downsize', type=int, default=2,
                            help='flow downsample factor for integration (default: 2)')
        parser.add_argument('--bidir', action='store_true', help='enable bidirectional cost function')
        parser.add_argument('--use-probs', action='store_true', help='enable probabilities')
        parser.add_argument('--kl-lambda', type=float, default=10,
                            help='prior lambda regularization for KL loss (default: 10)')
        parser.add_argument('--flow-logsigma-bias', type=float, default=-10,
                            help='negative value for initialization of the logsigma layer bias value')
        parser.add_argument('--rgl_lambda', type=float, default=1,
                            help='energy regularization coeff (default: 1)')

        parser.add_argument('--show_volumes', type=bool, default=False, help='visualize transformed volumes w napari')
        parser.add_argument('--num-classes', type=int, default=2, help='num of classes for segmentation')
        # parser.add_argument('--use_mask', action='store_true', help='use mask for registration')

        if is_train:
            parser.add_argument('--similarity', type=str, default='NCC', choices=['NCC', 'MIND'],
                                help='type of the similarity used for training voxelmorph')


        return parser


    def __init__(self, opt):
        """Initialize the class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        For visualization, we set 'visual_names' as 'real_A' (input real image),
        'real_B_rgb' (ground truth RGB image), and 'fake_B_rgb' (predicted RGB image)
        We convert the Lab image 'real_B' (inherited from Pix2pixModel) to a RGB image 'real_B_rgb'.
        we convert the Lab image 'fake_B' (inherited from Pix2pixModel) to a RGB image 'fake_B_rgb'.
        """
        super().__init__(opt)
        self.isTrain = opt.isTrain

        # specify the training losses you want to print out. The training/test scripts will call
        # <BaseModel.get_current_losses>
        self.loss_names = ['G', 'DefReg', 'DefRgl']
        self.loss_names += ['diff_dice', 'moving_dice', 'warped_dice']
        self.loss_names += ['warped_HD', 'moving_HD', 'diff_HD']
        self.loss_names += ['warped_ASD', 'moving_ASD', 'diff_ASD']

        self.loss_functions = ['backward_defReg', 'compute_landmark_loss']
        # specify the models you want to save to the disk. The training/test scripts will call
        # <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['DefReg']
        else:  # during test time, only load G
            self.model_names = ['DefReg']

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
        self.netDefReg = networks3d.init_net(self.netDefReg, gpu_ids=self.opt.gpu_ids)

        self.transformer_label = networks3d.init_net(vxm.layers.SpatialTransformer(size=opt.inshape), #, mode='nearest'
                                                     gpu_ids=opt.gpu_ids)
        self.resizer = networks3d.init_net(vxm.layers.ResizeTransform(vel_resize=0.5, ndims=3), gpu_ids=opt.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks3d.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                            opt.netD, opt.n_layers_D, norm=opt.norm,
                                            use_sigmoid=use_sigmoid, gpu_ids=self.gpu_ids)

        if self.isTrain:
            # define loss functions

            # self.criterionDefReg = getattr(sys.modules['models.networks'], self.opt.similarity + 'Loss')()
            self.criterionDefReg = DiceLoss(include_background=False, to_onehot_y=True)
            self.criterionDefRgl = networks.GradLoss('l2', loss_mult=self.opt.int_downsize)
            self.optimizer_DefReg = torch.optim.Adam(self.netDefReg.parameters(), lr=self.opt.lr)
            self.optimizers.append(self.optimizer_DefReg)

            self.transformer = vxm.layers.SpatialTransformer((1, 1, 80, 80, 80))
            # resize = vxm.layers.ResizeTransform(0.5, 1)

    # def name(self):
    #     return 'Pix2Pix3dModel'

    def set_input(self, input):
        self.clean_tensors()
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.fake_B = torch.zeros_like(self.real_A)
        self.mask_A = input['A_mask'].to(self.device).type(self.real_A.dtype)

        if input['B_mask_available'][0]:  # TODO in this way it only works with batch size 1!
            self.mask_B = input['B_mask'].to(self.device).type(self.real_A.dtype)
        else:
            self.mask_B = None

        self.set_mt_input(input, real_B=self.real_B, shape=self.real_B.shape,
                          dtype=self.real_B.dtype, device=self.real_B.device)

        self.init_loss_tensors()

        # self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.fixed = self.real_B * 0.5 + 0.5
        self.moving = self.real_A * 0.5 + 0.5

        # fov_mask_moving = self.moving > self.moving.min()
        # fov_mask_fixed = self.fixed > self.fixed.min()
        # if self.opt.use_mask:
        #     self.fov_mask = fov_mask_moving * fov_mask_fixed
        # else:
        #     self.fov_mask = torch.ones_like(fov_mask_moving)

        self.mask_moving = self.mask_A
        if self.mask_B is not None:
            self.mask_fixed = self.mask_B
        else:
            self.mask_fixed = None
        def_reg_output = self.netDefReg(self.moving, self.fixed, registration=not self.isTrain)

        if self.opt.bidir and self.isTrain:
            (self.deformed_moving, self.deformed_fixed, self.dvf) = def_reg_output
        else:
            (self.deformed_moving, self.dvf) = def_reg_output

        if self.isTrain:
            self.dvf = self.resizer(self.dvf)

        self.deformed_moving_mask = self.transformer_label(self.mask_moving, self.dvf)
        self.deformed_moving_mask_rounded = torch.round(self.deformed_moving_mask)
        self.landmark_A_dvf = self.transformer_label(self.landmarks_A, self.dvf.detach())
        self.compute_gt_dice()

    # def get_current_landmark_distances(self):
    #     return torch.tensor([0]), torch.tensor([0]), torch.tensor([0])

    def compute_gt_dice(self):
        """
        calculate the dice score between the deformed mask and the ground truth if we have it
        Returns
        -------

        """
        if self.mask_fixed is not None:
            shape = list(self.mask_fixed.shape)
            n = self.opt.num_classes  # number of classes
            shape[1] = n
            one_hot_fixed = F.one_hot(self.mask_fixed.to(torch.int64)).squeeze(dim=1).permute([0, 4, 1, 2, 3])
            one_hot_deformed = F.one_hot(self.deformed_moving_mask_rounded.to(torch.int64)).squeeze(dim=1).permute([0, 4, 1, 2, 3])
            one_hot_moving = F.one_hot(self.mask_moving.to(torch.int64)).squeeze(dim=1).permute([0, 4, 1, 2, 3])

            self.loss_warped_dice = compute_meandice(one_hot_deformed, one_hot_fixed,
                                                     include_background=False)
            self.loss_moving_dice = compute_meandice(one_hot_moving, one_hot_fixed, include_background=False)
            self.loss_diff_dice = self.loss_warped_dice - self.loss_moving_dice

            self.loss_warped_HD = compute_hausdorff_distance(one_hot_deformed, one_hot_fixed, percentile=95).mean()
            self.loss_moving_HD = compute_hausdorff_distance(one_hot_moving, one_hot_fixed, percentile=95).mean()
            self.loss_diff_HD = 100 * (self.loss_warped_HD - self.loss_moving_HD)/self.loss_moving_HD

            self.loss_warped_ASD = compute_average_surface_distance(one_hot_deformed, one_hot_fixed).mean()
            self.loss_moving_ASD = compute_average_surface_distance(one_hot_moving, one_hot_fixed).mean()
            self.loss_diff_ASD = 100 * (self.loss_warped_ASD - self.loss_moving_ASD) / self.loss_moving_ASD
        else:
            self.loss_warped_dice = None
            self.loss_moving_dice = None
            self.loss_diff_dice = None

            self.loss_warped_HD, self.loss_moving_HD, self.loss_diff_HD = None, None, None
            self.loss_warped_ASD, self.loss_moving_ASD, self.loss_diff_ASD = None, None, None

    def backward_defReg(self):

        if self.mask_fixed is not None:
            self.loss_DefReg = self.criterionDefReg(self.deformed_moving_mask, self.mask_fixed)
        else:
            self.loss_DefReg = torch.tensor([0.0], device=self.moving.device)

        self.loss_DefRgl = self.criterionDefRgl(self.dvf, None)
        self.loss_DefReg += self.loss_DefRgl * self.opt.rgl_lambda

        self.loss_G = self.loss_DefReg
        if torch.is_grad_enabled():
            self.loss_DefReg.backward()

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A)
        # update Def Reg
        self.optimizer_DefReg.zero_grad()  # set G's gradients to zero
        self.backward_defReg()  # calculate graidents for G
        self.optimizer_DefReg.step()  # udpate G's weights

    def log_mt_tensorboard(self, real_A, real_B, fake_B, writer: SummaryWriter, global_step: int = 0,
                           use_image_name=False, mode=''):
        return {}, None

    def log_tensorboard(self, writer: SummaryWriter, losses: OrderedDict = None, global_step: int = 0, save_gif=False,
                        use_image_name=False, mode='', epoch=0, save_pdf=False):

        wandb_dict = {}

        fig_def, tag = self.add_deformable_figures(mode, self.moving, self.mask_A, self.fixed, self.mask_fixed, self.deformed_moving,
                                    self.deformed_moving_mask_rounded, global_step, writer, use_image_name)
        wandb_dict[tag] = {}
        wandb_dict[tag]['Deformable'] = fig_def

        # save fig as pdf
        if save_pdf:
            pdf_dir = os.path.join(writer.log_dir, 'pdf/', f'{epoch:03}')
            if not os.path.isdir(pdf_dir):
                os.makedirs(pdf_dir, exist_ok=True)
            fig_def.savefig(f'{os.path.join(pdf_dir, tag + "deformable")}.pdf')

        # self.add_deformable_figures(mode=mode, global_step=global_step, writer=writer, use_image_name=use_image_name)
        if losses is not None:
            for key in losses:
                writer.add_scalar(f'losses/{key}', scalar_value=losses[key], global_step=global_step)

        loss_dict = self.add_losses(tag, global_step, writer, use_image_name=use_image_name)
        wandb_dict[tag].update(loss_dict[tag])

        return wandb_dict

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass




