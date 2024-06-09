import argparse
import os
from collections import OrderedDict
from typing import Tuple
import util.util as util
from util.image_pool import ImagePool
from util import tensorboard
from util import affine_transform
from models.base_model import BaseModel
from torch.autograd import Variable
from monai.visualize import img2tensorboard
import matplotlib.pyplot as plt

import torch
from torch.utils.tensorboard import SummaryWriter
import sys
from models import networks3d

os.environ['VXM_BACKEND'] = 'pytorch'
from voxelmorph import voxelmorph as vxm


class Pix2PixHDModel(BaseModel):

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

        # generator
        parser.add_argument('--n_downsample_global', type=int, default=2, help='number of downsampling layers in netG')
        parser.add_argument('--n_blocks_global', type=int, default=4, help='number of residual blocks in the global generator network')
        parser.add_argument('--n_blocks_local', type=int, default=4, help='number of residual blocks in the local enhancer network')
        parser.add_argument('--n_local_enhancers', type=int, default=1, help='number of local enhancers to use')
        # parser.add_argument('--niter_fix_global', type=int, default=0, help='number of epochs that we only train the outmost local enhancer')

        # for instance-wise features
        parser.add_argument('--no_instance', action='store_true', help='if specified, do *not* add instance map as input')
        parser.add_argument('--instance_feat', action='store_true', help='if specified, add encoded instance features as input')
        parser.add_argument('--label_feat', action='store_true', help='if specified, add encoded label features as input')
        parser.add_argument('--feat_num', type=int, default=3, help='vector length for encoded features')
        parser.add_argument('--n_downsample_E', type=int, default=3, help='# of downsampling layers in encoder')
        parser.add_argument('--nef', type=int, default=16, help='# of encoder filters in the first conv layer')
        parser.add_argument('--n_clusters', type=int, default=10, help='number of clusters for features')

        parser.add_argument('--show_volumes', type=bool, default=False, help='visualize transformed volumes w napari')
        parser.set_defaults(norm='batch', dataset_mode='volume')

        parser.set_defaults(pool_size=0)  # no image pooling
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--no-lsgan', type=bool, default=False)

        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT

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
        super().__init__(opt)
        self.isTrain = opt.isTrain

        self.set_visdom_names()

        # HD
        self.input_nc = opt.input_nc
        # Generator network
        self.netG_input_nc = opt.input_nc
        if self.isTrain:
            self.use_sigmoid = opt.no_lsgan
            self.netD_input_nc = self.netG_input_nc + 1
            if not opt.no_instance:
                self.netD_input_nc += 1
#
        self.use_features = opt.instance_feat or opt.label_feat
        self.gen_features = self.use_features

        self.set_networks(opt)

        if self.isTrain:

            # pix2pixHD optimizers
            # optimizer G
            params = list(self.netG.parameters())
            # if self.gen_features:
            #     params += list(self.netE.parameters())
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            # optimizer D
            params = list(self.netD.parameters())
            params += list(self.netD_DN.parameters())
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            # pix2pixHD
            self.fake_pool = ImagePool(opt.pool_size)
            # self.loss_filter = self.init_loss_filter(not opt.no_ganFeat_loss)
            self.criterionGAN = networks3d.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionFeat = torch.nn.L1Loss()

        self.first_phase_coeff = 1

    def set_networks(self, opt):
        # specify the models you want to save to the disk. The training/test scripts will call
        # <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D', 'D_DN']
        else:  # during test time, only load G
            self.model_names = ['G']

        # pix2pixHD

        self.netG = networks3d.define_G(input_nc=self.netG_input_nc, output_nc=opt.output_nc, ngf=opt.ngf,
                                        netG='local', # set to local since we want to train the HD in one pass
                                        n_downsample_global=opt.n_downsample_global,
                                        n_blocks_global=opt.n_blocks_global, n_local_enhancers=opt.n_local_enhancers,
                                        n_blocks_local=opt.n_blocks_local, norm=opt.norm, gpu_ids=self.gpu_ids)

        self.netD = networks3d.define_D_HD(self.netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, self.use_sigmoid,
                                           opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)
        self.netD_DN = networks3d.define_D_HD(self.netD_input_nc, opt.ndf, opt.n_layers_D - 1, opt.norm, self.use_sigmoid,
                                              opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)
        # self.netE = networks3d.define_G(input_nc=opt.output_nc, output_nc=opt.feat_num, ngf=opt.nef, netG='encoder',
        #                                 n_downsample_global=opt.n_downsample_E, norm=opt.norm, gpu_ids=self.gpu_ids)
        self.denoiser = networks3d.ImageDenoise()

    def set_visdom_names(self):
        # specify the training losses you want to print out. The training/test scripts will call
        # <BaseModel.get_current_losses>
        # Names so we can breakout loss
        # self.losses_pix2pix = ['G_GAN', 'G_GAN_Feat', 'D_real', 'D_fake']

        self.loss_names = ['G', 'G_GAN_Feat', 'G_GAN', 'D_real', 'D_fake',
                           'D_fake_dn', 'D_real_dn', 'G_GAN_dn', 'G_GAN_Feat_dn']
        self.visual_names = []
        self.loss_functions = ['backward_G', 'compute_D_loss']

    def clean_tensors(self):
        all_members = self.__dict__.keys()
        # print(f'{all_members}')
        # GPUtil.showUtilization()
        for item in all_members:
            if isinstance(self.__dict__[item], torch.Tensor):
                self.__dict__[item] = None
        torch.cuda.empty_cache()
        # GPUtil.showUtilization()

    def set_input(self, input):
        self.clean_tensors()
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.real_B_dn = self.denoiser(self.real_B)
        self.patient = input['Patient']
        self.mask_A = input['A_mask'].to(self.device).type(self.real_A.dtype)
        if input['B_mask_available'][0]:  # TODO in this way it only works with batch size 1
            self.mask_B = input['B_mask'].to(self.device).type(self.real_A.dtype)
        else:
            self.mask_B = None


        self.loss_G_GAN = torch.tensor([0.0])
        self.loss_G_GAN_Feat = torch.tensor([0.0])
        self.loss_D_real = torch.tensor([0.0])
        self.loss_D_fake = torch.tensor([0.0])
        self.loss_D_fake_dn = torch.tensor([0.0])
        self.loss_D_real_dn = torch.tensor([0.0])
        self.netG_input_nc = torch.tensor([0.0])

    def discriminate(self, input_label, test_image, netD: torch.nn.Module, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:
            fake_query = self.fake_pool.query(input_concat)
            return netD.forward(fake_query)
        else:
            return netD.forward(input_concat)

    def forward(self):
        # pix2pixHD
        self.input_cat = torch.cat((self.mask_A, self.real_A), dim=1)
        self.input_cat_dn = self.denoiser(self.input_cat)
        self.fake_B, self.fake_B_dn = self.netG.forward(self.real_A)

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        self.loss_G = 0.0
        ########   pix2pix HD    ########

        # GAN loss (Fake Passability Loss)
        pred_fake = self.netD.forward(torch.cat((self.input_cat, self.fake_B), dim=1))
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        pred_fake_dn = self.netD_DN.forward(torch.cat((self.input_cat_dn, self.fake_B_dn), dim=1))
        loss_D_fake_dn = self.criterionGAN(pred_fake_dn, True)
        self.loss_G_GAN_dn = loss_D_fake_dn.detach()
        self.loss_G_GAN += loss_D_fake_dn

        pred_real = self.discriminate(self.input_cat, self.real_B, netD=self.netD)
        pred_real_dn = self.discriminate(self.input_cat_dn, self.real_B_dn, netD=self.netD_DN)

        # GAN feature matching loss
        self.loss_G_GAN_Feat = 0
        self.loss_G_GAN_Feat_dn = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake[i]) - 1):
                    self.loss_G_GAN_Feat += D_weights * feat_weights * \
                                            self.criterionFeat(pred_fake[i][j],
                                                               pred_real[i][j].detach()) * self.opt.lambda_feat
            # Denoised image
            feat_weights = 4.0 / self.opt.n_layers_D
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake_dn[i]) - 1):
                    self.loss_G_GAN_Feat_dn += D_weights * feat_weights * \
                                            self.criterionFeat(pred_fake_dn[i][j],
                                                               pred_real_dn[i][j].detach()) * self.opt.lambda_feat

        self.loss_pix2pix = self.loss_G_GAN
        if not self.opt.no_ganFeat_loss:
            self.loss_pix2pix += self.loss_G_GAN_Feat + self.loss_G_GAN_Feat_dn

        ########   END pix2pix HD    ########

        self.loss_G = self.loss_pix2pix

        if torch.is_grad_enabled():
            self.loss_G.backward()

    def backward_D(self):

        # Fake Detection and Loss

        self.compute_D_loss()
        self.loss_D.backward()


    def optimize_parameters(self):
        self.forward()

        # update G
        # self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G

        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate gradients for G
        self.optimizer_G.step()  # update G's weights

        # Update D
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        pred_fake_pool = self.discriminate(self.input_cat, self.fake_B, netD=self.netD, use_pool=True)
        self.loss_D_fake = self.criterionGAN(pred_fake_pool, False)

        pred_fake_pool_dn = self.discriminate(self.input_cat_dn, self.fake_B_dn, netD=self.netD_DN, use_pool=True)
        loss_D_fake_dn = self.criterionGAN(pred_fake_pool_dn, False)
        self.loss_D_fake_dn = loss_D_fake_dn.detach()
        self.loss_D_fake += loss_D_fake_dn

        # Real Detection and Loss
        pred_real = self.discriminate(self.input_cat, self.real_B, netD=self.netD)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        pred_real_dn = self.discriminate(self.input_cat_dn, self.real_B_dn, netD=self.netD_DN)
        loss_D_real_dn = self.criterionGAN(pred_real_dn, True)
        self.loss_D_real_dn = loss_D_real_dn.detach()
        self.loss_D_real += loss_D_real_dn

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def update_learning_rate(self, epoch=0):
        super().update_learning_rate(epoch=epoch)

    def log_tensorboard(self, writer: SummaryWriter, losses: OrderedDict = None, global_step: int = 0, save_gif=False,
                        use_image_name=False, mode='', epoch=0, save_pdf=False):
        image = torch.add(torch.mul(self.real_A, 0.5), 0.5)
        image2 = torch.add(torch.mul(self.real_B, 0.5), 0.5)
        image3 = torch.add(torch.mul(self.fake_B, 0.5), 0.5)

        if save_gif:
            img2tensorboard.add_animated_gif(writer=writer, scale_factor=256, tag="GAN/Real A", max_out=85,
                                             image_tensor=image.squeeze(dim=0).cpu().detach().numpy(),
                                             global_step=global_step)
            img2tensorboard.add_animated_gif(writer=writer, scale_factor=256, tag="GAN/Real B", max_out=85,
                                             image_tensor=image2.squeeze(dim=0).cpu().detach().numpy(),
                                             global_step=global_step)
            img2tensorboard.add_animated_gif(writer=writer, scale_factor=256, tag="GAN/Fake B", max_out=85,
                                             image_tensor=image3.squeeze(dim=0).cpu().detach().numpy(),
                                             global_step=global_step)

        axs, fig = tensorboard.init_figure(3, 4)
        tensorboard.set_axs_attribute(axs)
        tensorboard.fill_subplots(self.real_A.cpu(), axs=axs[0, :], img_name='A')
        tensorboard.fill_subplots(self.fake_B.detach().cpu(), axs=axs[1, :], img_name='fake')
        tensorboard.fill_subplots(self.real_B.cpu(), axs=axs[2, :], img_name='B')
        tensorboard.fill_subplots(self.fake_B_dn.cpu(), axs=axs[3, :], img_name='fake_B_DN')
        if use_image_name:
            tag = mode + f'{self.patient}/GAN'
        else:
            tag = mode + 'GAN'
        writer.add_figure(tag=tag, figure=fig, global_step=global_step, close=False)

        wandb_dict = {mode + tag: fig}
        self.log_loss_tensorboard(global_step, losses, wandb_dict, writer)
        return wandb_dict

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        n_c = self.real_A.shape[2]
        # average over channel to get the real and fake image
        self.real_A_center_sag = self.real_A[:, :, int(n_c / 2), ...]
        self.fake_B_center_sag = self.fake_B[:, :, int(n_c / 2), ...]
        self.real_B_center_sag = self.real_B[:, :, int(n_c / 2), ...]

        n_c = self.real_A.shape[3]
        self.real_A_center_cor = self.real_A[:, :, :, int(n_c / 2), ...]
        self.fake_B_center_cor = self.fake_B[:, :, :, int(n_c / 2), ...]
        self.real_B_center_cor = self.real_B[:, :, :, int(n_c / 2), ...]

        n_c = self.real_A.shape[4]
        self.real_A_center_axi = self.real_A[..., int(n_c / 2)]
        self.fake_B_center_axi = self.fake_B[..., int(n_c / 2)]
        self.real_B_center_axi = self.real_B[..., int(n_c / 2)]
