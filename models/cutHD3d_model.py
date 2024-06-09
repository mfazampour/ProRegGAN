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

os.environ['VXM_BACKEND'] = 'pytorch'
from voxelmorph import voxelmorph as vxm


#
class CUTHD3dModel(BaseModel):
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
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False,
                            help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'],
                            help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")

        parser.add_argument('--n_downsample_global', type=int, default=2, help='number of downsampling layers in netG')
        parser.add_argument('--n_blocks_global', type=int, default=4,
                            help='number of residual blocks in the global generator network')
        parser.add_argument('--n_blocks_local', type=int, default=4,
                            help='number of residual blocks in the local enhancer network')
        parser.add_argument('--n_local_enhancers', type=int, default=1, help='number of local enhancers to use')
        # parser.add_argument('--niter_fix_global', type=int, default=0, help='number of epochs that we only train the outmost local enhancer')

        # for instance-wise features
        parser.add_argument('--no_instance', action='store_true',
                            help='if specified, do *not* add instance map as input')
        parser.add_argument('--instance_feat', action='store_true',
                            help='if specified, add encoded instance features as input')
        parser.add_argument('--label_feat', action='store_true',
                            help='if specified, add encoded label features as input')
        parser.add_argument('--feat_num', type=int, default=3, help='vector length for encoded features')
        parser.add_argument('--n_downsample_E', type=int, default=3, help='# of downsampling layers in encoder')
        parser.add_argument('--nef', type=int, default=16, help='# of encoder filters in the first conv layer')
        parser.add_argument('--n_clusters', type=int, default=10, help='number of clusters for features')
        parser.add_argument('--mask_contrastive_loss', action='store_true', help='considers real img mask when calculating contrastive loss')

        parser.add_argument('--coarse_only', action='store_true', help='if True only the coarse stream is trained')

        parser.set_defaults(pool_size=0, dataset_mode='volume', netG='local', dual_mode=False)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if opt.CUT_mode.lower() == "cut":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        elif opt.CUT_mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False, lambda_NCE=10.0, flip_equivariance=True,
                n_epochs=150, n_epochs_decay=50
            )
        else:
            raise ValueError(opt.CUT_mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE', 'G_GAN_Feat',
                           'D_fake_dn', 'D_real_dn', 'G_GAN_dn', 'G_GAN_Feat_dn']

        self.visual_names = ['real_A_center_sag', 'real_A_center_cor', 'real_A_center_axi']
        self.visual_names += ['fake_B_center_sag', 'fake_B_center_cor', 'fake_B_center_axi']
        self.visual_names += ['real_B_center_sag', 'real_B_center_cor', 'real_B_center_axi']

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B_center_sag', 'idt_B_center_cor', 'idt_B_center_axi']


        self.loss_functions = ['compute_G_loss', 'compute_D_loss']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D', 'D_DN', 'F_DN', 'D_N']
        else:  # during test time, only load G
            self.model_names = ['G']

        # HD
        # self.input_nc = opt.input_nc
        # Generator network
        if self.isTrain:
            self.use_sigmoid = opt.no_lsgan
            self.netD_input_nc = opt.input_nc #+ 1
            if not opt.no_instance:
                self.netD_input_nc += 1

        # define networks (both generator and discriminator)
        self.netG = networks3d.define_G(input_nc=opt.input_nc, output_nc=opt.output_nc, ngf=opt.ngf,
                                        netG=opt.netG,  # set to local since we want to train the HD in one pass
                                        n_downsample_global=opt.n_downsample_global, opt=self.opt,
                                        n_blocks_global=opt.n_blocks_global, n_local_enhancers=opt.n_local_enhancers,
                                        n_blocks_local=opt.n_blocks_local, norm=opt.norm, gpu_ids=self.gpu_ids)
        if isinstance(self.netG, networks3d.NoiseAware):
            self.netG.init_speckle_layer()

        self.netF = networks3d.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type,
                                        opt.init_gain, opt.no_antialias, gpu_ids=self.gpu_ids, opt=opt)
        self.netF_DN = networks3d.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type,
                                        opt.init_gain, opt.no_antialias, gpu_ids=self.gpu_ids, opt=opt)

        self.denoiser = networks3d.ImageDenoise()

        if self.isTrain:
            self.netD = networks3d.define_D(self.netD_input_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type,
                                            opt.init_gain, no_antialias=opt.no_antialias, gpu_ids=self.gpu_ids, opt=opt)

            self.netD_DN = networks3d.define_D(self.netD_input_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type,
                                               opt.init_gain, no_antialias=opt.no_antialias, gpu_ids=self.gpu_ids, opt=opt)

            # Discriminator network for the generated noise
            self.netD_N = networks3d.define_D(self.netD_input_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD,
                                               opt.init_type, opt.init_gain, no_antialias=opt.no_antialias,
                                               gpu_ids=self.gpu_ids, opt=opt)

            # define loss functions
            self.criterionGAN_syn = networks3d.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionNCE = PatchNCELoss(opt).to(self.device)
            # for _ in range(self.opt.n_local_enhancers + 1):
            #     self.criterionNCE.append(PatchNCELoss(opt).to(self.device))
            self.criterionFeat = torch.nn.L1Loss()

            params = list(self.netG.parameters())
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, opt.beta2))
            params = list(self.netD.parameters())
            params += list(self.netD_DN.parameters())
            params += list(self.netD_N.parameters())
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            # pix2pixHD
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr
            # self.loss_filter = self.init_loss_filter(not opt.no_ganFeat_loss)
            # self.criterionGAN = networks3d.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)

    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        self.set_input(data)
        bs_per_gpu = self.real_A.size(0) // max(len(self.opt.gpu_ids), 1)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]

        self.forward()  # compute fake images: G(A)
        if self.opt.isTrain:
            self.compute_D_loss().backward()  # calculate gradients for D
            self.compute_G_loss().backward()  # calculate graidents for G
            if self.opt.lambda_NCE > 0.0:
                params = list(self.netF.parameters())+list(self.netF_DN.parameters())
                self.optimizer_F = torch.optim.Adam(params, lr=self.opt.lr,
                                                    betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B_unfiltered'].to(self.device)
        self.real_B = input['B_unfiltered' if AtoB else 'A'].to(self.device)
        self.real_B_dn = input['B_denoised'].to(self.device)
        self.mask_A = input['A_mask'].to(self.device).type(self.real_A.dtype)

        if input['B_mask_available'][0]:
            self.mask_B = input['B_mask'].to(self.device).type(self.real_A.dtype)
        else:
            self.mask_B = None

        self.patient = input['Patient']
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

        if self.opt.load_second_sample:
            self.real_B = input['B2_B'].to(self.device)
            self.real_B_dn = input['B2_B_denoised'].to(self.device)
            if self.mask_B is not None:
                self.mask_B = input['B2_B_mask'].to(self.device).type(self.real_A.dtype)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # Both real_B and real_A if we also use the loss from the identity mapping: NCE(G(Y), Y)) in NCE loss

        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt else self.real_A

        # Inspired by GcGAN, FastCUT is trained with flip-equivariance augmentation, where
        # the input image to the generator is horizontally flipped, and the output features
        # are flipped back before computing the PatchNCE loss
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])
                self.real_dn = torch.flip(self.real_dn, [3])

        self.fake, self.fake_dn = self.netG(self.real)
        if self.opt.coarse_only:
            self.fake = self.fake.detach()

        self.fake_B = self.fake[:self.real_A.size(0)]
        self.fake_B_dn = self.fake_dn[:self.real_A.size(0)]

        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]
            self.idt_B_dn = self.fake_dn[self.real_A.size(0):]

    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.set_requires_grad(self.netD_DN, True)
        self.set_requires_grad(self.netD_N, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.set_requires_grad(self.netD_DN, False)
        self.set_requires_grad(self.netD_N, False)
        self.optimizer_G.zero_grad()
        if self.opt.lambda_NCE > 0.0 and self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        if self.opt.lambda_NCE > 0.0 and self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()

    def discriminate(self, input_label, test_image, netD: torch.nn.Module, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:
            fake_query = self.fake_pool.query(input_concat)
            return netD.forward(fake_query)
        else:
            return netD.forward(input_concat)

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake_pool = self.discriminate(self.real_A, self.fake_B.detach(), netD=self.netD, use_pool=True)
        self.loss_D_fake = self.criterionGAN_syn(pred_fake_pool, False).mean()

        pred_fake_pool_dn = self.discriminate(self.real_A, self.fake_B_dn.detach(), netD=self.netD_DN, use_pool=True)
        self.loss_D_fake_dn = self.criterionGAN_syn(pred_fake_pool_dn, False).mean()
        self.loss_D_fake += self.loss_D_fake_dn

        pred_fake_pool_n = self.discriminate(self.real_A, (self.fake_B - self.fake_B_dn).detach(), netD=self.netD_N, use_pool=True)
        self.loss_D_fake_n = self.criterionGAN_syn(pred_fake_pool_n, False).mean()
        self.loss_D_fake += self.loss_D_fake_n
        # Real
        pred_real = self.discriminate(self.real_A, self.real_B, netD=self.netD)
        self.loss_D_real = self.criterionGAN_syn(pred_real, True).mean()

        pred_real_dn = self.discriminate(self.real_A, self.real_B_dn, netD=self.netD_DN)
        self.loss_D_real_dn = self.criterionGAN_syn(pred_real_dn, True).mean()
        self.loss_D_real += self.loss_D_real_dn

        pred_real_n = self.discriminate(self.real_A, (self.real_B - self.real_B_dn).detach(), netD=self.netD_N)
        self.loss_D_real_n = self.criterionGAN_syn(pred_real_n, True).mean()
        self.loss_D_real += self.loss_D_real_n

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

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
            if not self.opt.coarse_only:
                self.loss_NCE += self.calculate_NCE_loss(self.real_A, self.fake_B, input_is_denoised=False,
                                                         mask=self.mask_A if self.opt.mask_contrastive_loss else None)
        else:
            self.loss_NCE= 0.0

        # For contrastive loss between Y and G(Y) but nce_idt is by default 0.0
        # Lambda = 1 by default
        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B, input_is_denoised=False,
                                                      mask=self.mask_B if self.opt.mask_contrastive_loss else None)
            self.loss_NCE_Y += self.calculate_NCE_loss(self.real_B_dn, self.idt_B_dn, input_is_denoised=True,
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

    def feature_matching_loss(self, pred_fake, pred_fake_dn, pred_real, pred_real_dn):
        loss_G_GAN_Feat = 0
        loss_G_GAN_Feat_dn = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake[i]) - 1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                                       self.criterionFeat(pred_fake[i][j],
                                                          pred_real[i][j].detach()) * self.opt.lambda_feat
            # Denoised image
            feat_weights = 4.0 / self.opt.n_layers_D
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake_dn[i]) - 1):
                    loss_G_GAN_Feat_dn += D_weights * feat_weights * \
                                          self.criterionFeat(pred_fake_dn[i][j],
                                                             pred_real_dn[i][
                                                                 j].detach()) * self.opt.lambda_feat
        return loss_G_GAN_Feat, loss_G_GAN_Feat_dn

    def calculate_NCE_loss(self, src, tgt, input_is_denoised=False, mask=None):

        # Only use the encoder part of the networks
        # the encoder learns to pay attention to the commonalities between the
        # two domains, such as object parts and shapes, while being invariant
        # to the differences, such as the textures of the animals.
        feat_q = self.netG(tgt, encode_only=True, input_is_denoised=input_is_denoised)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, encode_only=True, input_is_denoised=input_is_denoised)

        #  ....add the two layered MLP network that projects both input and
        #  output patch to a shared embedding space. For each layer’s features, we sample 256
        # random locations, and apply the 2-layer MLP (Net F) to acquire 256-dim final features
        if input_is_denoised:
            feat_k_pool, sample_ids = self.netF_DN(feat_k, self.opt.num_patches, None, mask=mask)
            feat_q_pool, _ = self.netF_DN(feat_q, self.opt.num_patches, sample_ids)
        else:
            feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None, mask=mask)
            feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0

        # Go through the feature layerss
        for f_q, f_k in zip(feat_q_pool, feat_k_pool):
            # Calculate patchloss in crit, f_q is from the transformed image and f_k is from the original one
            loss = self.criterionNCE(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / len(feat_q_pool)

    def create_figure(self, writer: SummaryWriter, global_step: int = 0, save_gif=False) -> plt.Figure:
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
            img2tensorboard.add_animated_gif(writer=writer, scale_factor=256, tag="GAN/IDT B", max_out=85,
                                             image_tensor=((self.idt_B * 0.5) + 0.5).squeeze(dim=0).cpu().detach().numpy(),
                                             global_step=global_step)

        axs, fig = tensorboard.init_figure(3, 7)
        tensorboard.set_axs_attribute(axs)
        tensorboard.fill_subplots(self.real_A.cpu(), axs=axs[0, :], img_name='A')
        tensorboard.fill_subplots(self.fake_B.detach().cpu(), axs=axs[1, :], img_name='fake')
        overlay = util.create_overlaid_tensor(self.fake_B.detach() * 0.5 + 0.5, self.mask_A)
        tensorboard.fill_subplots(overlay.detach().cpu(), axs=axs[2, :], img_name='mask overlaid on fake')
        tensorboard.fill_subplots(self.fake_B_dn.cpu(), axs=axs[3, :], img_name='fake denoised')
        tensorboard.fill_subplots(self.real_B.cpu(), axs=axs[4, :], img_name='B')
        tensorboard.fill_subplots(self.idt_B.cpu(), axs=axs[5, :], img_name='idt_B')
        tensorboard.fill_subplots(self.idt_B_dn.cpu(), axs=axs[6, :], img_name='idt_B denoised')
        return fig

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

        if self.opt.nce_idt and self.isTrain:
            n_c = int(self.real_A.shape[2] / 2)
            self.idt_B_center_sag = self.idt_B[:, :, n_c, ...]

            n_c = int(self.real_A.shape[3] / 2)
            self.idt_B_center_cor = self.idt_B[..., n_c, :]

            n_c = int(self.real_A.shape[4] / 2)
            self.idt_B_center_axi = self.idt_B[..., n_c]
