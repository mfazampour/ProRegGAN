import torch
from .base_model import BaseModel
from . import networks
from . import networks3d
import os
from util import affine_transform
from typing import Tuple
from util import distance_landmarks
from collections import OrderedDict

import torch
import math
import torchvision
import transforms3d
import torchio
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


def set_border_value(img: torch.Tensor, value= None):
    if value is None:
        value = img.min()
    img[:, :, 0, :, :] = value
    img[:, :, -1, :, :] = value
    img[:, :, :, 0, :] = value
    img[:, :, :, -1, :] = value
    img[:, :, :, :, 0] = value
    img[:, :, :, :, -1] = value
    return img

def transform_image(img: torch.Tensor, transform, device):
    # img = torch.tensor(img.view(1,1,9,256,256)).to(self.device)
    img = img.unsqueeze(dim=0)
    # img = set_border_value(img)
    grid = F.affine_grid(transform, img.shape).to(device)
    x_trans = F.grid_sample(img, grid, padding_mode='border')
    # x_trans = torch.tensor(x_trans.view(1,9,256,256))
    return x_trans.squeeze(dim=0)


class RegModel(BaseModel):
    """ This class implements the GAN registration model, for learning 6 parameters from moving images for registration
    with fixed images given paired data.

    GAN paper: https://arxiv.org/pdf/1804.11024.pdf
    """

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
        # changing the default values to match the GAN paper (https://arxiv.org/pdf/1804.11024.pdf)
        parser.set_defaults(norm='batch', netG='NormalNet', dataset_mode='multichannel', output_nc=1, netD='basic',
                            input_nc=9)

       # parser.add_argument('--netReg', default="NormalNet", help='regnet')


        if is_train:
            print("HERE AGAIN")

            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--reg_idt_B', action='store_true', help='use idt_B from CUT model instead of real B')
            parser.add_argument('--lambda_Reg', type=float, default=0.5, help='weight for the registration loss')
            parser.add_argument('--lr_Reg', type=float, default=0.0001, help='learning rate for the reg. network opt.')
            parser.add_argument('--show_volumes', type=bool, default=False, help='visualize transformed volumes w napari')


        print(parser)

        return parser

    def __init__(self, opt):
        """Initialize the reg class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        # default `log_dir` is "runs" - we'll be more specific here
        self.writer = SummaryWriter()
        self.device = torch.device("cpu")
        self.isTrain = opt.isTrain
        self.set_visdom_names()
        self.set_networks(opt)


        BaseModel.__init__(self, opt)
        self.isTrain = opt.isTrain        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        # square of difference between the transformation parameters as part of the generator loss
     #   self.loss_names = ['loss_G', 'D_real', 'D_fake','loss_D']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
      #  self.visual_names = ['real_A', 'real_A_inverted', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:

            # WGAN values from paper
            self.learning_rate = 0.002
            self.batch_size = 64
            self.weight_cliping_limit = 0.01

            self.criterionRigidReg = networks.RegistrationLoss()
            self.optimizer_RigidReg = torch.optim.Adam(self.netG.parameters(), lr=opt.lr_Reg,
                                                       betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_RigidReg)

            self.criterionL1 = torch.nn.L1Loss()
            self.criterionGAN = networks3d.GANLoss(opt.gan_mode).to(self.device)
            self.optimizer_D = torch.optim.RMSprop(self.netD.parameters(), lr=self.learning_rate)
            self.optimizers.append(self.optimizer_D)



            self.loss_landmarks = 0
        #    self.criterionRigidReg = networks.RegistrationLoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
        #    D_params = list(self.netD.parameters())
        #    G_params = list(self.netG.parameters())
        #    self.optimizer_D = torch.optim.RMSprop(D_params, lr=self.learning_rate)
        #    self.optimizer_G = torch.optim.RMSprop(G_params, lr=self.learning_rate)

       #     self.optimizers.append(self.optimizer_G)
       #     self.optimizers.append(self.optimizer_D)


    def get_model(self, name):

        if name == "G":
            return self.netG
        if name == 'D':
            return self.netD

    def set_networks(self, opt):
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
                self.model_names = ['G']

        print(f"modeltype: {opt.netG}")
        self.netG = networks3d.define_reg_model(model_type=opt.netG, n_input_channels=2,num_classes=6, gpu_ids=opt.gpu_ids, img_shape=opt.inshape)
        self.netD = networks3d.define_D(input_nc=opt.input_nc + opt.output_nc, ndf= opt.ndf, netD=opt.netD,
                                      n_layers_D=opt.n_layers_D, norm=opt.norm, init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=opt.gpu_ids)

        # extract shape from sampled input
        inshape = opt.inshape
        # device handling
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in opt.gpu_ids])


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
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        # AtoB = self.opt.direction == 'AtoB'
        self.clean_tensors()
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.patient = input['Patient']
        self.landmarks_A = input['A_landmark'].to(self.device).unsqueeze(dim=0)
        self.landmarks_B = input['B_landmark'].to(self.device).unsqueeze(dim=0)

        affine, self.gt_vector = affine_transform.create_random_affine(self.real_B.shape[0],
                                                                       self.real_B.shape[-3:],
                                                                       self.real_B.dtype,
                                                                       device=self.real_B.device)

        self.deformed_B = affine_transform.transform_image(self.real_B, affine, self.real_B.device)
        self.deformed_LM_B = affine_transform.transform_image(self.landmarks_B, affine, self.landmarks_B.device)

        if self.opt.show_volumes:
            affine_transform.show_volumes([self.real_B, self.deformed_B])

        self.mask_A = input['A_mask'].to(self.device).type(self.real_A.dtype)
        if input['B_mask_available'][0]:  # TODO in this way it only works with batch size 1
            self.mask_B = input['B_mask'].to(self.device).type(self.real_A.dtype)
            self.transformed_maskB = affine_transform.transform_image(self.mask_B, affine, self.mask_B.device)
        else:
            self.mask_B = None

        ###
        self.init_loss_tensors()


    def init_loss_tensors(self):

        self.loss_D_fake = torch.tensor([0.0])
        self.loss_G = torch.tensor([0.0])
        self.loss_D_real = torch.tensor([0.0])
        self.loss_D = torch.tensor([0.0])

      #  self.loss_RigidReg = torch.tensor([0.0])
      #  self.diff_dice= torch.tensor([0.0])
        self.loss_landmarks_diff = torch.tensor([0.0])
        self.loss_landmarks = torch.tensor([0.0])

    def set_visdom_names(self):

        # rigid registration
        self.loss_names = ['G', 'D_real', 'D_fake', 'D', 'landmarks_diff','landmarks']
        self.visual_names = ['diff_center_sag', 'diff_center_cor', 'diff_center_axi']
        self.visual_names += ['diff_orig_center_sag', 'diff_orig_center_cor', 'diff_orig_center_axi']
        self.visual_names += ['deformed_center_sag', 'deformed_center_cor', 'deformed_center_axi']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.reg_params = self.netG(torch.cat([self.real_B, self.deformed_B], dim=1))
        self.transformed_Image = self.get_transformed_images()
        self.compute_landmark_loss()

        #output_matrix = self.create_matrix(output_g)
        #print("shape")
        #print(output_matrix.shape)
        #output_matrix = output_matrix.view(1, 3, 4)
        #self.mri_inv_output_transf = transform_image(self.mri_random_deformed, output_matrix, self.device)
        #img_grid = torchvision.utils.make_grid(self.mri_inv_output_transf[:, 0:2, :, :])
        #self.writer.add_image('train_img_correct', img_grid)
    # def create_matrix(self, output):
    #     print(output[0][3], output[0][4], output[0][5])
    #     rot_matr = torchgeometry.angle_axis_to_rotation_matrix((output[0][0],
    #                                             output[0][1],
    #                                             output[0][2]))
    #     new_rot_matr = torch.cat(((rot_matr), ([output[0][3]], [output[0][4]], [output[0][5]])),1)
    #     return new_rot_matr


    def get_transformed_images(self):
        # Transform back the deformed B
        reg_Image = affine_transform.transform_image(self.real_B,
                                                     affine_transform.tensor_vector_to_matrix(self.reg_params.detach()),
                                                     device=self.real_B.device)

        return reg_Image

    def compute_landmark_loss(self):

        # Calc landmark difference from original landmarks and deformed

        loss_landmarks_beginning = distance_landmarks.get_distance_lmark(self.deformed_LM_B,self.landmarks_B,
                                                                         self.landmarks_B.device)

        # Add affine transform from G to the original landmarks to match deformed
        self.LM_affine = affine_transform.transform_image(self.landmarks_B,
                                                          affine_transform.tensor_vector_to_matrix(
                                                                     self.reg_params.detach()),
                                                          device=self.deformed_LM_B.device)


        # Calc landmark difference from deformed landmarks and the affine(Deformed)
        self.loss_landmarks = distance_landmarks.get_distance_lmark(self.deformed_LM_B, self.LM_affine,
                                                                    self.LM_affine.device)

        self.loss_landmarks = loss_landmarks_beginning - self.loss_landmarks

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""

        # Randomize if the input to D is the real or fake tranform

        pred_fake = self.netD(torch.cat((self.deformed_B, self.transformed_Image),1))  # TODO add ultrasound image as the input to the network
        # this should be pretty low (since the output of the G is used to transform the image)

       # self.mri_inv_original_transf = self.transform_image(self.mri_random_deformed, self.original_transf_matrix_inv[:3, :].view(1, 3, 4))
        pred_real = self.netD(torch.cat((self.deformed_B,self.deformed_B),1))
        # this should be a high value since we are using the GT transform

        self.loss_D_fake = self.criterionGAN(pred_fake, target_is_real=False)
        self.loss_D_real = self.criterionGAN(pred_real, target_is_real=True)

        # Critic Loss = [average critic score on real images] – [average critic score on fake images]
        self.loss_D = self.loss_D_fake + self.loss_D_real
     #   self.writer.add_scalar("Loss/loss_D", self.loss_D)
        self.loss_D.backward()



    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""

        # Generator Loss = -[average critic score on fake images]
        #https: // machinelearningmastery.com / how - to - implement - wasserstein - loss - for -generative - adversarial - networks /
        self.loss_G = self.criterionRigidReg(self.reg_params, self.gt_vector) * self.opt.lambda_Reg

        pred_fake = self.netD(torch.cat((self.deformed_B,self.transformed_Image),1))
        self.loss_D_fake = self.criterionGAN(pred_fake, target_is_real=False)
      #  self.writer.add_scalar("Loss/loss_G",  self.loss_D_fake)
        self.loss_G = self.loss_G + self.loss_D_fake

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A)

        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        self.backward_D()  # calculate gradients for D
        self.optimizer_D.step()  # update D's weights
        for p in self.netD.parameters():
            p.data.clamp_(-self.weight_cliping_limit, self.weight_cliping_limit)

        self.train_counter += 1
        if self.train_counter == 2:
            # update G
            self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
            self.optimizer_G.zero_grad()  # set G's gradients to zero
            self.backward_G()  # calculate graidents for G
            self.optimizer_G.step()  # udpate G's weights
            # reset the counter
            self.train_counter = 0


    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        super().compute_visuals()

        self.reg_image = self.get_transformed_images()

        self.diff_images = self.reg_image - self.deformed_B
        self.diff_orig = self.real_B - self.deformed_B

        n_c = self.real_B.shape[2]

        self.reg_center_sag = self.reg_image[:, :, int(n_c / 2), ...]
        self.diff_center_sag = self.diff_images[:, :, int(n_c / 2), ...]
        self.diff_orig_center_sag = self.diff_orig[:, :, int(n_c / 2), ...]
        self.deformed_center_sag = self.deformed_B[:, :, int(n_c / 2), ...]

        n_c = self.real_B.shape[3]
        self.reg_center_cor = self.reg_image[:, :, :, int(n_c / 2), ...]
        self.diff_center_cor = self.diff_images[:, :, :, int(n_c / 2), ...]
        self.diff_orig_center_cor = self.diff_orig[:, :, :, int(n_c / 2), ...]
        self.deformed_center_cor = self.deformed_B[:, :, :, int(n_c / 2), ...]

        n_c = self.real_B.shape[4]
        self.reg_center_axi = self.reg_image[..., int(n_c / 2)]
        self.diff_center_axi = self.diff_images[..., int(n_c / 2)]
        self.diff_orig_center_axi = self.diff_orig[..., int(n_c / 2)]
        self.deformed_center_axi = self.deformed_B[..., int(n_c / 2)]


    def update_learning_rate(self, epoch=0):
        super().update_learning_rate(epoch=epoch)
        if epoch >= self.opt.epochs_before_reg:
            self.first_phase_coeff = 0

    def log_tensorboard(self, writer: SummaryWriter, losses: OrderedDict = None, global_step: int = 0, save_gif=False,
                        use_image_name=False, mode='', epoch=0, save_pdf=False):
        super().log_tensorboard(writer=writer, losses=losses, global_step=global_step, save_gif=save_gif,
                                use_image_name=use_image_name, mode=mode, epoch=epoch)
        return {}


