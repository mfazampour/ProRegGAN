from .reg_model import RegModel
import torch
from skimage import color  # used for lab2rgb
import numpy as np


class MultiChannelModel(RegModel):
    """This is a subclass of Pix2PixModel for multi-channel images (each channel is actually a neighboring slice).

    The model training requires '-dataset_model multichannel' dataset.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        By default, we use 'colorization' dataset for this model.
        See the original pix2pix paper (https://arxiv.org/pdf/1611.07004.pdf) and colorization results (Figure 9 in the paper)
        """
        RegModel.modify_commandline_options(parser, is_train)
        parser.set_defaults(dataset_mode='multichannel')
        if is_train:
            parser.add_argument('--lambda_smoothness', type=float, default=100.0,
                                help='weight for between slices smoothness loss')
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
        # reuse the pix2pix model
        RegModel.__init__(self, opt)
        self.loss_names = ['G', 'D_real', 'D_fake', 'D']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'real_A_inverted', 'real_B']
        if self.isTrain:
            self.smoothnessL1 = torch.nn.L1Loss()

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        n_c = self.original_mri.shape[1]
        # average over channel to get the real and fake image
        self.real_A_center = self.original_mri[:, int(n_c / 2), ...].unsqueeze(1)
        self.real_B_center = self.original_us[:, int(n_c / 2), ...].unsqueeze(1)
        self.fake_B_center = self.mri_inv_original_transf[:, int(n_c / 2), ...].unsqueeze(1)
        self.real_A_avg = torch.mean(self.original_mri, dim=1, keepdim=True)
        self.real_B_avg = torch.mean(self.original_us, dim=1, keepdim=True)
        self.fake_B_avg = torch.mean(self.mri_inv_original_transf, dim=1, keepdim=True)
        self.smoothness_fake = torch.sum(torch.abs(self.mri_inv_original_transf[:, :-1, ...] - self.mri_inv_original_transf[:, 1:, ...]),
                                         dim=1, keepdim=True)

    def smoothness(self, real_A_inverted):
        if self.mri_inv_original_transf.shape[1] == 1:
            return 0
        return self.smoothnessL1(real_A_inverted[:, :-1, ...], real_A_inverted[:, 1:, ...])


