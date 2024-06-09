import os.path

os.environ["TORCHIO_HIDE_CITATION_PROMPT"] = str(1)

import random
import torch
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import median_filter
import torchio
from torchio.transforms import (
    RescaleIntensity,
    RandomAffine,
    RandomElasticDeformation,
    Compose,
    OneOf,
    Resample,
    RandomFlip,
    CropOrPad,
    Lambda
)
from skimage.exposure import equalize_adapthist
import csv

from util import create_landmarks
from data.base_dataset import BaseDataset
from util.util import lee_filter_creator
from util.simulate_deformation import simulate_deformation


try:
    import napari
except:
    print("failed to import napari")


def load_image_file(path: str) -> np.ndarray:
    img = sitk.ReadImage(path)
    return img


class VolumeDataset(BaseDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        """
        parser.add_argument('--visualize_volume', type=bool, default=False, help='Set visualize to False. it\'s only '
                                                                                 'used for debugging.')
        parser.add_argument('--load_mask', type=bool, default=False, help='load prostate mask for seg. loss')
        parser.add_argument('--inshape', type=int, nargs='+', default=[80] * 3,
                            help='after cropping shape of input. '
                                 'default is equal to image size. specify if the input can\'t path through UNet')
        parser.add_argument('--origshape', type=int, nargs='+', default=[80] * 3,
                            help='original shape of input images')
        parser.add_argument('--min_size', type=int, default=80, help='minimum length of the axes')
        parser.add_argument('--transforms', nargs='+', default=[],
                            help='list of possible augmentations, currently [flip, affine]')
        parser.add_argument('--denoising', nargs='+', default=[],
                            help='list of possible denoising, currently [median, lee_filter]')
        parser.add_argument('--denoising_size', type=int, default=3, help='size of the denoising filter kernel')
        parser.add_argument('--load_uncropped', action='store_true', help='load the original uncropped TRUS')
        parser.add_argument('--apply_clahe', action='store_true', help='apply CLAHE to the TRUS images')
        parser.add_argument('--augment_data', action='store_true', help='Augment data using deformations and noise')
        parser.add_argument('--mask_B_required', action='store_true', help='If set to true and "load_mask" only loads the data with mask B available')
        parser.add_argument('--deform_data', action='store_true', help='deform data using random elastic deformation to simulate prostate motion')

        parser.add_argument('--load_proreggan_label', action='store_true', help='load proreggan generated label for the TRUS images')

        # parser.add_argument('--replaced_denoised', action='store_true', help='replace B with the denoised version')
        return parser

    def __init__(self, opt, mode=None):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt, mode)
        self.opt = opt

        print(f'dataroot: {self.root}')
        self.load_mask = opt.load_mask

        assert self.opt.direction == 'AtoB' or not self.load_mask, "can't load masks in BtoA mode"

        self.patients = self.read_list_of_patients()
        random.shuffle(self.patients)
        self.subjects = {}
        # self.mr = {}
        # self.trus = {}

        assert (self.opt.load_size >= self.opt.crop_size)  # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

        self.input_size = opt.inshape
        # if input_size is a string, then parse it to a list of integers using spaces as separators
        if isinstance(self.input_size, str):
            self.input_size = [int(x) for x in self.input_size.split()]
        self.min_size = opt.min_size

        self.transform = self.create_transforms()

        self.means = []
        self.std = []

    @staticmethod
    def clip_image(x: torch.Tensor):
        [l, h] = np.quantile(x.cpu().numpy(), [0.02, 0.98])
        x[x < l] = l
        x[x > h] = h
        return x

    @staticmethod
    def median_filter_creator(size: int):
        def median_image(x: torch.Tensor):
            im = median_filter(x.squeeze().cpu().numpy(), size=size)
            im = torch.tensor(im, device=x.device, dtype=x.dtype)
            im = torch.reshape(im, x.shape)
            return im
        return median_image

    def create_transforms(self):
        transforms = []

        # clipping to remove outliers (if any)
        # clip_intensity = Lambda(VolumeDataset.clip_image, types_to_apply=[torchio.INTENSITY])
        # transforms.append(clip_intensity)

        rescale = RescaleIntensity((-1, 1))
        # normalize with mu = 0 and sigma = 1/3 to have data in -1...1 almost
        # ZNormalization()

        # transforms.append(rescale)

        # if self.mode == 'train':
        #     # transforms = [rescale]
        #     if 'affine' in self.opt.transforms:
        #         transforms.append(RandomAffine(translation=5, p=0.8))
        #
        #     if 'flip' in self.opt.transforms:
        #         transforms.append(RandomFlip(axes=(0, 2), p=0.8))

        if (self.mode == "train") & self.opt.augment_data:
            spatial = OneOf(
                {RandomAffine(translation=5): 0.8, RandomElasticDeformation(): 0.2},
                p=0.75,
            )
            transforms += [RandomFlip(axes=(0, 2), p=0.25), spatial]
            # transforms += [RandomBiasField()]

        transforms.append(CropOrPad(self.input_size, padding_mode='minimum'))
        transform = Compose(transforms)

        self.denoising_transform = None
        if len(self.opt.denoising) > 0:
            if 'median' in self.opt.denoising:
                self.denoising_transform = Lambda(VolumeDataset.median_filter_creator(self.opt.denoising_size),
                                                  types_to_apply=[torchio.INTENSITY])
            if 'lee_filter' in self.opt.denoising:
                self.denoising_transform = Lambda(lee_filter_creator(self.opt.denoising_size),
                                                  types_to_apply=[torchio.INTENSITY])

        self.zoom_transform = Compose([Resample(0.5), CropOrPad(self.opt.origshape, padding_mode='minimum')])

        return transform

    def reverse_resample(self, min_value=-1):
        transforms = [Resample(1 / self.ratio)]
        return Compose(transforms + [CropOrPad(self.opt.origshape, padding_mode=min_value)])

    def read_list_of_patients(self):
        csv_path = os.path.join(self.opt.dataroot, 'filtered_images.csv')
        filtered_samples = []
        if os.path.exists(csv_path):
            with open(csv_path, newline='') as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    if i == 0:
                        continue
                    filtered_samples += row
        patients = []
        for root, dirs, files in os.walk(self.root):
            if ('nonrigid' in root) or ('cropped' not in root):
                continue
            if np.array([id.lower() in root.lower() for id in filtered_samples]).any():  # patient data is among the filtered ones
                continue
            if 'trus_cut.mhd' not in files:
                continue
            if self.opt.load_mask and self.opt.mask_B_required:
                if 'trus_tree.mhd' not in files and 'trus_tree_unet.mhd' not in files:
                    continue
            patients.append(root)
        return patients

    def __getitem__(self, index):
        sample, subject = self.load_subject_(index)
        subject = self.load_landmarks(sample, subject)

        # resample all images to the same resolution
        t = Resample(subject['mr'])
        for key in subject.keys():
            if key == 'mr':
                continue
            subject[key] = t(subject[key])

        transformed_ = self.transform(subject)

        # read the second sample using a random index
        if self.opt.load_second_sample:
            index2 = np.random.randint(0, len(self.patients))
            # index2 = np.random.randint(len(self.patients) - 10, len(self.patients))
            sample2, subject2 = self.load_subject_(index2)

            # resample all images to the same resolution
            t = Resample(subject2['mr'])
            for key in subject2.keys():
                if key == 'mr':
                    continue
                subject2[key] = t(subject2[key])

            transformed_2 = self.transform(subject2)
            second_sample_dict = self.create_data_dict(sample2, subject2, transformed_2)
        else:
            second_sample_dict = None

        if self.opt.visualize_volume:
            try:
                with napari.gui_qt():
                    napari.view_image(np.stack([transformed_['mr'].data.squeeze().numpy(),
                                                transformed_['trus'].data.squeeze().numpy()]))
            except:
                pass

        dict_ = self.create_data_dict(sample, subject, transformed_, second_sample_dict)

        return dict_

    def load_landmarks(self, sample, subject):
        landmarks_a = create_landmarks.getLandmarks(sample + "/mr.mhd", sample[:-8] + "/mr_pcd.txt")
        landmarks_b = create_landmarks.getLandmarks(sample + "/trus.mhd", sample[:-8] + "/trus_pcd.txt")

        # create a torchio.labelmap from each landmark, call it 'mr_landmarks' and 'trus_landmarks'
        landmarks_a = torchio.LabelMap(tensor=landmarks_a.unsqueeze(0), affine=subject['mr'].affine)
        landmarks_b = torchio.LabelMap(tensor=landmarks_b.unsqueeze(0), affine=subject['trus'].affine)
        # add the landmarks to the subject
        subject.add_image(landmarks_a, 'mr_landmarks')
        subject.add_image(landmarks_b, 'trus_landmarks')

        return subject

    def create_data_dict(self, sample, subject, transformed_, second_sample_dict=None):
        dict_ = {
            'A': transformed_['mr'].data[:, :self.input_size[0], :self.input_size[1], :self.input_size[2]],
            'B': transformed_['trus'].data[:, :self.input_size[0], :self.input_size[1], :self.input_size[2]],
            'B_unfiltered': transformed_['trus_unfiltered'].data[:, :self.input_size[0], :self.input_size[1], :self.input_size[2]],
            'modality_A': 'MR',
            'modality_B': 'US',
            'affine_orig': subject['trus'].affine
        }
        # read the landmarks from the subject if they exist, otherwise a tensor of zeros
        dict_['A_landmark'] = transformed_['mr_landmarks'].data if 'mr_landmarks' in subject.keys() else torch.zeros_like(dict_['A'])
        dict_['B_landmark'] = transformed_['trus_landmarks'].data if 'trus_landmarks' in subject.keys() else torch.zeros_like(dict_['B'])

        # set data paths
        self.set_sample_path(dict_, sample)

        if self.opt.direction == 'AtoB':
            if self.denoising_transform is not None:
                dict_['B_denoised'] = transformed_['trus_denoised'].data[:, :self.input_size[0], :self.input_size[1],
                                      :self.input_size[2]]
            else:
                dict_['B_denoised'] = torch.clone(dict_['B'])
            if self.load_mask:
                dict_['A_mask'] = transformed_['mr_tree'].data[:, :self.input_size[0], :self.input_size[1],
                                  :self.input_size[2]].type(torch.uint8)
                if 'trus_tree' in transformed_.keys():
                    dict_['B_mask'] = transformed_['trus_tree'].data[:, :self.input_size[0], :self.input_size[1],
                                      :self.input_size[2]].type(torch.uint8)
                    dict_['B_mask_available'] = True
                else:
                    dict_['B_mask'] = torch.zeros_like(dict_['A_mask'])
                    dict_['B_mask_available'] = False
        else:
            dict_['A_mask'] = torch.clone(dict_['A']) * 0  # TODO replace this with actual mask
            dict_['B_mask_available'] = False

        if second_sample_dict is not None:
            # add 'B' and 'B_unfiltered' and 'B_denoised' from second sample to dict with key 'B2'
            for key in ['B', 'B_unfiltered', 'B_denoised']:
                dict_['B2_' + key] = second_sample_dict[key]
            # add 'B_mask' and 'B_mask_available' from second sample to dict with key 'B2'
            for key in ['B_mask', 'B_mask_available']:
                dict_['B2_' + key] = second_sample_dict[key]

        if self.opt.dual_mode:
            trus_zoomed = self.zoom_transform(subject['trus'])
            dict_['B_zoomed'] = trus_zoomed.data[:, :self.input_size[0], :self.input_size[1], :self.input_size[2]]
            if second_sample_dict is not None:
                dict_['B2_zoomed'] = second_sample_dict['B_zoomed']

        return dict_

    def set_sample_path(self, dict_, sample):
        dict_['Patient'] = sample.split('/')[-4].replace(' ', '')
        dict_['A_paths'] = sample + "/mr.mhd"
        dict_['B_paths'] = sample + "/trus_cut.mhd"

    def load_subject_(self, index):
        sample = self.patients[index % len(self.patients)]

        # load mr and turs file if it hasn't already been loaded
        # if sample not in self.subjects:

        mask_path = sample + "/mr_tree_unet.mhd" if os.path.isfile(sample + "/mr_tree_unet.mhd") else sample + "/mr_tree.mhd"

        # print(f'loading patient {sample}')
        trus_path = sample + "/trus.mhd" if self.opt.load_uncropped else sample + "/trus_cut.mhd"
        subject = torchio.Subject(mr=torchio.ScalarImage(sample + "/mr.mhd"),
                                  trus=torchio.ScalarImage(trus_path))
        subject.add_image(torchio.ScalarImage(tensor=subject['trus'].data, affine=subject['trus'].affine), 'trus_unfiltered')
        if self.load_mask:
            # load MR mask
            subject.add_image(torchio.LabelMap(mask_path), 'mr_tree')
            # load TRUS mask
            if self.mode =='train' and self.opt.load_proreggan_label and os.path.isfile(sample + "/trus_tree_proreggan.mhd"):
                subject.add_image(torchio.LabelMap(sample + "/trus_tree_proreggan.mhd"), 'trus_tree')
            elif os.path.isfile(sample + "/trus_tree_unet.mhd"):
                subject.add_image(torchio.LabelMap(sample + "/trus_tree_unet.mhd"), 'trus_tree')
            elif os.path.isfile(sample + "/trus_tree.mhd"):
                subject.add_image(torchio.LabelMap(sample + "/trus_tree.mhd"), 'trus_tree')
            if 'trus_tree' in subject.keys():
                if subject['trus_tree'].data.sum() == 0:
                    subject.remove_image('trus_tree')

        if self.denoising_transform is not None:
            subject.add_image(self.denoising_transform(subject['trus']), 'trus_denoised')
            # subject['trus'] = self.denoising_transform(subject['trus'])
        if self.opt.apply_clahe:
            if self.denoising_transform is not None:
                img = subject['trus_denoised'].data
            else:
                img = subject['trus'].data
            img = (img - img.min()) / (img.max() - img.min())
            img1 = equalize_adapthist(img[0, ...].numpy(), 8)
            img1[img1 < 0.01] = 0.0
            img[0, ...] = torch.tensor(img1)
            subject['trus'].set_data(img)

        rescale = RescaleIntensity((-1, 1))
        subject = rescale(subject)

        if self.mode == "train" and self.opt.deform_data:
            # if higher than a random number, deform the image
            if random.random() > 0.0:
                # get parent folder of the sample
                parent = os.path.dirname(sample)
                subject['mr_parent'] = torchio.ScalarImage(parent + "/mr.mhd")
                deformed_image, deformed_label = simulate_deformation(mask=subject['mr_tree'].data,
                                                                      label_path=mask_path, image_path=parent + "/mr.mhd",
                                                                      image_min=subject['mr_parent'].data.min())
                # todo: for now just replace the original image with the deformed one, also the label, but this should be changed
                data = torch.tensor(sitk.GetArrayFromImage(deformed_image).astype(np.int)).permute(2, 1, 0).unsqueeze(0)
                data[data > subject['mr_parent'].data.max()] = subject['mr_parent'].data.max()
                subject['mr_parent'] = torchio.ScalarImage(tensor=data.float(), affine=subject['mr_parent'].affine)
                t = Resample(subject['mr'])
                subject['mr'] = t(subject['mr_parent'])
                data = torch.tensor(sitk.GetArrayFromImage(deformed_label).astype(np.int)).permute(2, 1, 0).unsqueeze(0)
                subject['mr_tree'] = torchio.LabelMap(tensor=data.to(torch.uint8), affine=subject['mr_tree'].affine)

            # self.subjects[sample] = subject
        # subject = self.subjects[sample]
        return sample, subject

    def __len__(self):
        return len(self.patients)

    def name(self):
        return 'VolumeDataset'
