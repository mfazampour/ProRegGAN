import os.path

os.environ["TORCHIO_HIDE_CITATION_PROMPT"] = str(1)

import random
import torch
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import median_filter
import torchio
import nibabel as nib
from nibabel.affines import voxel_sizes
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
from .volume_dataset import VolumeDataset


def normalize_rotation_matrix(affine_matrix):
    # Ensure the matrix is a numpy array
    affine_matrix = np.array(affine_matrix)

    # Extract the upper-left 3x3 submatrix (rotation and scaling)
    rotation_matrix = affine_matrix[:3, :3]

    # Normalize each row to have a norm of 1
    for i in range(3):
        row_norm = np.linalg.norm(rotation_matrix[i])
        if row_norm != 0:
            rotation_matrix[i] /= row_norm

    # Put the normalized rotation matrix back into the affine matrix
    affine_matrix[:3, :3] = rotation_matrix

    return affine_matrix

class MuProRegDataset(VolumeDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.
        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        Returns:
            the modified parser.
        """
        parser = VolumeDataset.modify_commandline_options(parser, is_train)

        return parser

    def read_list_of_patients(self):
        if self.mode == "train":
            dataset_type = "train"
            start_case = 0
            end_case = 63
        elif self.mode == "val" or self.mode == "validation" or self.mode == "test":
            dataset_type = "val"
            start_case = 64
            end_case = 72
        main_folder_path = self.opt.dataroot
        dataset_list = []
        for case_number in range(start_case, end_case + 1):
            case_str = f"case{case_number:06d}"  # Formatting the case number to match the file naming
            mr_image_path = f"{main_folder_path}/{dataset_type}/mr_images/{case_str}_reg.nii.gz"
            mr_label_path = f"{main_folder_path}/{dataset_type}/mr_labels/{case_str}_reg.nii.gz"
            us_image_path = f"{main_folder_path}/{dataset_type}/us_images/{case_str}_reg.nii.gz"
            us_label_path = f"{main_folder_path}/{dataset_type}/us_labels/{case_str}_reg.nii.gz"
            # check if all paths exist, if not, skip this case
            if not os.path.exists(mr_image_path) or not os.path.exists(mr_label_path) or not os.path.exists(us_image_path) or not os.path.exists(us_label_path):
                continue
            dataset_list.append((mr_image_path, mr_label_path, us_image_path, us_label_path))
        return dataset_list

    def load_subject_(self, index):
        sample = self.patients[index % len(self.patients)]

        mr_path = sample[0]
        mr_label_path = sample[1]
        trus_path = sample[2]
        trus_label_path = sample[3]

        # load mr and turs file if it hasn't already been loaded
        if sample not in self.subjects:
            # print(f'loading patient {sample}')
            subject = torchio.Subject(mr=self.create_scalar_image_from_nib(mr_path), trus=self.create_scalar_image_from_nib(trus_path))
            subject.add_image(torchio.ScalarImage(tensor=subject['trus'].data, affine=subject['trus'].affine), 'trus_unfiltered')
            if self.load_mask:
                subject = self.create_labelmaps_from_nii(mr_label_path, subject, 'mr_tree', 'mr_landmarks')
                # subject.add_image(torchio.LabelMap(mr_label_path), 'mr_tree')
                if os.path.isfile(trus_label_path):
                    subject = self.create_labelmaps_from_nii(trus_label_path, subject, 'trus_tree', 'trus_landmarks')
                if 'trus_tree' in subject.keys():
                    if subject['trus_tree'].data.sum() == 0:  # if the trus tree is empty, remove it
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
            self.subjects[sample] = subject
        subject = self.subjects[sample]
        return sample, subject

    def load_landmarks(self, sample, subject):
        # landmarks are already loaded in the subject
        return subject

    def set_sample_path(self, dict_, sample):
        # this is the format of sample[0]: '/home/farid/Downloads/muProReg//train/mr_images/case000012_reg.nii.gz'
        # extract the case number from the path
        dict_['Patient'] = sample[0].split('/')[-1].split('_')[0].replace('case', '')
        dict_['A_paths'] = sample[0]
        dict_['B_paths'] = sample[2]

    def create_labelmaps_from_nii(self, file_path, subject, label_name, landmark_name):
        # Read the image file
        nii_image = nib.load(file_path)
        image_data = nii_image.get_fdata()
        image_data = np.transpose(image_data, (1, 2, 0, 3))  # to match the in-house data format
        affine = nii_image.affine
        affine = normalize_rotation_matrix(affine)

        # create an empty tensor to hold the landmark data
        landmark_tensor = torch.zeros(image_data.shape[:3], dtype=torch.float32)

        # Assuming the last dimension is the number of sub-images
        for i in range(image_data.shape[-1]):
            sub_image_data = image_data[..., i]
            if i == 0:
                # Add an extra dimension to make the tensor 4D
                sub_image_tensor = torch.tensor(sub_image_data, dtype=torch.float32).unsqueeze(0)
                # Create a LabelMap for each sub-image
                labelmap = torchio.LabelMap(tensor=sub_image_tensor, affine=affine)
                subject.add_image(labelmap, label_name)
            else:
                # where the sub-image is not zero, set the corresponding landmark to i
                landmark_tensor[sub_image_data != 0] = i

        # Create a LabelMap for the landmarks
        landmark_labelmap = torchio.LabelMap(tensor=landmark_tensor.unsqueeze(0), affine=affine)

        # Add the landmark LabelMap to the subject
        subject.add_image(landmark_labelmap, landmark_name)

        return subject

    def create_scalar_image_from_nib(self, file_path):
        # Read the image file using nibabel
        nii_image = nib.load(file_path)
        image_data = nii_image.get_fdata()
        image_data = np.transpose(image_data, (1, 2, 0))  # to match the in-house data format
        affine = nii_image.affine
        affine = normalize_rotation_matrix(affine)

        # Convert the numpy array to a PyTorch tensor
        tensor = torch.tensor(image_data, dtype=torch.float32)

        # Check if the tensor is 4D, if not, add a channel dimension
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)

        # Create and return a torchio.ScalarImage
        return torchio.ScalarImage(tensor=tensor, affine=affine)

    def name(self):
        return 'muProRegDataset'
