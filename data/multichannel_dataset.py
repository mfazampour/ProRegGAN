import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
from scipy.io import loadmat, savemat
import numpy as np
import torch

def load(pth: str) -> np.ndarray:
    """

    Returns
    -------
    Multi channel image data as a numpy object
    """
    return loadmat(pth)['data']


class MultiChannelDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert (self.opt.load_size >= self.opt.crop_size)  # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.input_nc
        self.output_nc = self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        # path should be a .mat file
        AB = load(AB_path)
        # AB = Image.open(AB_path).convert('RGB')
        # split AB image into A and B
        h, w, c = AB.shape
        w2 = int(w / 2)
        A = AB[:, :w2, :]
        B = AB[:, w2:, :]

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, (A.shape[1], A.shape[0]))
        A_transform = get_transform(self.opt, transform_params, grayscale=True)
        B_transform = get_transform(self.opt, transform_params, grayscale=True)

        list_A = []
        list_B = []
        for i in range(A.shape[2]):
            a = A[..., i] * 256
            a = Image.fromarray(a)
            a = A_transform(a)
            list_A.append(a)

            b = B[..., i] * 256
            b = Image.fromarray(b)
            b = B_transform(b)
            list_B.append(b)
        A = torch.cat(list_A, dim=0)
        B = torch.cat(list_B, dim=0)

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
