from typing import List

import torch
import torch.nn.functional as F
import numpy as np
import rising
from rising.transforms.functional import affine as rising_affine

try:
    import napari
except:
    print("failed to load napari")

from util import se3


def set_border_value(img: torch.Tensor, value=None):
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
    """

    Parameters
    ----------
    img
    transform
    device

    Returns
    -------
    transformed image
    """
    x_trans = []
    for i in range(img.shape[0]):
        grid = F.affine_grid(transform[i:i+1, :3, :], img[i:i+1, ...].shape, align_corners=False).to(device)
        x_trans += [F.grid_sample(img[i:i+1, ...], grid, mode='nearest', padding_mode='border', align_corners=False)]
    # x_trans = torch.tensor(x_trans.view(1,9,256,256))
    return torch.stack(x_trans, dim=0).squeeze(dim=1)


def create_random_affine(n, img_shape=torch.tensor([128.0, 128.0, 128.0]), dtype=torch.float,
                         device=torch.device('cpu')):
    """
    creates a random rotation (in axis-angle presentation) and translation and returns the affine matrix, and the 6D pose
    Parameters
    ----------
    img_shape : need to normalize the translation to the img_shape since F.affine_grid will expect it like that
    n : batch size
    dtype
    device

    Returns
    -------
    affine
    vector
    """
    rotation = (2 * torch.rand((n, 3), dtype=dtype) - 1) * 0.4
    translation = (2 * torch.rand((n, 3), dtype=dtype) - 1) * 0.2
    vector = torch.cat((rotation, translation), dim=1)
    affines = torch.zeros((n, 4, 4), dtype=dtype)
    for i in range(n):
        affine = se3.vector_to_matrix(vector[i, :])
        affines[i, ...] = affine.clone()
    # vector[:, -3:] *= torch.tensor(img_shape, dtype=translation.dtype, device=translation.device)
    return affines.to(device), vector.to(device)


def tensor_vector_to_matrix(t: torch.Tensor):
    affines = torch.zeros((t.shape[0], 4, 4), dtype=t.dtype)
    for i in range(t.shape[0]):
        affine = se3.vector_to_matrix(t[i, :].cpu())
        affines[i, ...] = affine.clone()
    return affines.to(t.device)


def tensor_matrix_to_vector(t: torch.Tensor):
    vectors = torch.zeros((t.shape[0], 6), dtype=t.dtype)
    for i in range(t.shape[0]):
        vector = se3.matrix_to_vector(t[i, :].cpu())
        vectors[i, ...] = vector
    return vectors.to(t.device)


def show_volumes(img_list: List[torch.Tensor]):
    img_list_np = []
    for t in img_list:
        img_list_np.append(t[0, ...].cpu().squeeze().numpy())
    try:
        with napari.gui_qt():
            napari.view_image(np.stack(img_list_np))
    except:
        print("failed to load napari")


def apply_random_affine(img: torch.Tensor, affine: torch.Tensor = None, rotation=10, translation=10, batchsize=1, interp_mode='bilinear'):
    rotation = (2 * np.random.rand(3) - 1) * rotation
    translation = (2 * np.random.rand(3) - 1) * translation
    if affine is None:
        affine = rising_affine.parametrize_matrix(scale=1, rotation=rotation, translation=translation, ndim=3,
                                                  batchsize=batchsize, device=img.device)
    img = rising_affine.affine_image_transform(img, affine, interpolation_mode=interp_mode, padding_mode='border')
    return img, affine
