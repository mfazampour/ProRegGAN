from typing import List

import torch
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import gridspec

WRITE_NEXT_ORDER = False

def set_axs_attribute(axs):
    for ax in list(axs.flatten()):
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

def format_with_order(a,b):
    # return r'${}_{{e^{{{}}}}}$'.format(a, b)
    return r'$\underset{{\times \ e({})}}{{{}}}$'.format(b, a)
    # return r'${}_{{e{}}}$'.format(a, b)

def fmt(x, pos):
    """
    Format color bar labels to show scientific label
    """
    global WRITE_NEXT_ORDER
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    # return r'${} \ e^{{{}}}$'.format(a, b)
    if pos % 3 == 0:
        if b != 0:
            WRITE_NEXT_ORDER = False
            return format_with_order(a, b)
        else:
            WRITE_NEXT_ORDER = True
            return r'${}$'.format(a)
    elif WRITE_NEXT_ORDER:
        WRITE_NEXT_ORDER = False
        return format_with_order(a, b)
    else:
        return r'${}$'.format(a)


def fill_subplots(img: torch.Tensor, axs: List[plt.Axes], img_name='', fontsize=10, cmap='gray',
                  fig: plt.Figure=None, show_colorbar=False, normalize=True, permute=True):
    if cmap == 'gray' and normalize:  # map image to 0...1
        img = (img - img.min())/(img.max() - img.min())
    elif cmap is None:  # cliping data to 0...255
        img[img < 0] = 0
        if img.dtype == torch.int:
            img[img > 255] = 255
        else:
            img[img > 1] = 1

    if permute:
        img = img.permute((*range(0, len(img.shape)-3), -1, -2, -3)).flip([-3])
    shape = img.shape[-3:]
    img0 = axs[0].imshow(img[0, :, int(shape[0] / 2), :, :].permute(dims=(1, 2, 0)).squeeze().numpy(), cmap=cmap)
    # axs[0].set_title(f'{img_name} central slice \n in sagittal view', fontsize=fontsize)
    axs[0].set_ylabel(f'{img_name}', fontsize=fontsize)
    axs[0].yaxis.set_visible(True)
    if show_colorbar:
        axs[0].tick_params(axis='y', which='both', left=False, right=False, labelleft=True, pad=20)
    else:
        axs[0].tick_params(axis='y', which='both', left=False, right=False, labelleft=True)
    axs[0].set_yticklabels([])
    img1 = axs[1].imshow(img[0, :, :, int(shape[1] / 2), :].permute(dims=(1, 2, 0)).squeeze().numpy(), cmap=cmap)
    # axs[1].set_title(f'{img_name} central slice \n in coronal view', fontsize=fontsize)
    img2 = axs[2].imshow(img[0, :, :, :, int(shape[2] / 2)].permute(dims=(1, 2, 0)).squeeze().numpy(), cmap=cmap)
    # axs[2].set_title(f'{img_name} central slice \n in axial view', fontsize=fontsize)
    if show_colorbar and fig is not None:
        set_colorbar(img0, axs[0], fig, fontsize)
        set_colorbar(img1, axs[1], fig, fontsize)
        set_colorbar(img2, axs[2], fig, fontsize)


def set_colorbar(img, ax: plt.Axes, fig: plt.Figure, fontsize):
    cb = fig.colorbar(img, ax=ax, orientation='horizontal', format=ticker.FuncFormatter(fmt), pad=0.2)
    cb.ax.tick_params(labelsize=fontsize)
    cb.ax.locator_params(axis='x', nbins=4)


def create_figure(fixed: torch.Tensor, moving: torch.Tensor, registered: torch.Tensor, jacob: torch.Tensor,
                  deformation: torch.Tensor, log_sigma: torch.Tensor = None, mean: torch.Tensor = None) -> plt.Figure:
    nrow = 9
    if log_sigma is not None:
        nrow += 3
    if mean is not None:
        nrow += 3
    ncol = 3
    axs, fig = init_figure(ncol, nrow)

    # fig, axs = plt.subplots(5, 3)
    set_axs_attribute(axs)
    fill_subplots(fixed, axs=axs[0, :], img_name='Fixed')
    fill_subplots(moving, axs=axs[1, :], img_name='Moving')
    fill_subplots(fixed - moving, axs=axs[2, :], img_name='Fix-Mov')
    fill_subplots(registered, axs=axs[3, :], img_name='Registered')
    fill_subplots(fixed - registered, axs=axs[4, :], img_name='Fix-Reg')
    fill_subplots(deformation[:, 0:1, ...], axs=axs[5, ...], img_name='Def. X', cmap='RdBu', fig=fig, show_colorbar=True)
    fill_subplots(deformation[:, 1:2, ...], axs=axs[6, ...], img_name='Def. Y', cmap='RdBu', fig=fig, show_colorbar=True)
    fill_subplots(deformation[:, 2:3, ...], axs=axs[7, ...], img_name='Def. Z', cmap='RdBu', fig=fig, show_colorbar=True)
    fill_subplots(jacob, axs=axs[8, :], img_name='Det. Jacob.', cmap='RdBu', fig=fig, show_colorbar=True)
    idx = 8
    if log_sigma is not None:
        fill_subplots(log_sigma[:, 0:1, ...], axs=axs[idx + 1, ...], img_name='LogSigma X', cmap='RdBu', fig=fig, show_colorbar=True)
        fill_subplots(log_sigma[:, 1:2, ...], axs=axs[idx + 2, ...], img_name='LogSigma Y', cmap='RdBu', fig=fig, show_colorbar=True)
        fill_subplots(log_sigma[:, 2:3, ...], axs=axs[idx + 3, ...], img_name='LogSigma Z', cmap='RdBu', fig=fig, show_colorbar=True)
        idx += 3
    if mean is not None:
        fill_subplots(mean[:, 0:1, ...], axs=axs[idx + 1, ...], img_name='mean X', cmap='RdBu', fig=fig, show_colorbar=True)
        fill_subplots(mean[:, 1:2, ...], axs=axs[idx + 2, ...], img_name='mean Y', cmap='RdBu', fig=fig, show_colorbar=True)
        fill_subplots(mean[:, 2:3, ...], axs=axs[idx + 3, ...], img_name='mean Z', cmap='RdBu', fig=fig, show_colorbar=True)
    return fig

def create_seg_figure(fixed: torch.Tensor, moving: torch.Tensor, registered: torch.Tensor) -> plt.Figure:
    nrow = 5
    ncol = 3
    axs, fig = init_figure(ncol, nrow)

    # fig, axs = plt.subplots(5, 3)
    set_axs_attribute(axs)
    fill_subplots(fixed, axs=axs[0, :], img_name='Fixed', cmap='jet')
    fill_subplots(moving, axs=axs[1, :], img_name='Moving', cmap='jet')
    fill_subplots((fixed - moving).abs() * fixed, axs=axs[2, :], img_name='Fix-Mov', cmap='jet')
    fill_subplots(registered, axs=axs[3, :], img_name='Registered', cmap='jet')
    fill_subplots((fixed - registered).abs() * fixed, axs=axs[4, :], img_name='Fix-Reg', cmap='jet')
    return fig


def init_figure(ncol, nrow) -> (List[plt.Axes], plt.Figure):
    fig = plt.figure(figsize=(2 * ncol + 1, 2 * nrow + 1))  # , constrained_layout=True)
    spec = gridspec.GridSpec(nrow, ncol, figure=fig,
                             wspace=0.2, hspace=0.2,
                             top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                             left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))
    # spec = fig.add_gridspec(ncols=3, nrows=5, width_ratios=[0.5]*3, height_ratios=[1]*5)
    # spec.update(wspace=0.025, hspace=0.05)
    axs = []
    for i in range(nrow):
        tmp = []
        for j in range(ncol):
            ax = fig.add_subplot(spec[i, j])
            tmp.append(ax)
        axs.append(tmp)
    axs = np.asarray(axs)
    return axs, fig


def normalize_intensity(img: torch.Tensor):
    img_ = (img - img.min())/(img.max() - img.min() + 1e-5)
    return img_

def image_over_image(img1: torch.Tensor, img2: torch.Tensor):
    overlay = normalize_intensity(img1.detach()).repeat(1, 3, 1, 1, 1)
    overlay *= 0.8
    overlay[:, 0:1, ...] += (normalize_intensity(img2.detach())) * 0.8
    overlay[overlay > 1] = 1
    return overlay


def mask_over_image(mask: torch.Tensor, img: torch.Tensor, one_hot=False):
    overlay = normalize_intensity(img.detach()).repeat(1, 3, 1, 1, 1)
    if one_hot:
        mask_ = mask
    else:
        mask_ = torch.argmax(mask, dim=1, keepdim=True)
    overlay[:, 0:1, ...] += 0.5 * mask_.detach()
    overlay *= 0.8
    overlay[overlay > 1] = 1
    # overlay = mask.repeat(1, 3, 1, 1, 1)
    # overlay[:, 0:1, ...] = img.detach()
    # overlay[:, 2, ...] = 0
    return overlay


def mask_over_mask(mask1: torch.Tensor, mask2: torch.Tensor):
    overlay = mask1.repeat(1, 3, 1, 1, 1)
    overlay[:, 0:1, ...] = mask2.detach()
    overlay[:, 2, ...] = 0
    return overlay
