import torch
from plotly.subplots import make_subplots
import plotly
import plotly.express as px


def init_figure(nrow, ncol, row_titles, column_titles, title) -> plotly.graph_objs.Figure:

    fig = make_subplots(rows=nrow, cols=ncol, row_titles=row_titles, column_titles=column_titles, x_title=title)
    return fig

    # fig = plt.figure(figsize=(2 * ncol + 1, 2 * nrow + 1))  # , constrained_layout=True)
    # spec = gridspec.GridSpec(nrow, ncol, figure=fig,
    #                          wspace=0.2, hspace=0.2,
    #                          top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
    #                          left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))
    # # spec = fig.add_gridspec(ncols=3, nrows=5, width_ratios=[0.5]*3, height_ratios=[1]*5)
    # # spec.update(wspace=0.025, hspace=0.05)
    # axs = []
    # for i in range(nrow):
    #     tmp = []
    #     for j in range(ncol):
    #         ax = fig.add_subplot(spec[i, j])
    #         tmp.append(ax)
    #     axs.append(tmp)
    # axs = np.asarray(axs)
    # return axs, fig


def fill_subplots(img: torch.Tensor, fig: plotly.graph_objs.Figure, row: int, cmap='gray',
                  show_colorbar=False, normalize=True, permute=True):
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

    row += 1  # plotly index is not 0 based
    fig.add_trace(px.imshow(img[0, :, int(shape[0] / 2), :, :].permute(dims=(1, 2, 0)).squeeze().numpy()).data[0],
                  row=row, col=1)
    fig.add_trace(px.imshow(img[0, :, :, int(shape[1] / 2), :].permute(dims=(1, 2, 0)).squeeze().numpy()).data[0],
                  row=row, col=2)
    fig.add_trace(px.imshow(img[0, :, :, :, int(shape[2] / 2)].permute(dims=(1, 2, 0)).squeeze().numpy()).data[0],
                  row=row, col=3)
    # axs[0].set_title(f'{img_name} central slice \n in sagittal view', fontsize=fontsize)
    # if show_colorbar:
    #     axs[0].tick_params(axis='y', which='both', left=False, right=False, labelleft=True, pad=20)
    # else:
    #     axs[0].tick_params(axis='y', which='both', left=False, right=False, labelleft=True)
    # axs[0].set_yticklabels([])
    # if show_colorbar and fig is not None:
    #     set_colorbar(img0, axs[0], fig, fontsize)
    #     set_colorbar(img1, axs[1], fig, fontsize)
    #     set_colorbar(img2, axs[2], fig, fontsize)
    pass