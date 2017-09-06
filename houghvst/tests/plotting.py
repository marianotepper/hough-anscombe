import colorcet as cc
from matplotlib.colors import colorConverter, LinearSegmentedColormap
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
from scipy.ndimage import generic_filter1d
import seaborn.apionly as sns
from skimage.segmentation import find_boundaries
import warnings


class Cut:
    def __init__(self, orientation, idx, color):
        self.orientation = orientation
        self.idx = idx
        self.color = color


class PlotPatch:
    def __init__(self, box, color):
        self.box = box
        self.color = color


def transversal_cuts(img, cuts, cmap='gray', normalize=False, axes=None):
    if normalize:
        vmin, vmax = img.min(), img.max()
    else:
        vmin, vmax = 0, 255

    if axes is None:
        ax = plt.subplot(121)
    else:
        ax = axes[0]
    ax.imshow(img, vmin=vmin, vmax=vmax, cmap=cmap)
    for c in cuts:
        if c.orientation == 'v':
            lines = ax.plot([c.idx, c.idx], [0, img.shape[0] - 0.5])
        if c.orientation == 'h':
            lines = ax.plot([0, img.shape[1] - 0.5], [c.idx, c.idx])
        plt.setp(lines, color=c.color)
        plt.setp(lines, linewidth=1)
        ax.axis('off')

    with sns.axes_style('whitegrid'):
        if axes is None:
            ax = plt.subplot(122)
        else:
            ax = axes[1]

        def test(x):
            return (x * 0.5).sum()

        for c in cuts:
            if c.orientation == 'v':
                curve = img[:, c.idx]
            if c.orientation == 'h':
                curve = img[c.idx, :]
            ax.plot(curve, color=c.color, alpha=0.5)


def plot_patches_overlay(img, patches, selection=[], cmap='gray',
                         normalize=False):
    if normalize:
        vmin, vmax = img.min(), img.max()
    else:
        vmin, vmax = 0, 255

    if len(selection) == 0:
        subplot_idx = 121
    else:
        subplot_idx = 111
    fig = plt.gcf()
    grid = ImageGrid(fig, subplot_idx,
                      nrows_ncols=(1, 1),
                      direction="row",
                      axes_pad=0.05,
                      add_all=True,
                      share_all=True)
    grid[0].imshow(img, vmin=vmin, vmax=vmax, cmap=cmap)
    grid[0].axis('off')
    for p in patches:
        if p.color == 'w':
            zorder = 1
        else:
            zorder = 2
        rect = mpatches.Rectangle((p.box[1], p.box[0]), p.box[2], p.box[3],
                                  linewidth=2, edgecolor=p.color,
                                  facecolor='none', zorder=zorder)
        grid[0].add_artist(rect)

    if len(selection) == 0:
        nrows = int(np.ceil(np.sqrt(len(selection))))
        ncols = int(np.ceil(np.sqrt(len(selection))))

        grid = ImageGrid(fig, 122,
                          nrows_ncols=(nrows, ncols),
                          direction="row",
                          axes_pad=0.15,
                          add_all=True,
                          share_all=True)

        for ax, idx in zip(grid, selection):
            p = patches[idx]
            crop = img[p.box[0]: p.box[0] + p.box[2], p.box[1]: p.box[1] + p.box[3]]

            plot_patch(crop, edgecolor=p.color, ax=ax, vmin=vmin, vmax=vmax,
                       cmap=cmap)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            plt.tight_layout()


def plot_patch(patch, edgecolor=None, ax=None, vmin=None, vmax=None, cmap=None):
    if ax is None:
        ax = plt.gca()

    im_plot = ax.imshow(patch, vmin=vmin, vmax=vmax, cmap=cmap)

    for loc in ['bottom', 'top', 'left', 'right']:
        ax.spines[loc].set_color(edgecolor)
        ax.spines[loc].set_linewidth(4)
    ax.tick_params(axis='both',
                   which='both',
                   left='off', right='off',
                   top='off', bottom='off',
                   labelleft='off', labelbottom='off')

    return im_plot


def plot_vst_accumulator_space(acc_space, cmap=cc.m_bgy, ax=None):
    if ax is None:
        ax = plt.gca()



    alpha_step0 = acc_space.alpha_range[1] - acc_space.alpha_range[0]
    alpha_step1 = acc_space.alpha_range[-1] - acc_space.alpha_range[-2]
    sigma_step0 = acc_space.sigma_range[1] - acc_space.sigma_range[0]
    sigma_step1 = acc_space.sigma_range[-1] - acc_space.sigma_range[-2]
    im_plt = ax.imshow(acc_space.score, cmap=cmap,
                       extent=(acc_space.alpha_range[0] - alpha_step0 / 2,
                               acc_space.alpha_range[-1] + alpha_step1 / 2,
                               acc_space.sigma_range[-1] + sigma_step1 / 2,
                               acc_space.sigma_range[0] - sigma_step0 / 2)
                       )
    plt.axis('tight')
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$\sigma$')

    plt.colorbar(im_plt)


def plot_slic_boundaries(img, slic_labels, color='r', normalize=False,
                    ax=None):
    if ax is None:
        ax = plt.gca()
    if normalize:
        vmin, vmax = img.min(), img.max()
    else:
        vmin, vmax = 0, 255

    ax.imshow(img, vmin=vmin, vmax=vmax, cmap='gray')

    img_bounds = find_boundaries(slic_labels)
    print(img.shape, img_bounds.shape)

    color1 = colorConverter.to_rgba('black')
    color2 = colorConverter.to_rgba(color)
    cmap = LinearSegmentedColormap.from_list('my_cmap2', [color1, color2], N=2)
    cmap._init()
    cmap._lut[:, -1] = np.arange(5)

    ax.imshow(img_bounds, cmap=cmap, cax=ax)
