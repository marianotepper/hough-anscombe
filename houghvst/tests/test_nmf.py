import colorcet as cc
import glob
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import os
import peakutils
from skimage.transform import downscale_local_mean
import sklearn.decomposition as sk_dec
from sklearn.utils.linear_assignment_ import linear_assignment
import seaborn.apionly as sns
import tifffile
import houghvst.estimation.gat as gat
from houghvst.estimation.utils import half_sample_mode


def read_movie(filename, crop, downscale):
    if os.path.isdir(filename):
        movie = [tifffile.imread(fn) for fn in glob.iglob(filename + '/*.tif*')]
        print(movie[0].shape)
        movie = np.stack(movie, axis=0)
        print(movie.shape)
    else:
        movie = tifffile.imread(filename + '.tif')
    if crop is not None:
        movie = movie[:, crop[0]:crop[2], crop[1]:crop[3]]
    if downscale is not None:
        movie = downscale_local_mean(movie, (downscale, 1, 1))[:-1, :, :]
    print(movie.shape)
    return movie.astype(np.float)


def compute_nmf(movie, n_components=10):
    Y = movie.reshape((movie.shape[0], -1)).T

    minY = Y.min()
    if minY < 0:
        Y -= Y.min()

    model = sk_dec.NMF(n_components=n_components, random_state=0)
    A = model.fit_transform(Y)
    C = model.components_
    return A, C


def correct_components_plain_NMF(A, C):
    max_vals = A.max(axis=0)
    A /= max_vals[np.newaxis, :]
    C *= max_vals[:, np.newaxis]
    return A, C


def correct_components_VST_NMF(A, C, sigma_sq, alpha):
    A_inv = np.zeros_like(A)
    C_inv = np.zeros_like(C)
    for k in range(A.shape[1]):
        comp_r1 = np.outer(A[:, k], C[k, :])
        comp_r1_inv = gat.compute_inverse_gat(comp_r1, sigma_sq, alpha=alpha,
                                              method='asym')
        print(comp_r1_inv.min(), comp_r1_inv.max())

        model = sk_dec.NMF(n_components=1, random_state=0)
        A_inv[:, k] = np.squeeze(model.fit_transform(comp_r1_inv - comp_r1_inv.min()))
        C_inv[k, :] = np.squeeze(model.components_)

    max_vals = A_inv.max(axis=0)
    A_inv /= max_vals[np.newaxis, :]
    C_inv *= max_vals[:, np.newaxis]

    return A_inv, C_inv


def process(params):
    movie = read_movie(params['file'], params['crop'], params['downscale'])
    movie_gat = gat.compute_gat(movie, params['sigma_sq'],
                                alpha=params['alpha'])

    n_components = params['n_components']

    A, C = compute_nmf(movie, n_components=n_components)
    A, C = correct_components_plain_NMF(A, C)
    A_gat, C_gat = compute_nmf(movie_gat, n_components=n_components)
    A_gat_inv, C_gat_inv = correct_components_VST_NMF(A_gat, C_gat,
                                                      params['sigma_sq'],
                                                      params['alpha'])

    mat = A.T.dot(A_gat_inv)
    mat /= np.linalg.norm(A, axis=0)[:, np.newaxis]
    mat /= np.linalg.norm(A_gat_inv, axis=0)[np.newaxis, :]
    mat = mat.max() - mat

    la_idx = linear_assignment(mat)
    A_gat_inv = A_gat_inv[:, la_idx[:, 1]]
    C_gat_inv = C_gat_inv[la_idx[:, 1], :]

    n_rows = int(np.ceil(n_components / 10))
    n_cols = n_components // n_rows

    fig = plt.figure(figsize=(12, 6))
    grid = ImageGrid(fig, (0.05, 0.05, 0.9, 0.9),
                     nrows_ncols=(n_rows, n_cols), direction="row",
                     axes_pad=0.02, add_all=True, share_all=True)

    for i in range(n_components):
        ax = grid[i]
        img = A[:, i].reshape(movie.shape[1:])
        ax.imshow(img, cc.m_fire)
        ax.axis('off')

    plt.savefig(params['file'] + '_comp_all_plain.pdf')

    fig = plt.figure(figsize=(12, 6))
    grid = ImageGrid(fig, (0.05, 0.05, 0.9, 0.9),
                     nrows_ncols=(n_rows, n_cols), direction="row",
                     axes_pad=0.02, add_all=True, share_all=True)

    for i in range(n_components):
        ax = grid[i]
        img = A_gat_inv[:, i].reshape(movie_gat.shape[1:])
        ax.imshow(img, cc.m_fire)
        ax.axis('off')

        plt.savefig(params['file'] + '_comp_all_vst.pdf')

    with sns.axes_style('white'):
        for i in range(n_components):
            print('Component {}'.format(i))

            trace = C[i, :]
            hsm = half_sample_mode(trace)
            trace = (trace - hsm) / (trace.max() - hsm)
            trace_gat = C_gat_inv[i, :]
            trace_gat = (trace_gat - trace_gat.min()) / (trace_gat.max()
                                                         - trace_gat.min())
            fig, ax = plt.subplots(1, 1)
            ax.plot(trace_gat, label='VST+NMF', alpha=0.7, zorder=1)
            ax.plot(trace, label='NMF', alpha=0.7, zorder=0)

            peaks = peakutils.indexes(trace, thres=0,
                                      min_dist=int(0.1 * len(movie)))
            peaks_inner = np.argsort(trace[peaks])[-5:]
            peaks = np.sort(peaks[peaks_inner])

            img_locs = np.linspace(0.1 * len(movie), 0.9 * len(movie),
                                   num=len(peaks))

            for k, p in enumerate(peaks):
                imagebox = OffsetImage(movie[p], zoom=0.7, cmap='viridis')
                imagebox.image.axes = ax
                ab = AnnotationBbox(imagebox,
                                    (p, np.maximum(trace[p], trace_gat[p])),
                                    xybox=(img_locs[k], 1.2),
                                    xycoords='data',
                                    boxcoords='data',
                                    pad=0., arrowprops=dict(arrowstyle="->"))
                ax.add_artist(ab)

            imagebox = OffsetImage(A_gat_inv[:, i].reshape(movie.shape[1:]),
                                   cmap=cc.m_fire, zoom=0.7)
            imagebox.image.axes = ax
            ab = AnnotationBbox(imagebox, (1.15, 0.54),
                                xybox=(1.15, 0.54),
                                xycoords='axes fraction',
                                boxcoords='axes fraction',
                                pad=0.)
            ax.add_artist(ab)

            imagebox = OffsetImage(A[:, i].reshape(movie.shape[1:]),
                                   cmap=cc.m_fire, zoom=0.7)
            imagebox.image.axes = ax
            ab = AnnotationBbox(imagebox, (1.15, 0.33),
                                xybox=(1.15, 0.33),
                                xycoords='axes fraction',
                                boxcoords='axes fraction',
                                pad=0.)
            ax.add_artist(ab)

            ylim = ax.get_ylim()
            ax.set_ylim(ylim[0], 1.35)
            plt.legend(labelspacing=6, bbox_to_anchor=(1.3, 0.7))
            fig.tight_layout(rect=(0, 0, 0.78, 1))
            plt.savefig(params['file'] + '_comp{}.pdf'.format(i))


if __name__ == '__main__':
    dn = '../../images/'
    all_params = [
        dict(file=dn + 'demoMovie',
             crop=None,
             downscale=None,
             alpha=287.49402878208724,
             sigma_sq=93594.22992884477,
             n_components=40),
        dict(file=dn + 'neurofinder.00.00/images',
             # crop=(128, 128, 192, 192),
             crop=(300, 64, 364, 128),
             downscale=None,
             alpha=202.76579576072038,
             sigma_sq=30260.257162256297,
             n_components=10),
    ]

    for params in all_params:
        process(params)

    plt.show()
