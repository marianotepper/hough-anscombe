import glob
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import os
import peakutils
import sklearn.decomposition as sk_dec
from sklearn.utils.linear_assignment_ import linear_assignment
import seaborn.apionly as sns
import tifffile
import houghvst.estimation.gat as gat


def read_movie(filename, crop):
    if os.path.isdir(filename):
        movie = [tifffile.imread(fn) for fn in glob.iglob(filename + '/*.tif*')]
        print(movie[0].shape)
        movie = np.stack(movie, axis=0)
        print(movie.shape)
    else:
        movie = tifffile.imread(filename + '.tif')
    if crop is not None:
        movie = movie[:, crop[0]:crop[2], crop[1]:crop[3]]
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
    max_vals = A.max(axis=0)
    A /= max_vals[np.newaxis, :]
    C *= max_vals[:, np.newaxis]
    return A, C


def process(params):
    movie = read_movie(params['file'], params['crop'])
    movie_gat = gat.compute_gat(movie, params['sigma_sq'],
                                alpha=params['alpha'])

    # plt.figure()
    # plt.plot(movie_gat[:, 54, 29] / movie_gat[:, 54, 29].max())
    # plt.plot(movie[:, 54, 29] / movie[:, 54, 29].max())
    #
    # print(np.var(movie[:, 54, 29], ddof=1))
    # print(np.var(movie_gat[:, 54, 29], ddof=1))
    #
    # plt.figure()
    # plt.plot(movie_gat[:, 22, 25] / movie_gat[:, 22, 25].max())
    # plt.plot(movie[:, 22, 25] / movie[:, 22, 25].max())
    #
    # print(np.var(movie[:, 22, 25], ddof=1))
    # print(np.var(movie_gat[:, 22, 25], ddof=1))
    #
    # plt.figure()
    # plt.plot(movie_gat[:, 26, 28] / movie_gat[:, 26, 28].max())
    # plt.plot(movie[:, 26, 28] / movie[:, 26, 28].max())
    #
    # print(np.var(movie[:, 26, 28], ddof=1))
    # print(np.var(movie_gat[:, 26, 28], ddof=1))
    #
    # return

    n_components = params['n_components']

    A, C = compute_nmf(movie, n_components=n_components)
    A_gat, C_gat = compute_nmf(movie_gat, n_components=n_components)

    mat = A.T.dot(A_gat)
    mat /= np.linalg.norm(A, axis=0)[:, np.newaxis]
    mat /= np.linalg.norm(A_gat, axis=0)[np.newaxis, :]
    mat = mat.max() - mat

    la_idx = linear_assignment(mat)
    A_gat = A_gat[:, la_idx[:, 1]]
    C_gat = C_gat[la_idx[:, 1], :]

    row_factor = int(np.ceil(n_components / 10))
    n_rows = 2 * row_factor
    n_cols = n_components // row_factor
    fig = plt.figure()
    grid = ImageGrid(fig, (0.05, 0.05, 0.9, 0.9),
                     nrows_ncols=(n_rows, n_cols), direction="row",
                     axes_pad=0.02, add_all=True, share_all=True)

    for i in range(n_components):
        ax = grid[i]
        img = A[:, i].reshape(movie.shape[1:])
        ax.imshow(img)
        ax.axis('off')

    for i in range(n_components):
        ax = grid[i + n_components]
        img = A_gat[:, i].reshape(movie_gat.shape[1:])
        ax.imshow(img)
        ax.axis('off')

    plt.savefig(params['file'] + '_comp_all.pdf')

    C_gat_inv = gat.compute_inverse_gat(C_gat, params['sigma_sq'],
                                        alpha=params['alpha'], method='asym')

    with sns.axes_style('white'):
        for i in range(n_components):
            print('Component {}'.format(i))

            trace = C[i, :]
            trace = (trace - trace.min()) / (trace.max() - trace.min())
            trace_gat = C_gat_inv[i, :]
            trace_gat = (trace_gat - trace_gat.min()) / (trace_gat.max()
                                                         - trace_gat.min())
            fig, ax = plt.subplots(1, 1)
            ax.plot(trace_gat, label='VST+NMF', alpha=0.7, zorder=1)
            ax.plot(trace, label='NMF', alpha=0.7, zorder=0)

            peaks = peakutils.indexes(trace, thres=0, min_dist=200)
            peaks_inner = np.argsort(trace[peaks])[-5:]
            peaks = np.sort(peaks[peaks_inner])

            img_locs = np.linspace(200, len(movie) - 200, num=len(peaks))
            print(img_locs)
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

            imagebox = OffsetImage(A_gat[:, i].reshape(movie.shape[1:]),
                                   zoom=0.7)
            imagebox.image.axes = ax
            ab = AnnotationBbox(imagebox, (1.15, 0.54),
                                xybox=(1.15, 0.54),
                                xycoords='axes fraction',
                                boxcoords='axes fraction',
                                pad=0.)
            ax.add_artist(ab)

            imagebox = OffsetImage(A[:, i].reshape(movie.shape[1:]), zoom=0.7)
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
             alpha=287.49402878208724,
             sigma_sq=93594.22992884477,
             n_components=30),
        # dict(file=dn + 'neurofinder.00.00/images',
        #      # crop=(128, 128, 192, 192),
        #      crop=(300, 64, 364, 128),
        #      alpha=202.76579576072038,
        #      sigma_sq=30260.257162256297,
        #      n_components=10),
    ]

    for params in all_params:
        process(params)

    plt.show()
