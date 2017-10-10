import colorcet as cc
import glob
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import os
import peakutils
from scipy.io import loadmat
from skimage.transform import downscale_local_mean
import sklearn.decomposition as sk_dec
import sklearn.metrics as sk_metrics
from sklearn.utils.linear_assignment_ import linear_assignment
import seaborn.apionly as sns
import tifffile
import houghvst.estimation.gat as gat
from houghvst.estimation.utils import half_sample_mode
from houghvst.oasis.functions import deconvolve


def read_movie(filename, downscale):
    if os.path.isdir(filename):
        movie = [tifffile.imread(fn) for fn in glob.iglob(filename + '/*.tif*')]
        movie = movie[0]
    else:
        movie = tifffile.imread(filename + '.tif')
    movie = movie[::2]
    if downscale is not None:
        movie = downscale_local_mean(movie, (downscale, 1, 1))[:-1, :, :]
    print(movie.shape, movie.min(), movie.max())
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
    # A_inv = A.copy()
    # C_inv = C.copy()
    A_inv = np.zeros_like(A)
    C_inv = np.zeros_like(C)
    for k in range(A.shape[1]):
        comp_r1 = np.outer(A[:, k], C[k, :])
        comp_r1_inv = gat.compute_inverse_gat(comp_r1, sigma_sq, alpha=alpha,
                                              method='asym')
        # print(comp_r1_inv.min(), comp_r1_inv.max())

        model = sk_dec.NMF(n_components=1, random_state=0)
        A_inv[:, k] = np.squeeze(model.fit_transform(comp_r1_inv - comp_r1_inv.min()))
        C_inv[k, :] = np.squeeze(model.components_)

    max_vals = A_inv.max(axis=0)
    A_inv /= max_vals[np.newaxis, :]
    C_inv *= max_vals[:, np.newaxis]

    return A_inv, C_inv


def process(params):
    print(params['file'])

    gt_data = loadmat(params['gtfile'])
    gt_data = gt_data['obj']['timeSeriesArrayHash'][0][0][0][0][2]
    t_frame = np.squeeze(gt_data[0][0][0][0][3])
    gt_detected_spikes = np.squeeze(gt_data[0][4][0][0][1]).astype(np.bool)
    t_ephys = np.squeeze(gt_data[0][3][0][0][3])
    gt_spike_time = t_ephys[gt_detected_spikes]
    # fmean_roi = np.squeeze(gt_data[0][0][0][0][1])
    # fmean_neuropil = np.squeeze(gt_data[0][1][0][0][1])
    # fmean_comp = fmean_roi - 0.7 * fmean_neuropil

    movie = read_movie(params['file'], params['downscale'])
    movie_gat = gat.compute_gat(movie, params['sigma_sq'],
                                alpha=params['alpha'])

    n_components = params['n_components']

    A, C = compute_nmf(movie, n_components=n_components)
    A, C = correct_components_plain_NMF(A, C)
    A_gat, C_gat = compute_nmf(movie_gat, n_components=n_components)
    A_gat_inv, C_gat_inv = correct_components_VST_NMF(A_gat, C_gat,
                                                      params['sigma_sq'],
                                                      params['alpha'])

    assignement_cost = A.T.dot(A_gat_inv)
    assignement_cost /= np.linalg.norm(A, axis=0)[:, np.newaxis]
    assignement_cost /= np.linalg.norm(A_gat_inv, axis=0)[np.newaxis, :]
    assignement_cost = assignement_cost.max() - assignement_cost

    la_idx = linear_assignment(assignement_cost)
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

    def fit(spikes):
        locs = np.searchsorted(t_frame, gt_spike_time)
        locs[locs == len(t_frame)] = len(t_frame) - 1

        y = np.zeros_like(t_frame)
        y[locs] = 1

        threshold_set = 10. ** np.arange(-6, 1.5, 0.01)
        bin_spikes = [spikes > thresh for thresh in threshold_set]
        scores = [sk_metrics.f1_score(y, b) for b in bin_spikes]
        idx = np.argmax(scores)

        return bin_spikes[idx], scores[idx]

    for scaling in [90, 100]:
        def adjust_trace(tr):
            hsm = half_sample_mode(tr)
            return (tr - hsm) / (np.percentile(tr, scaling) - hsm)

        with sns.axes_style('white'):
            for temp_comp, method in zip([C, C_gat_inv], ['NMF', 'VST+NMF']):
                # plot component 1 (0 is the background)
                print('\t{}\tComponent 1 (scaling={}%)'.format(method, scaling))

                trace = adjust_trace(temp_comp[1, :])
                trace_dec, trace_spikes = deconvolve(trace)[:2]
                trace_spikes, score = fit(trace_spikes)
                print('\tF1 score: {}'.format(score))

                fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                ax.plot(t_frame, trace, label=method, alpha=0.85,
                        zorder=2)

                ax.plot(t_frame, trace_dec, label='Denoised', alpha=0.85,
                        zorder=3)

                ax.plot(gt_spike_time,
                        np.ones_like(gt_spike_time) * trace.max() * 1.1,
                        linestyle='none', marker='|', color='k',
                        label='GT Spikes')
                ax.plot(t_frame[trace_spikes],
                        trace_spikes[trace_spikes] * trace.max() * 1.08,
                        linestyle='none', marker='|', color='#e41a1c',
                        label='Spikes')

                # ax.set_ylim(trace.min() * 1.05, 1.35)
                plt.legend(numpoints=5, bbox_to_anchor=(1.2, 0.7))
                fig.tight_layout(rect=(0, 0, 0.85, 1))
                plt.savefig(params['file']
                            + '_comp1_scaling{}.pdf'.format(scaling))

if __name__ == '__main__':
    # Some files are commented as the number of frames in the tif file
    # does not match the number of frames in the mat file.
    dn = '../../images/cai-1/GCaMP5k_9cells_Akerboom2012/'
    all_params = [
        dict(file=dn + 'raw_data/20110714_cell1',
             gtfile=dn + 'processed_data/data_071411_cell1_005.mat',
             crop=None,
             downscale=None,
             alpha=86.95090392805471,
             sigma_sq=8982.002316160868,
             n_components=2),
        dict(file=dn + 'raw_data/20110727_cell2',
             gtfile=dn + 'processed_data/data_072711_cell2_002.mat',
             crop=None,
             downscale=None,
             alpha=155.15533014748925,
             sigma_sq=15126.323372061219,
             n_components=2),
        # dict(file=dn + 'raw_data/20110803_cell2',
        #      gtfile=dn + 'processed_data/data_080311_cell2_001.mat',
        #      crop=None,
        #      downscale=None,
        #      alpha=59.5853590570023,
        #      sigma_sq=11119.180837770971,
        #      n_components=2),
        # dict(file=dn + 'raw_data/20110805_cell7',
        #      gtfile=dn + 'processed_data/data_080511_cell7_002.mat',
        #      crop=None,
        #      downscale=None,
        #      alpha=141.30216559138532,
        #      sigma_sq=10592.671682502383,
        #      n_components=2),
        dict(file=dn + 'raw_data/20110805_cell12',
             gtfile=dn + 'processed_data/data_080511_cell12_002.mat',
             crop=None,
             downscale=None,
             alpha=11.456571088741311,
             sigma_sq=16577.164983194525,
             n_components=2),
        dict(file=dn + 'raw_data/20110826_cell1',
             gtfile=dn + 'processed_data/data_082611_cell1_002.mat',
             crop=None,
             downscale=None,
             alpha=155.6601984017587,
             sigma_sq=8148.439865447937,
             n_components=2),
        # dict(file=dn + 'raw_data/20110826_cell2',
        #      gtfile=dn + 'processed_data/data_071411_cell1_005.mat',
        #      crop=None,
        #      downscale=None,
        #      alpha=122.34553934898572,
        #      sigma_sq=17008.93955492347,
        #      n_components=2),
        # dict(file=dn + 'raw_data/20110901_cell1',
        #      gtfile=dn + 'processed_data/data_082611_cell2_001.mat',
        #      crop=None,
        #      downscale=None,
        #      alpha=116.61861766862225,
        #      sigma_sq=8488.796397174288,
        #      n_components=2),
        dict(file=dn + 'raw_data/20110907_cell4',
             gtfile=dn + 'processed_data/data_090711_cell4003.mat',
             crop=None,
             downscale=None,
             alpha=76.77286749443172,
             sigma_sq=31092.645880180826,
             n_components=2),
    ]

    for params in all_params:
        process(params)

    plt.show()