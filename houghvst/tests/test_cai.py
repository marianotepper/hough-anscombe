import colorcet as cc
import glob
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage
import seaborn.apionly as sns
import sklearn.decomposition as sk_dec
import sklearn.metrics as sk_metrics
import tifffile
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy.io import loadmat

import houghvst.estimation.gat as gat
from houghvst.utils.deconvolution import deconvolve_black_box
from houghvst.utils.tools import linear_assignment, switch_component_order,\
    detrend_and_normalize


def read_movie(filename, crop, spatial_downscale, use_motion_corrected=True):
    if use_motion_corrected:
        suffix = '/motion_corrected.tif'
    else:
        suffix = '/cell*.tif*'
    movie = [tifffile.imread(fn) for fn in glob.iglob(filename + suffix)]
    movie = np.concatenate(movie, axis=0)
    print(movie.shape, movie.min(), movie.max())

    if crop is not None:
        movie = movie[:, crop[0]:crop[2], crop[1]:crop[3]]

    if 'Akerboom2012' in filename and not use_motion_corrected:
        movie = movie[::2].astype(np.float)

    if spatial_downscale is not None:
        sigma = 2 / (6 * spatial_downscale)
        # sigma = (1 - spatial_downscale) / 2
        movie = ndimage.gaussian_filter1d(movie, sigma, axis=1)
        movie = ndimage.gaussian_filter1d(movie, sigma, axis=2)
        movie = ndimage.zoom(movie, (1, spatial_downscale, spatial_downscale))

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
    normalization_vals = np.linalg.norm(A, axis=0)
    A /= normalization_vals[np.newaxis, :]
    C *= normalization_vals[:, np.newaxis]
    return A, C


def correct_components_VST_NMF(A, C, sigma_sq, alpha):
    # A_inv = A.copy()
    # C_inv = C.copy()
    A_inv = np.zeros_like(A)
    C_inv = np.zeros_like(C)
    for k in range(A.shape[1]):
        # A_inv[:, k] = gat.compute_inverse_gat(A[:, k], sigma_sq, alpha=alpha,
        #                                       method='asym')
        # C_inv[k, :] = gat.compute_inverse_gat(C[k, :], sigma_sq, alpha=alpha,
        #                                       method='asym')

        comp_r1 = np.outer(A[:, k], C[k, :])
        comp_r1_inv = gat.compute_inverse_gat(comp_r1, sigma_sq, alpha=alpha,
                                              method='asym')
        # print(comp_r1_inv.min(), comp_r1_inv.max())
        comp_r1_inv -= comp_r1_inv.min()

        model = sk_dec.NMF(n_components=1, random_state=0)
        A_inv[:, k] = np.squeeze(model.fit_transform(comp_r1_inv))
        C_inv[k, :] = np.squeeze(model.components_)

    normalization_vals = A_inv.max(axis=0)
    A_inv /= normalization_vals[np.newaxis, :]
    C_inv *= normalization_vals[:, np.newaxis]
    return A_inv, C_inv


def plot_spatial_components(A, movie_shape, filename):
    n_components = A.shape[1]

    n_rows = int(np.ceil(n_components / 10))
    n_cols = n_components // n_rows

    fig = plt.figure(figsize=(12, 6))
    grid = ImageGrid(fig, (0.05, 0.05, 0.9, 0.9),
                     nrows_ncols=(n_rows, n_cols), direction="row",
                     axes_pad=0.02, add_all=True, share_all=True)

    for i in range(n_components):
        ax = grid[i]
        img = A[:, i].reshape(movie_shape[1:])
        ax.imshow(img, cc.m_fire)
        ax.axis('off')

    plt.savefig(filename)


class SpikeData:
    def __init__(self, trace, trace_denoised, spikes, score, prec_rec_curve):
        self.trace = trace
        self.trace_denoised = trace_denoised
        self.spikes = spikes
        self.score = score
        self.prec_rec_curve = prec_rec_curve


def fit_spike_threshold(t_frame, gt_spike_time, trace, detrend_scale):
    best_score = 0
    for quantile in range(0, 21, 1):
        trace = detrend_and_normalize(trace, detrend_scale, quantile=quantile)
        trace_denoised, spikes = deconvolve_black_box(trace)

        locs = np.searchsorted(t_frame, gt_spike_time)
        locs[locs == len(t_frame)] = len(t_frame) - 1

        y = np.zeros_like(t_frame)
        y[locs] = 1

        threshold_set = 10. ** np.arange(-6, 0.01, 0.01)
        bin_spikes = [spikes > thresh for thresh in threshold_set]
        scores = [sk_metrics.matthews_corrcoef(y, b) for b in bin_spikes]
        idx = np.argmax(scores)

        if scores[idx] > best_score:
            best_score = scores[idx]
            print('Quantile', quantile, 'Best score', best_score)
            spike_data = SpikeData(trace, trace_denoised, bin_spikes[idx],
                                   scores[idx], None)

    spike_data.prec_rec_curve = sk_metrics.precision_recall_curve(y, spikes)
    return spike_data


def process(config):
    print(config['file'])
    detrend_scale = config['detrend_scale']
    sigma_sq = config['sigma_sq']
    alpha = config['alpha']

    gt_data = loadmat(config['gtfile'])
    gt_data = gt_data['obj']['timeSeriesArrayHash'][0][0][0][0][2]
    t_frame = np.squeeze(gt_data[0][0][0][0][3])
    gt_detected_spikes = np.squeeze(gt_data[0][4][0][0][1]).astype(np.bool)
    t_ephys = np.squeeze(gt_data[0][3][0][0][3])
    gt_spike_time = t_ephys[gt_detected_spikes]

    n_components = config['n_components']

    movie = read_movie(config['file'], config['crop'], config['downscale'])
    movie_shape = movie.shape
    movie_gat = gat.compute_gat(movie - movie.mean(), sigma_sq, alpha=alpha)

    A, C = compute_nmf(movie, n_components=n_components)
    A, C = correct_components_plain_NMF(A, C)

    A_gat, C_gat = compute_nmf(movie_gat, n_components=n_components)
    A_gat_inv, C_gat_inv = correct_components_VST_NMF(A_gat, C_gat,
                                                      sigma_sq, alpha)
    # A_gat_inv, C_gat_inv = correct_components_plain_NMF(A_gat, C_gat)

    la_idx = linear_assignment(A, A_gat_inv)
    A_gat_inv, C_gat_inv = switch_component_order(A_gat_inv, C_gat_inv, la_idx)

    np.savez('test.npz', A=A, C=C, A_gat_inv=A_gat_inv, C_gat_inv=C_gat_inv,
             movie_shape=movie_shape)

    data_npz = np.load('test.npz')
    A = data_npz['A']
    C = data_npz['C']
    A_gat_inv = data_npz['A_gat_inv']
    C_gat_inv = data_npz['C_gat_inv']
    movie_shape = data_npz['movie_shape']

    plot_spatial_components(A, movie_shape,
                            config['file'] + '_comp_all_NMF.pdf')
    plot_spatial_components(A_gat_inv, movie_shape,
                            config['file'] + '_comp_all_VST+NMF.pdf')

    # return

    sns.set_style(style='white')

    dic_spike_data = {}
    for temp_comp, method in zip([C, C_gat_inv], ['NMF', 'VST+NMF']):
        # plot component 1 (0 is the background)
        print('\t{}\tComponent 1'.format(method))

        if 'show_comp' in config:
            idx_comp = config['show_comp']
        else:
            idx_comp = -1

        crude_trace = temp_comp[idx_comp, :]
        spike_data = fit_spike_threshold(t_frame, gt_spike_time, crude_trace,
                                         detrend_scale)
        dic_spike_data[method] = spike_data
        print('\tF1 score: {}'.format(spike_data.score))

        for lw in [None, 0.5]:
            fig, ax = plt.subplots(1, 1, figsize=(10, 3))
            ax.plot(t_frame, spike_data.trace, linewidth=lw, label=method)
            ax.plot(t_frame, spike_data.trace_denoised, linewidth=lw,
                    label='Denoised')

            ax.plot(gt_spike_time,
                    np.ones_like(gt_spike_time) * -0.08,
                    linestyle='none', marker='|', color='k',
                    label='GT Spikes')
            ax.plot(t_frame[spike_data.spikes],
                    spike_data.spikes[spike_data.spikes] * -0.035,
                    linestyle='none', marker='|', color='#e41a1c',
                    label='Spikes')

            plt.legend(numpoints=5, bbox_to_anchor=(1.17, 0.7))
            fig.tight_layout(rect=(0, 0, 0.85, 1))
            suffix = '_{}_comp1'.format(method)
            if lw is not None:
                suffix += '_fine'
            suffix += '.pdf'
            plt.savefig(config['file'] + suffix)

    plt.figure()
    for method in ['NMF', 'VST+NMF']:
        curve = dic_spike_data[method].prec_rec_curve
        plt.plot(curve[1], curve[0], label=method)
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        plt.legend()


def akerboom_configurations():
    # Some files are commented as the number of frames in the tif file
    # does not match the number of frames in the mat file.
    dn = '../../images/cai-1/GCaMP5k_9cells_Akerboom2012/'

    all_configurations = [
        # dict(file=dn + 'raw_data/20110714_cell1',
        #      gtfile=dn + 'processed_data/data_071411_cell1_005.mat',
        #      crop=(0, 1, 19, 32),
        #      downscale=None,
        #      detrend_scale=0.1,
        #      alpha=86.95090392805471,
        #      sigma_sq=8982.002316160868,
        #      n_components=2),
        dict(file=dn + 'raw_data/20110727_cell2',
             gtfile=dn + 'processed_data/data_072711_cell2_002.mat',
             crop=(0, 1, 19, 30),
             downscale=0.1,
             detrend_scale=None,
             alpha=155.15533014748925,
             sigma_sq=15126.323372061219,
             n_components=2),
        # # dict(file=dn + 'raw_data/20110803_cell2',
        # #      gtfile=dn + 'processed_data/data_080311_cell2_001.mat',
        # #      crop=None,
        # #      downscale=None,
        # #      alpha=59.5853590570023,
        # #      sigma_sq=11119.180837770971,
        # #      n_components=2),
        # # dict(file=dn + 'raw_data/20110805_cell7',
        # #      gtfile=dn + 'processed_data/data_080511_cell7_002.mat',
        # #      crop=None,
        # #      downscale=None,
        # #      alpha=141.30216559138532,
        # #      sigma_sq=10592.671682502383,
        # #      n_components=2),
        # dict(file=dn + 'raw_data/20110805_cell12',
        #      gtfile=dn + 'processed_data/data_080511_cell12_002.mat',
        #      crop=None,
        #      downscale=None,
        #      detrend_scale=0.2,
        #      alpha=11.456571088741311,
        #      sigma_sq=16577.164983194525,
        #      n_components=2),
        # dict(file=dn + 'raw_data/20110826_cell1',
        #      gtfile=dn + 'processed_data/data_082611_cell1_002.mat',
        #      crop=None,
        #      downscale=None,
        #      detrend_scale=0.1,
        #      alpha=155.6601984017587,
        #      sigma_sq=8148.439865447937,
        #      n_components=2),
        # # dict(file=dn + 'raw_data/20110826_cell2',
        # #      gtfile=dn + 'processed_data/data_071411_cell1_005.mat',
        # #      crop=None,
        # #      downscale=None,
        # #      alpha=122.34553934898572,
        # #      sigma_sq=17008.93955492347,
        # #      n_components=2),
        # # dict(file=dn + 'raw_data/20110901_cell1',
        # #      gtfile=dn + 'processed_data/data_082611_cell2_001.mat',
        # #      crop=None,
        # #      downscale=None,
        # #      alpha=116.61861766862225,
        # #      sigma_sq=8488.796397174288,
        # #      n_components=2),
        # dict(file=dn + 'raw_data/20110907_cell4',
        #      gtfile=dn + 'processed_data/data_090711_cell4003.mat',
        #      crop=(0, 0, 19, 32),
        #      downscale=None,
        #      detrend_scale=0.1,
        #      alpha=76.77286749443172,
        #      sigma_sq=31092.645880180826,
        #      n_components=2),
    ]
    return all_configurations


def chen_configurations():
    # Some files are commented as the number of frames in the tif file
    # does not match the number of frames in the mat file.
    dn = '../../images/cai-1/GCaMP6f_11cells_Chen2013/'

    all_configurations = [
        # dict(file=dn + 'raw_data/20120502_cell1/001',
        #      gtfile=dn + 'processed_data/data_20120502_cell1_001.mat',
        #      crop=(65, 65, 193, 193),
        #      downscale=0.25,
        #      detrend_scale=0.25,
        #      alpha=313.7514382025155,
        #      sigma_sq=7993.614815720497,
        #      n_components=2),
        # dict(file=dn + 'raw_data/20120502_cell1/002',
        #      gtfile=dn + 'processed_data/data_20120502_cell1_002.mat',
        #      crop=(80, 55, 190, 160),
        #      downscale=None,
        #      detrend_scale=None,
        #      alpha=346.1340250496927,
        #      sigma_sq=10379.221299614164,
        #      n_components=1),
        # dict(file=dn + 'raw_data/20120502_cell3/001',
        #      gtfile=dn + 'processed_data/data_20120502_cell3_001.mat',
        #      crop=(80, 55, 190, 160),
        #      downscale=None,
        #      detrend_scale=None,
        #      alpha=346.1340250496927,
        #      sigma_sq=10379.221299614164,
        #      n_components=1),
        dict(file=dn + 'raw_data/20120521_cell1/001',
             gtfile=dn + 'processed_data/data_20120521_cell1_001.mat',
             crop=(85, 70, 195, 160),
             downscale=1,
             detrend_scale=0.3,
             alpha=429.43153652712374,
             sigma_sq=91999.70469939584,
             n_components=2,
             show_comp=0),
    ]
    return all_configurations


def main():
    all_configurations = akerboom_configurations()
    # all_configurations = chen_configurations()

    for config in all_configurations:
        process(config)

    plt.show()

if __name__ == '__main__':
    main()