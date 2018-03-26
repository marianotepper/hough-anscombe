from collections import namedtuple
import glob
import h5py
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn.apionly as sns
import tifffile
import timeit
from skimage import draw, measure
from sklearn.neighbors.kde import KernelDensity

import houghvst.estimation.estimation as est
from houghvst.estimation.gat import compute_gat
from houghvst.estimation.regions import im2col
from houghvst.tests.measures import compare_variance_stabilization, \
    compute_temporal_mean_var
from houghvst.tests.plotting import plot_vst_accumulator_space
from houghvst.utils.stats import poisson_gaussian_noise

GroundTruth = namedtuple('GroundTruth', ['movie', 'alpha', 'sigma_sq'])


def load_toy_example():
    movie = np.array([np.repeat(np.arange(50., 562.)[:, np.newaxis],
                                512, axis=1)])
    movie = np.repeat(movie, 100, axis=0)
    for st in zip(np.random.randint(0, 492, size=50),
                  np.random.randint(0, 492, size=50)):
        rr, cc = draw.circle(st[0], st[1], 20)
        movie[:, rr, cc] += 100
    movie /= 5
    movie += 20
    print(movie.min(), movie.max())

    sigma_gt = 30
    alpha_gt = 500
    movie_noisy = poisson_gaussian_noise(movie, sigma_gt, alpha_gt)
    print(movie_noisy.min(), movie_noisy.max())
    # movie_noisy = np.maximum(movie_noisy, 0)
    # movie_noisy = np.minimum(movie_noisy, 8000)
    print(movie_noisy.min(), movie_noisy.max())

    print('PSNR', measure.compare_psnr(alpha_gt * movie, movie_noisy,
                                       data_range=alpha_gt * movie.max()))

    gt_movie_gat = compute_gat(movie_noisy, sigma_gt ** 2, alpha=alpha_gt)
    _, temp_vars = compute_temporal_mean_var(gt_movie_gat)
    print('Temporal variance',
          'MEAN={}, STD={}'.format(temp_vars.mean(),
                                   temp_vars.std(ddof=1)))

    return movie_noisy, GroundTruth(movie, alpha_gt, sigma_gt ** 2)


def load_calcium_examples(select):
    if select == 'Yr_reduced':
        s = '../../images/' \
            'Yr_reduced_0000_d1_463_d2_472_d3_1_order_C_frames_1000_.tif'
        movie = tifffile.imread(s)
        # movie = movie[:, 0:76, 370:].astype(np.float)
        movie = movie.astype(np.float)
    if select == 'k53':
        s = '../../images/k53_20160530_RSM_125um_41mW_zoom2p2_00001_00012.tif'
        movie = tifffile.imread(s)
        # crop dark corners:
        movie = movie[:, 100:-100, 100:-100].astype(np.float)

    elif select == 'demo':
        movie = tifffile.imread('../../images/demoMovie.tif')
        movie = movie.astype(np.float)

    elif select == 'quiet':
        f = h5py.File('../../images/quietBlock.h5_at', 'r')
        movie = np.array(f['quietBlock'], dtype=np.float)[:1000]

    elif 'neurofinder' in select:
        dn = '../../images/' + select + '/images'
        movie = [tifffile.imread(fn) for fn in glob.iglob(dn + '/*.tif*')]
        movie = np.stack(movie, axis=0)
        # crop dark corners:
        movie = movie[:, 100:-100, 100:-100].astype(np.float)

    elif 'cai-1' in select:
        dn = '../../images/' + select
        movie = [tifffile.imread(fn) for fn in glob.iglob(dn + '/*.tif*')]
        movie = np.concatenate(movie, axis=0)
        movie = movie.astype(np.float)

        if 'Chen2013' in select:
            movie = movie[::100]
        if 'Akerboom2012' in select:
            movie = movie[::2]

    print(movie.shape)
    return movie, None


def test_vst_estimation_movie(movie, idx=None, gt=None, filename=None):
    if idx is None:
        movie_train = movie[::200]
    else:
        movie_train = movie[idx]

    block_size = 8
    stride = 8

    t = timeit.default_timer()
    blocks_train = []
    for img in movie_train:
        blocks_train.append(im2col(img, block_size, stride))
    blocks_train = np.vstack(blocks_train)

    sigma_sq_init, alpha_init = est.initial_estimate_sigma_alpha(blocks_train)
    print('\tTime', timeit.default_timer() - t)
    print('\talpha = {}; sigma^2 = {}'.format(alpha_init, sigma_sq_init))

    t = timeit.default_timer()
    res = est.estimate_vst_movie(movie_train, stride=stride)
    print('\tTime', timeit.default_timer() - t)

    blocks = []
    for img in movie:
        blocks.append(im2col(img, block_size, stride))
    blocks = np.vstack(blocks)
    plot_vst_estimation(movie, blocks, sigma_sq_init, alpha_init,
                        res, 0, gt=gt, filename=filename)


def test_vst_estimation_frame(movie, idx=0, gt=None, filename=None):
    img = movie[idx]

    block_size = 8
    stride = 8

    t = timeit.default_timer()
    blocks = im2col(img, block_size, stride)
    sigma_sq_init, alpha_init = est.initial_estimate_sigma_alpha(blocks)
    print('\tTime', timeit.default_timer() - t)
    print('\talpha = {}; sigma^2 = {}'.format(alpha_init, sigma_sq_init))

    t = timeit.default_timer()
    res = est.estimate_vst_image(img, stride=stride)
    print('\tTime', timeit.default_timer() - t)

    plot_vst_estimation(movie, blocks, sigma_sq_init, alpha_init,
                        res, idx, gt=gt, filename=filename)


def set_tick_label_size(axes, size='x-large'):
    for tick_axis in [axes.xaxis, axes.yaxis]:
        for tick in tick_axis.get_major_ticks():
            tick.label.set_fontsize(size)


def plot_vst_estimation(movie, blocks, sigma_sq_init, alpha_init,
                        res, idx, gt=None, filename=None):
    img = movie[idx]
    if gt is not None:
        img_gt = gt.movie[idx]

    means, variances = est.compute_mean_var(blocks)

    if gt is not None:
        movie_gat = compute_gat(movie, sigma_sq_init, alpha=alpha_init)
        _, temp_vars = compute_temporal_mean_var(movie_gat)
        print('---> Temporal variance',
              'MEAN={}, STD={}'.format(temp_vars.mean(),
                                       temp_vars.std(ddof=1)))

        compare_variance_stabilization(img_gt, img, gt.sigma_sq, gt.alpha,
                                       sigma_sq_init, alpha_init)

        gt_movie_gat = compute_gat(movie, res.sigma_sq, alpha=res.alpha)
        _, temp_vars = compute_temporal_mean_var(gt_movie_gat)
        print('---> Temporal variance',
              'MEAN={}, STD={}'.format(temp_vars.mean(),
                                       temp_vars.std(ddof=1)))

        compare_variance_stabilization(img_gt, img, gt.sigma_sq, gt.alpha,
                                       res.sigma_sq, res.alpha)

    line_cmap = ['#377eb8', '#e41a1c']

    with sns.axes_style('white'):
        if gt is None:
            plt.figure(figsize=(24, 5))
            gs = gridspec.GridSpec(1, 5, width_ratios=[1, 3, 3, 3, 3],
                                   left=0.02, right=0.98, wspace=0.3)

            axes0 = plt.subplot(gs[0, 0])
            axes1 = plt.subplot(gs[0, 1])
            axes2 = plt.subplot(gs[0, 2])
            axes3 = plt.subplot(gs[0, 3])
            axes4 = plt.subplot(gs[0, 4])

            axes0.imshow(img, cmap='viridis')
            axes0.axis('off')
            axes0.set_title('Input image', fontsize='xx-large')
        else:
            plt.figure(figsize=(24, 5))
            gs = gridspec.GridSpec(2, 5, width_ratios=[2, 2, 3, 3, 3],
                                   left=0.02, right=0.98, wspace=0.3)

            axes00 = plt.subplot(gs[0, 0])
            axes10 = plt.subplot(gs[1, 0])
            axes1 = plt.subplot(gs[:, 1])
            axes2 = plt.subplot(gs[:, 2])
            axes3 = plt.subplot(gs[:, 3])
            axes4 = plt.subplot(gs[:, 4])

            axes00.imshow(img_gt, cmap='viridis')
            axes00.axis('off')
            axes00.set_title('Noiseless image', fontsize='xx-large')
            axes10.imshow(img, cmap='viridis')
            axes10.axis('off')
            axes10.set_title('Noisy image', fontsize='xx-large')

        scatter_color = '#a6cee3'

        # Using rasterized=True drastically reduces the file size
        axes1.plot(means, variances, '.', alpha=0.2,
                   color=scatter_color, markeredgecolor='none', rasterized=True,
                   label='Patch')

        x = np.array([[means.min()], [means.max()]])
        axes1.plot(x, alpha_init * x + sigma_sq_init, color=line_cmap[0],
                   label='Initial estimation')

        axes1.plot(x, res.alpha * x + res.sigma_sq, color=line_cmap[1],
                   label='Refined estimation')

        if gt is not None:
            axes1.plot(x, gt.alpha * x + gt.sigma_sq, color='k',
                       label='Ground truth')

        set_tick_label_size(axes1)

        axes1.set_xlabel('Mean', fontsize='xx-large')
        axes1.set_ylabel('Variance', fontsize='xx-large')

        # lgnd = axes1.legend(markerscale=5, fontsize='xx-large')
        # for h in lgnd.legendHandles:
        #     h._sizes = [600]

        xdiff = np.percentile(means, 99) - means.min()
        ydiff = np.percentile(variances, 99) - variances.min()
        axes1.set_xlim((means.min() - 0.1 * xdiff,
                        np.percentile(means, 99) + 0.1 * xdiff))
        axes1.set_ylim((variances.min() - 0.1 * ydiff,
                        np.percentile(variances, 99) + 0.1 * ydiff))

        axes1.set_title('Patch mean vs patch variance', fontsize='xx-large')

        # Accumulator space
        plot_vst_accumulator_space(res.acc_space_init, ax=axes2,
                                   plot_focus=True)
        set_tick_label_size(axes2)
        axes2.set_title('Coarse accumulator space', fontsize='xx-large')

        # Zoom-in on the accumulator space
        plot_vst_accumulator_space(res.acc_space, ax=axes3, plot_estimates=True)
        set_tick_label_size(axes3)
        axes3.set_title('Focused accumulator space', fontsize='xx-large')

        # Density plot of block variances
        blocks_gat = compute_gat(blocks, sigma_sq_init, alpha=alpha_init)
        _, variances_init = est.compute_mean_var(blocks_gat)

        blocks_gat = compute_gat(blocks, res.sigma_sq, alpha=res.alpha)
        _, variances = est.compute_mean_var(blocks_gat)

        data_range = (np.minimum(variances_init.min(), variances.min()),
                      np.maximum(variances_init.max(), variances.max()))
        samples = np.linspace(*data_range, num=1000)

        kde = KernelDensity(kernel='gaussian', bandwidth=0.05)
        probas_both = []
        for vs, color, label in zip([variances_init, variances], line_cmap,
                                    ['Initial', 'Refined']):
            kde.fit(vs[:, np.newaxis])
            probas = np.exp(kde.score_samples(samples[:, np.newaxis]))
            probas /= probas.sum()
            probas_both.append(probas)
            axes4.fill_between(samples, probas, color=color, alpha=0.3,
                               label=label + ' estimation')
            loc = np.argmax(probas)
            axes4.plot([samples[loc], samples[loc]], [0, probas[loc]], '-',
                       color=color)
            axes4.plot([1, 1], [0, probas[loc] * 1.05], 'k:')

        probas_both = np.maximum(*probas_both)
        idx_nnz = np.where(probas_both > 1e-4)[0]
        axes4.set_xlim(samples[0], samples[idx_nnz[-1]])

        set_tick_label_size(axes4)

        axes4.legend(fontsize='xx-large', loc='upper right')
        axes4.set_xlabel('Patch variance', fontsize='xx-large')
        axes4.set_title('Patch variance density', fontsize='xx-large')

        if filename is not None:
            if 'cai-1' in filename:
                filename = filename.replace('/raw_data/', '_')
                filename = filename.replace('/', '_')
            plt.savefig(filename + '_estimation.pdf')


def main():
    # movie_noisy, gt = load_toy_example()
    # test_vst_estimation_frame(movie_noisy, idx=0, gt=gt)

    tests = [
        # 'Yr_reduced',
        # 'k53',
        'demo',
        # 'quiet',
        # 'neurofinder.00.00',
        # 'cai-1/GCaMP5k_9cells_Akerboom2012/raw_data/20110714_cell1',
        # 'cai-1/GCaMP5k_9cells_Akerboom2012/raw_data/20110727_cell2',
        # 'cai-1/GCaMP5k_9cells_Akerboom2012/raw_data/20110803_cell2',
        # 'cai-1/GCaMP5k_9cells_Akerboom2012/raw_data/20110805_cell7',
        # 'cai-1/GCaMP5k_9cells_Akerboom2012/raw_data/20110805_cell12',
        # 'cai-1/GCaMP5k_9cells_Akerboom2012/raw_data/20110826_cell1',
        # 'cai-1/GCaMP5k_9cells_Akerboom2012/raw_data/20110826_cell2',
        # 'cai-1/GCaMP5k_9cells_Akerboom2012/raw_data/20110901_cell1',
        # 'cai-1/GCaMP5k_9cells_Akerboom2012/raw_data/20110907_cell4',
        # 'cai-1/GCaMP6f_11cells_Chen2013/raw_data/20120502_cell1/001',
        # 'cai-1/GCaMP6f_11cells_Chen2013/raw_data/20120502_cell1/002',
        # 'cai-1/GCaMP6f_11cells_Chen2013/raw_data/20120502_cell3/001',
        # 'cai-1/GCaMP6f_11cells_Chen2013/raw_data/20120521_cell1/001'
    ]

    for name in tests:
        print(name)
        movie_noisy, gt = load_calcium_examples(name)
        movie_noisy -= movie_noisy.mean()
        # test_vst_estimation_frame(movie_noisy, idx=0, gt=gt, filename=name)
        test_vst_estimation_movie(movie_noisy, gt=gt, filename=name)


if __name__ == '__main__':
    main()
    plt.show()
