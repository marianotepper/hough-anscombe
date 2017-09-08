from collections import namedtuple
import h5py
import matplotlib.pyplot as plt
import numpy as np
from skimage import draw, measure
import tifffile
import timeit
import houghvst.estimation.estimation as est
from houghvst.estimation.utils import poisson_gaussian_noise
from houghvst.tests.measures import compare_variance_stabilization
from houghvst.tests.plotting import plot_vst_accumulator_space


GroundTruth = namedtuple('GroundTruth', ['movie', 'alpha', 'sigma'])


def load_toy_examples():
    np.random.seed(1)

    # movie = np.array([128. + np.zeros((512, 512))])
    # movie[:, :256, :] = 128
    # movie[:, 256:, :] = 512

    movie = np.array([np.repeat(np.arange(0., 512.)[:, np.newaxis],
                                512, axis=1)])
    movie = np.maximum(movie - 100, 0)
    movie /= 100
    for st in zip(np.random.randint(0, 492, size=50),
                  np.random.randint(0, 492, size=50)):
        rr, cc = draw.circle(st[0], st[1], 10)
        movie[:, rr, cc] += 500
    movie /= 100
    movie += 20
    print(movie.min(), movie.max())

    sigma_gt = 30
    alpha_gt = 100
    movie_noisy = poisson_gaussian_noise(movie, sigma_gt, alpha_gt)
    print(movie_noisy.min(), movie_noisy.max())
    # movie_noisy = np.maximum(movie_noisy, 0)
    # movie_noisy = np.minimum(movie_noisy, 8000)
    print(movie_noisy.min(), movie_noisy.max())

    print('PSNR', measure.compare_psnr(movie, movie_noisy,
                                       data_range=movie.max()))

    return movie_noisy, GroundTruth(movie, alpha_gt, sigma_gt)


def load_calcium_examples():
    movie = tifffile.imread('../../images/k53_20160530_RSM_125um_41mW_zoom2p2_00001_00012.tif')
    # movie = tifffile.imread('images/demoMovie.tif')
    # f = h5py.File('images/quietBlock.h5_at', 'r')
    # movie = np.array(f['quietBlock'], dtype=np.float32)[:10]

    print(movie.shape, movie.max())
    return movie, None


def trimming_effect(img, gt=None):
    stride_list = [1, 8]
    perc_kept_list = [(0, 1), (0.1, 0.9)]
    perc_kept_titles = ['No trimming', 'Trimming']

    scatter_cmap = ['#a6cee3', '#fb9a99']
    line_cmap = ['#1f78b4', '#e31a1c']
    line_style_list = ['-', '--']

    fig, axes = plt.subplots(1, len(perc_kept_list) + 1, figsize=(24, 8))

    axes[0].imshow(img)
    axes[0].axis('off')

    for k, perc_kept in enumerate(perc_kept_list):
        print('Trim = {}'.format(perc_kept))

        plt_handle_labels = []
        for i, stride in enumerate(stride_list):
            scatter_color = scatter_cmap[i]
            line_color = line_cmap[i]
            line_style = line_style_list[i]
            label = 'stride={}'.format(stride)
            print(label)

            t = timeit.default_timer()
            means, variances = est.compute_blocks_mean_var(img, stride=stride,
                                                           perc_kept=perc_kept)
            sigma_est, alpha_est = est.regress_sigma_alpha(means, variances)
            print('\tTime', timeit.default_timer() - t)
            print(
                '\talpha = {}; sigma = {}'.format(alpha_est, sigma_est))

            sca_h = axes[k + 1].scatter(means, variances, marker='.', alpha=0.5,
                                        color=scatter_color, edgecolors='none')
            plt_handle_labels.append((sca_h, 'Patches ({})'.format(label)))

            x = np.array([means.min(), means.max()])
            plt_h1 = axes[k + 1].plot(x, alpha_est * x + sigma_est ** 2,
                                      color=line_color, linestyle=line_style,
                                      linewidth=2)
            plt_handle_labels.append((plt_h1[0], 'Estimation ({})'.format(label)))

            axes[k + 1].set_title(perc_kept_titles[k])

            if gt is not None and i == len(stride_list) - 1:
                plt_h2 = axes[k + 1].plot(x, gt.alpha * x + gt.sigma ** 2,
                                         color='k')
                axes[k + 1].fill_between(x, 0.9 * gt.alpha * x + gt.sigma ** 2,
                                         1.1 * gt.alpha * x + gt.sigma ** 2,
                                         facecolor='k', alpha=0.2)
                plt_h3 = axes[k + 1].fill(np.NaN, np.NaN, facecolor='k',
                                          alpha=0.2)
                plt_handle_labels.append(((plt_h2[0], plt_h3[0]),
                                          r'Ground truth ($\pm 10\%$)'))

            axes[k + 1].set_xlabel('Block means')
            axes[k + 1].set_ylabel('Block variances')

            lgnd = axes[k + 1].legend(*list(zip(*plt_handle_labels)),
                                      loc='upper left')
            for h in lgnd.legendHandles:
                h._sizes = [200]


def frame_test(img, gt=None):
    stride_list = [8]
    perc_kept = (0, 1)

    scatter_cmap = ['#a6cee3', '#fb9a99']
    line_cmap = ['#1f78b4', '#e31a1c']
    line_style_list = ['-', '-.']

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    axes[0].imshow(img)
    axes[0].axis('off')

    for i, stride in enumerate(stride_list):
        scatter_color = scatter_cmap[i]
        line_color = line_cmap[i]
        line_style = line_style_list[i]
        label = 'stride={}'.format(stride)
        print(label)

        t = timeit.default_timer()
        means, variances = est.compute_blocks_mean_var(img, stride=stride)
        sigma_est, alpha_est = est.regress_sigma_alpha(means, variances)
        print('\tTime', timeit.default_timer() - t)
        print('\talpha = {}; sigma = {}'.format(alpha_est, sigma_est))

        if gt is not None:
            compare_variance_stabilization(gt.movie, img, gt.sigma, gt.alpha,
                                           sigma_est, alpha_est)

        t = timeit.default_timer()
        res = est.estimate_sigma_alpha_image(img, stride=stride,
                                             perc_kept=perc_kept)
        print('\tTime', timeit.default_timer() - t)

        if gt is not None:
            compare_variance_stabilization(gt.movie, img, gt.sigma, gt.alpha,
                                           res.sigma, res.alpha)

        axes[1].scatter(means, variances, marker='.', alpha=0.5,
                        color=scatter_color, edgecolors='none',
                        label='Patches ({})'.format(label))

        x = np.array([[means.min()], [means.max()]])
        axes[1].plot(x, alpha_est * x + sigma_est ** 2,
                     color=line_color, linestyle=line_style, linewidth=2,
                     label='Initial estimation ({})'.format(label))

        if gt is not None:
            axes[1].plot(x, res.alpha * x + res.sigma ** 2, color='g',
                         label='Refined estimation ({})'.format(label))

            axes[1].plot(x, gt.alpha * x + gt.sigma ** 2, color='k',
                         label='Ground truth')

        axes[1].set_xlabel('Block means')
        axes[1].set_ylabel('Block variances')

        lgnd = axes[1].legend()
        for h in lgnd.legendHandles:
            h._sizes = [200]

        plot_vst_accumulator_space(res.acc_space, ax=axes[2])


def stability_test(movie):
    stride = 8
    perc_kept = (0.1, 0.9)

    sigmas_initial = []
    sigmas_refined = []
    alphas_initial = []
    alphas_refined = []

    for idx_img, img in enumerate(movie):
        print('Frame number {}/{}'.format(idx_img, len(movie)))
        t = timeit.default_timer()
        means, variances = est.compute_blocks_mean_var(img, stride=stride,
                                                       perc_kept=perc_kept)
        sigma_est, alpha_est = est.regress_sigma_alpha(means, variances)
        print('\tTime', timeit.default_timer() - t)
        print('\talpha = {}; sigma = {}'.format(alpha_est, sigma_est))

        t = timeit.default_timer()
        sigma_tuned, alpha_tuned, _ = est.estimate_sigma_alpha_image(img,
                                                                     stride=stride,
                                                                     perc_kept=perc_kept)
        print('\tTime', timeit.default_timer() - t)

        sigmas_initial.append(sigma_est)
        sigmas_refined.append(sigma_tuned)
        alphas_initial.append(alpha_est)
        alphas_refined.append(alpha_tuned)

    fig, ax1 = plt.subplots()
    plt_h1 = ax1.plot(sigmas_initial, color='#a6cee3')
    plt_h2 = ax1.plot(sigmas_refined, color='#1f78b4')
    ax1.set_ylabel(r'$\sigma$', color='#1f78b4')
    ax1.tick_params('y', colors='#1f78b4')
    ax2 = ax1.twinx()
    plt_h3 = ax2.plot(alphas_initial, color='#fb9a99')
    plt_h4 = ax2.plot(alphas_refined, color='#e31a1c')
    ax2.set_ylabel(r'$\alpha$', color='#e31a1c')
    ax2.tick_params('y', colors='#e31a1c')

    plt.legend([plt_h1[0], plt_h2[0], plt_h3[0], plt_h4[0]],
               [r'Initial $\sigma$', r'Refined $\sigma$',
                r'Initial $\alpha$', r'Refined $\alpha$'])


def apply_vst(movie, alpha, sigma):
    import houghvst.estimation.gat as gat
    movie_gat = gat.compute(movie, sigma, alpha=alpha)
    tifffile.imsave('result1.tif', movie_gat.astype(np.float32))

    plt.figure()
    plt.subplot(121)
    plt.imshow(movie[0, :, :])
    plt.subplot(122)
    plt.imshow(movie_gat[0, :, :])

    plt.figure()
    plt.subplot(121)
    plt.imshow(movie[1775, :, :])
    plt.subplot(122)
    plt.imshow(movie_gat[1775, :, :])


def main():
    # movie_noisy, gt = load_toy_examples()
    movie_noisy, gt = load_calcium_examples()

    # trimming_effect(movie_noisy[0], gt=gt)

    # idx_img = 1775
    idx_img = 0
    frame_test(movie_noisy[idx_img], gt=gt)
    # apply_vst(movie_noisy, 120.2286364943876, 171.09473684210528)

    # stability_test(movie_noisy[::100])


if __name__ == '__main__':
    main()
    plt.show()
