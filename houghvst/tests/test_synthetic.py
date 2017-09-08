import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage, stats
import timeit
from noise import estimate_awgn, gat, regions, utils, ponomarenko
import plotting, measures


def test():
    # im = misc.face(gray=True)
    # im = misc.ascent()
    im = 128 + np.zeros((469, 704))
    im[:256, :] = 128
    im[256:, :] = 255
    # im = 128 + np.zeros((24, 10000))
    # im[:12, :] = 128
    # im[12:, :] = 255
    # im = np.arange(50, 562)[:, np.newaxis]
    # im = np.repeat(im, 512, axis=1)
    # im = ndimage.imread('images/dice.png', mode='F')
    # im = ndimage.imread('images/computer.png', mode='F')
    # im = ndimage.imread('images/traffic.png', mode='F')
    # im = ndimage.imread('images/bag.png', mode='F')
    im = im.astype(np.float)
    # print(im.shape, im.min())

    sigma_gt = 12
    alpha_gt = 5
    img_noisy = utils.poisson_gaussian_noise(im, sigma_gt, alpha_gt)

    print('Noisy img range', img_noisy.min(), img_noisy.max())

    # t = timeit.default_timer()
    # segments, labels, img_noisy_resized = estimate_awgn.find_segments(img_noisy)
    # print('time', timeit.default_timer() - t)
    #
    # plt.figure()
    # plt.subplot(121)
    # plt.imshow(img_noisy, cmap='gray')
    # plt.subplot(122)
    # plotting.plot_slic_boundaries(img_noisy_resized, labels, normalize=True)

    # return

    # sigma = 10.4
    # alpha = 5.2
    sigma = 14.3
    alpha = 4.9

    # t = timeit.default_timer()
    # estimate_awgn.score_std(img_noisy, segments, sigma, alpha)
    # print('time', timeit.default_timer() - t)

    t = timeit.default_timer()
    ponomarenko.vst_estimation_point(img_noisy, sigma, alpha)
    print('time', timeit.default_timer() - t)

    measures.compare_variance_stabilization(im, img_noisy, sigma_gt, alpha_gt,
                                            sigma, alpha)
    return

    # # sigma_min, sigma_max = 12, 12.2
    # # alpha_min, alpha_max = 2, 10
    sigma_min, sigma_max = 5, 20
    alpha_min, alpha_max = 4.8, 5.2
    # sigma_min, sigma_max = 5, 100
    # alpha_min, alpha_max = 1, 5
    step = 0.1
    sigmas = np.arange(10 * sigma_min, 10 * sigma_max, 10 * step) / 10.
    alphas = np.arange(10 * alpha_min, 10 * alpha_max, 10 * step) / 10.

    t = timeit.default_timer()
    # sigma_est, alpha_est, score = estimate_awgn.vst_estimation(img_noisy, segments, sigmas, alphas)
    sigma_est, alpha_est, score = ponomarenko.vst_estimation(img_noisy, sigmas,
                                                               alphas)
    print('time', timeit.default_timer() - t)

    plt.figure()
    plotting.plot_vst_score(score, sigma_min, sigma_max, alpha_min, alpha_max,
                            step)
    plt.annotate(r'Ground truth: $\sigma={}$, $\alpha={}$'.format(sigma_gt,
                                                                  alpha_gt),
                 xy=(alpha_gt, sigma_gt), xycoords='data',
                 xytext=(0.05, 0.95), textcoords='axes fraction',
                 va='top', ha='left',
                 color='w', size='large',
                 arrowprops=dict(facecolor='w', edgecolor='none', shrink=0.,
                                 width=1, headwidth=3, headlength=3,
                                 )
                 )
    plt.annotate(r'Estimated: $\sigma={}$, $\alpha={}$'.format(sigma_est,
                                                               alpha_est),
                 xy=(alpha_est, sigma_est), xycoords='data',
                 xytext=(0.05, 0.05), textcoords='axes fraction',
                 va='bottom', ha='left',
                 color='w', size='large',
                 arrowprops=dict(facecolor='w', edgecolor='none', shrink=0.,
                                 width=1, headwidth=3, headlength=3)
                 )

    measures.compare_variance_stabilization(im, img_noisy, sigma_gt, alpha_gt,
                                            sigma_est, alpha_est)


def simulate_awgn_correction():
    n_tests = 100
    estimations = np.zeros((n_tests,))
    for i in range(n_tests):
        im = np.random.randn(512, 512)
        estimations[i] = estimate_awgn.estimate_awgn_laplacian(im)

    plt.figure()
    plt.plot(estimations / estimations.mean())
    # plt.plot([0, n_tests-1], [estimations.mean()] * 2)
    # plt.plot([0, n_tests - 1], [estimations.mean() + estimations.std()] * 2, 'k')
    # plt.plot([0, n_tests - 1], [estimations.mean() - estimations.std()] * 2, 'k')

if __name__ == '__main__':
    np.random.seed(1)

    # score = 0
    # for i in range(10):
    #     score += test()
    #
    # sigma_min, sigma_max = 5, 15
    # # alpha_min, alpha_max = 0.2, 10
    # alpha_min, alpha_max = 4, 6.2
    # step = 0.2
    # sigmas = np.arange(10 * sigma_min, 10 * sigma_max, 10 * step) / 10.
    # alphas = np.arange(10 * alpha_min, 10 * alpha_max, 10 * step) / 10.
    #
    # plt.figure()
    # plotting.plot_vst_score(score, sigma_min, sigma_max, alpha_min, alpha_max,
    #                         step)

    test()
    # simulate_awgn_correction()

    plt.show()
