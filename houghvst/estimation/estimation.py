from collections import namedtuple
from itertools import product
import sklearn.linear_model as lm
import numpy as np
from scipy.fftpack import dct
from houghvst.estimation import gat, regions


def _compute_index(img_size, p):
    return int(np.round(img_size * p))


def _dctii(arr, axis=-1):
    return dct(arr, axis=axis, norm='ortho')


def compute_blocks_dct(blocks):
    return _dctii(_dctii(blocks, axis=-1), axis=-2)


def argsort_blocks_dct(blocks_dct, mask):
    var_low = np.mean(blocks_dct[..., mask] ** 2, axis=1)
    return np.argsort(var_low)


class NoiseSorter:
    def __init__(self, block_size=8, hl_thresh=None, stride=1,
                 perc_kept=(0., 1.)):
        if hl_thresh is None:
            hl_thresh = block_size + 1

        self.block_size = block_size
        self.stride = stride
        self.hl_thresh = hl_thresh
        self.perc_kept = perc_kept

        self.low_mask = np.zeros((block_size, block_size), dtype=np.bool)
        for i, j in product(range(block_size), repeat=2):
            k = i + j
            if 0 < k < hl_thresh:
                self.low_mask[i, j] = True

        self.blocks = None
        self.order = None
        self._blocks_dct = None

    def fit_img(self, img):
        blocks = regions.im2col(img, self.block_size, stride=self.stride)
        self.fit_blocks(blocks)

    def fit(self, blocks):
        blocks_dct = compute_blocks_dct(blocks)

        var_low = np.mean(blocks_dct[..., self.low_mask] ** 2, axis=1)
        order = np.argsort(var_low)

        try:
            k0 = _compute_index(len(blocks), self.perc_kept[0])
            k1 = _compute_index(len(blocks), self.perc_kept[1])
            s = slice(k0, k1)
        except TypeError:
            k = _compute_index(len(blocks), self.perc_kept)
            s = slice(None, k)

        self.order = order[s]
        self.blocks = blocks[self.order]
        self.blocks_dct = blocks_dct[self.order]


class PonomarenkoNoiseEstimator(NoiseSorter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.high_mask = np.logical_not(self.low_mask)
        self.high_mask[0, 0] = False

    def compute_noise(self):
        counts = np.arange(1, len(self.blocks_dct) + 1)
        var_high = (np.cumsum(self.blocks_dct ** 2, axis=0)
                    / counts[:, np.newaxis, np.newaxis])

        sigmas2 = np.median(var_high[:, self.high_mask], axis=1)
        sigmas = np.sqrt(sigmas2)
        return sigmas


def compute_score(sigmas, tol):
    x = (sigmas - 1) / tol
    weights = np.exp(-x ** 2)
    score = np.sum(weights)
    if score > 0:
        sigma_est = np.sum(sigmas * weights) / score
    else:
        sigma_est = np.nan
    return sigma_est, score


AccumulatorSpace = namedtuple('AccumulatorSpace', ['score', 'sigma_range',
                                                   'alpha_range'])


def hough_estimation(blocks, sigma_range, alpha_range, tol=1e-2, **kwargs):
    noise_estimator = PonomarenkoNoiseEstimator(**kwargs)
    score = np.zeros((len(sigma_range), len(alpha_range)))

    for i_a, alpha in enumerate(alpha_range):
        for i_s, sigma in enumerate(sigma_range):
            # print('{} / {}'.format(i_a, len(alpha_range)), '--',
            #       '{} / {}'.format(i_s, len(sigma_range)))

            blocks_gat = gat.compute(blocks, sigma, alpha=alpha)
            noise_estimator.fit_blocks(blocks_gat)
            sigmas = noise_estimator.compute_noise()
            score[i_s, i_a] = compute_score(sigmas, tol)[1]

    max_score_idx = np.argmax(score)
    best_params = np.unravel_index(max_score_idx, score.shape)
    sigma_est = sigma_range[best_params[0]]
    alpha_est = alpha_range[best_params[1]]

    print('\tHighest score=', score[best_params[0], best_params[1]])

    acc = AccumulatorSpace(score, sigma_range, alpha_range)
    return sigma_est, alpha_est, acc


def vst_estimation_point(img, sigma, alpha, tol=2e-3):
    return hough_estimation(img, [sigma], [alpha], tol=tol)


def compute_blocks_mean_var(img, block_size=8, stride=1):
    blocks = regions.im2col(img, block_size, stride)
    return compute_mean_var(blocks)


def compute_mean_var(blocks):
    means = np.mean(blocks, axis=(1, 2))
    variances = np.var(blocks, axis=(1, 2), ddof=1)
    return means, variances


def regress_sigma_alpha(means, variances):
    mu = means.mean()
    # mu=0

    reg = lm.HuberRegressor(alpha=0, epsilon=1.01, fit_intercept=True)
    reg.fit(means[:, np.newaxis] - mu, variances)

    alpha_est = reg.coef_[0]
    sigma_est = np.sqrt(reg.intercept_ - alpha_est * mu)
    print('\talpha = {}; sigma = {}'.format(alpha_est, sigma_est))
    return sigma_est, alpha_est


def initial_estimate_sigma_alpha(blocks):
    means, variances = compute_mean_var(blocks)
    sigma_est, alpha_est = regress_sigma_alpha(means, variances)
    return sigma_est, alpha_est


def estimate_sigma_alpha_image(img, **kwargs):
    blocks = regions.im2col(img, kwargs['block_size'], kwargs['stride'])
    return estimate_sigma_alpha_blocks(blocks, **kwargs)


def estimate_sigma_alpha_blocks(blocks, **kwargs):
    _, alpha_est = initial_estimate_sigma_alpha(blocks)
    print('\tinitial alpha = {}'.format(alpha_est))

    alpha_range = [alpha_est]
    sigma_range = np.arange(0, 500, 1)
    sigma_est, _, _ = hough_estimation(blocks, sigma_range, alpha_range,
                                       **kwargs)
    print('\tinitial sigma = {}'.format(sigma_est))

    sigma_range = np.linspace(sigma_est * 0.9, sigma_est * 1.1, num=20)
    alpha_range = np.linspace(alpha_est * 0.8, alpha_est * 1.2, num=30)
    sigma_est, alpha_est, acc_space = hough_estimation(blocks, sigma_range,
                                                       alpha_range, **kwargs)
    print('\talpha = {}; sigma = {}'.format(alpha_est, sigma_est))

    return sigma_est, alpha_est, acc_space
