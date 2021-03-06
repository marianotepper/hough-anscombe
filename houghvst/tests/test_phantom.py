import matplotlib.pyplot as plt
import matplotlib.lines as plt_lines
import matplotlib.patches as plt_patches
import matplotlib.ticker as plt_tick
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import seaborn.apionly as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import tifffile
from houghvst.estimation.estimation import compute_mean_var, \
    estimate_vst_movie, estimate_vst_image
from houghvst.estimation.gat import compute_gat
from houghvst.estimation.regions import im2col
from houghvst.tests.measures import assess_variance_stabilization,\
    compute_temporal_mean_var


def fix_scientific_notation(x):
    s = '{:.1e}'.format(x)
    s = s.replace('+0', '')
    return s


def ground_truth_pixelwise(dir_name, files, box=None, plt_regression=True):
    colors = sns.color_palette('Pastel1', len(files))
    colors_means = sns.color_palette('Set1', len(files))

    with sns.axes_style("white"):
        fig, axes_scatter = plt.subplots(1, 1, figsize=(8, 4))
        means_means = []
        variances_means = []
        all_means = []
        all_variances = []
        for k, fn in enumerate(files):
            movie = tifffile.imread(dir_name + fn)
            if box is None:
                movie = movie[:, ::2, :]
            else:
                movie = movie[:, box[0]:box[1]:2, box[2]:box[3]]
            movie = movie.astype(np.float32)

            means, variances = compute_temporal_mean_var(movie)
            means = means.flatten()
            variances = variances.flatten()

            means_means.append(means.mean())
            variances_means.append(variances.mean())

            if plt_regression:
                all_means.extend(means)
                all_variances.extend(variances)

            axes_scatter.plot(means, variances, '.', alpha=0.7,
                              color=colors[k], markeredgecolor='none',
                              label='Movie {}'.format(k + 1), zorder=10 - k,
                              rasterized=True)

        if plt_regression:
            all_means = np.array(all_means)[:, np.newaxis]
            all_variances = np.array(all_variances)

            mod = LinearRegression()
            mod.fit(all_means, all_variances)
            x = np.array([all_means.min(), all_means.max()])[:, np.newaxis]
            axes_scatter.plot(x, mod.predict(x), 'k-', zorder=11)
            print('Linear fit:', mod.coef_[0], mod.intercept_)
            print('R2 score', r2_score(all_variances, mod.predict(all_means)))

        for k, (mean, variance) in enumerate(zip(means_means, variances_means)):
            axes_scatter.scatter(mean, variance, marker='+',
                                 s=1000, linewidth=2, color=colors_means[k],
                                 edgecolor='k', zorder=1000)

        axes_scatter.yaxis.set_major_formatter(
            plt_tick.FuncFormatter(lambda t, pos: fix_scientific_notation(t)))

        axes_scatter.set_xlabel('Mean', fontsize='large')
        axes_scatter.set_ylabel('Variance', fontsize='large')

        markers1 = [plt_lines.Line2D([], [], marker='o', color=colors[k],
                                     linestyle='None')
                    for k in range(len(files))]
        label1 = ['Movie {} - pixel'.format(k+1)
                  for k in range(len(files))]
        markers2 = [plt_lines.Line2D([], [], marker='+', color=colors_means[k],
                                     mec=colors_means[k], linestyle='None',
                                     markersize=10, markeredgewidth=2)
                    for k in range(len(files))]
        label2 = ['Movie {} - pixel AVG'.format(k+1)
                  for k in range(len(files))]
        plt.legend(markers1 + markers2, label1 + label2,
                   bbox_to_anchor=(1.01, 1), loc='best', fontsize='large')
        fig.tight_layout(rect=(0, 0, 0.73, 1))
        plt.savefig('ground_truth_pixelwise.pdf')

    return means_means, variances_means


def ground_truth_patchwise(dir_name, files, gt_means_means, gt_vars_means,
                           box=None, block_size=8, stride=8):
    colors = sns.color_palette('Pastel1', len(files))
    colors_means = sns.color_palette('Set1', len(files))

    with sns.axes_style("white"):
        fig, axes = plt.subplots(1, 1, figsize=(8, 4))
        for k, fn in enumerate(files):
            movie = tifffile.imread(dir_name + fn)
            if box is None:
                movie = movie[:, ::2, :]
            else:
                movie = movie[:, box[0]:box[1]:2, box[2]:box[3]]
            movie = movie.astype(np.float32)

            means = []
            variances = []
            for i in range(0, len(movie), 10):
                blocks_i = im2col(movie[i], block_size, stride)
                means_i, vars_i = compute_mean_var(blocks_i)
                means.append(means_i)
                variances.append(vars_i)

            means = np.hstack(means)
            variances = np.hstack(variances)

            axes.plot(means, variances, '.', alpha=0.7, color=colors[k],
                      markeredgecolor='none', zorder=10 - k, rasterized=True)

            axes.scatter(means.mean(), variances.mean(), marker='x', s=1000,
                         linewidth=2, color=colors_means[k],
                         edgecolor=colors_means[k], zorder=1000)

            axes.scatter(gt_means_means[k], gt_vars_means[k], marker='+',
                         s=1000, linewidth=2, color=colors_means[k],
                         edgecolor=colors_means[k], zorder=1000)

        axes.yaxis.set_major_formatter(
            plt_tick.FuncFormatter(lambda t, pos: fix_scientific_notation(t)))

        axes.set_xlabel('Mean', fontsize='large')
        axes.set_ylabel('Variance', fontsize='large')

        markers1 = [plt_lines.Line2D([], [], marker='o', color=colors[k],
                                     linestyle='None')
                    for k in range(len(files))]
        label1 = ['Movie {} - patch'.format(k+1)
                  for k in range(len(files))]
        markers2 = [plt_lines.Line2D([], [], marker='x', color=colors_means[k],
                                     mec=colors_means[k], linestyle='None',
                                     markersize=10, markeredgewidth=2)
                    for k in range(len(files))]
        label2 = ['Movie {} - patch AVG'.format(k+1)
                  for k in range(len(files))]
        markers3 = [plt_lines.Line2D([], [], marker='+', color=colors_means[k],
                                     mec=colors_means[k], linestyle='None',
                                     markersize=10, markeredgewidth=2)
                    for k in range(len(files))]
        label3 = ['Movie {} - pixel AVG'.format(k+1)
                  for k in range(len(files))]
        plt.legend(markers1 + markers2 + markers3, label1 + label2 + label3,
                   bbox_to_anchor=(1.01, 1), loc='best', fontsize='large')
        fig.tight_layout(rect=(0, 0, 0.7, 1))
        plt.savefig('ground_truth_patchwise.pdf')


def ground_truth_estimate_vst(dir_name, files, box=None, block_size=8,
                              stride=8):
    variances_all_single_image = []
    variances_all_multi_image = []

    for k, fn in enumerate(files):
        print(k)

        movie = tifffile.imread(dir_name + fn)
        if box is None:
            movie = movie[:, ::2, :]
        else:
            movie = movie[:, box[0]:box[1]:2, box[2]:box[3]]
        movie = movie.astype(np.float32)

        img_gt = movie.mean(axis=0)
        img_noisy = movie[0]

        # single image estimation
        print('Single-image estimation')
        res = estimate_vst_image(img_noisy, block_size=block_size,
                                 stride=stride)

        movie_gat = compute_gat(movie, res.sigma_sq, res.alpha)
        means, variances = compute_temporal_mean_var(movie_gat)
        print('Temporal variance MEAN={}, STD={}'.format(variances.mean(),
                                                         variances.std(ddof=1)))

        variances = []
        for i in range(len(movie)):
            img_noisy = movie[i]
            v = assess_variance_stabilization(img_gt, img_noisy, res.sigma_sq,
                                              res.alpha, verbose=False,
                                              correct_noiseless=False)
            variances.append(v)
        variances = np.array(variances)
        print('variance AVG', variances.mean())
        variances_all_single_image.append(variances)

        # multi-image estimation
        print('Multi-image estimation')
        res = estimate_vst_movie(movie[::200])

        movie_gat = compute_gat(movie, res.sigma_sq, res.alpha)
        means, variances = compute_temporal_mean_var(movie_gat)
        print('Temporal variance MEAN={}, STD={}'.format(variances.mean(),
                                                         variances.std(ddof=1)))
        variances = []
        for i in range(len(movie)):
            img_noisy = movie[i]
            v = assess_variance_stabilization(img_gt, img_noisy, res.sigma_sq,
                                              res.alpha, verbose=False,
                                              correct_noiseless=False)
            variances.append(v)
        variances = np.array(variances)
        print('variance AVG', variances.mean())
        variances_all_multi_image.append(variances)

    with sns.axes_style('white'):
        plt.figure()
        plt.plot([0, 2 * len(files) - 1], [1, 1], color='k', alpha=0.2,
                 zorder=0)
        pos = np.arange(0, 2 * len(files), 2) + 0.2
        vio_single = plt.violinplot(variances_all_single_image, positions=pos)
        pos = np.arange(1, 2 * len(files), 2) - 0.2
        vio_multi = plt.violinplot(variances_all_multi_image, positions=pos)
        plt.xticks(2 * np.arange(len(files)) + 0.5,
                   ['Movie {}'.format(k+1) for k in range(len(files))],
                   fontsize='large')
        plt.ylabel('Stabilized noise variance per frame', fontsize='large')

        p_single = plt_patches.Patch(
            facecolor=vio_single['bodies'][0].get_facecolor().flatten(),
            edgecolor=vio_single['cbars'].get_edgecolor().flatten(),
            label='Single-image estimation')
        p_multi = plt_patches.Patch(
            facecolor=vio_multi['bodies'][0].get_facecolor().flatten(),
            edgecolor=vio_multi['cbars'].get_edgecolor().flatten(),
            label='Multi-image estimation')
        plt.legend(handles=[p_single, p_multi], loc='upper right',
                   fontsize='large')
        plt.savefig('ground_truth_single-multi.pdf')


def movie_plot(dir_name, files, box=None):
    with sns.axes_style("white"):
        fig = plt.figure(figsize=(9, 4))
        grid = ImageGrid(fig, rect=(0.05, 0, 0.9, 1),
                         nrows_ncols=(2, len(files)),
                         direction="row",
                         axes_pad=0.1,
                         add_all=True,
                         share_all=True)

        for k, fn in enumerate(files):
            movie = tifffile.imread(dir_name + fn)
            if box is None:
                movie = movie[:, ::2, :]
            else:
                movie = movie[:, box[0]:box[1]:2, box[2]:box[3]]

            print('Movie min:', movie.min(), 'max:', movie.max())

            grid.axes_row[0][k].imshow(movie[0], cmap='viridis')
            grid.axes_row[0][k].set_title('Movie {}'.format(k+1))
            grid.axes_row[0][k].grid(False)
            grid.axes_row[0][k].tick_params(axis='both',
                                       which='both',
                                       bottom='off', top='off',
                                       left='off', right='off',
                                       labelbottom='off', labelleft='off')
            grid.axes_row[0][k].spines['top'].set_visible(False)
            grid.axes_row[0][k].spines['right'].set_visible(False)
            grid.axes_row[0][k].spines['bottom'].set_visible(False)
            grid.axes_row[0][k].spines['left'].set_visible(False)

            grid.axes_row[1][k].imshow(np.mean(movie, axis=0), cmap='viridis')
            grid.axes_row[1][k].grid(False)
            grid.axes_row[1][k].tick_params(axis='both',
                                       which='both',
                                       bottom='off', top='off',
                                       left='off', right='off',
                                       labelbottom='off', labelleft='off')
            grid.axes_row[1][k].spines['top'].set_visible(False)
            grid.axes_row[1][k].spines['right'].set_visible(False)
            grid.axes_row[1][k].spines['bottom'].set_visible(False)
            grid.axes_row[1][k].spines['left'].set_visible(False)

        grid.axes_row[0][0].set_ylabel('Individual\nframe')
        grid.axes_row[1][0].set_ylabel('Temporal\nmean')
        # plt.tight_layout()
        plt.savefig('ground_truth_movie.pdf')


def main():
    dir_name = '/Users/mtepper/Dropbox (Simons Foundation)/Noise-data-Lloyd/'
    files = ['20170718_FluoroSlide_t-0{:02d}_Cycle00001_Ch1.tif'.format(k)
             for k in range(6, 11)]

    # movie_plot(dir_name, files)

    box = [50, -50, 50, -50]
    ground_truth_estimate_vst(dir_name, files, box=box)

    box = (210, 300, 210, 300)
    res = ground_truth_pixelwise(dir_name, files, box=box, plt_regression=True)
    gt_means_means, gt_vars_means = res
    ground_truth_patchwise(dir_name, files, gt_means_means, gt_vars_means,
                           box=box)


if __name__ == '__main__':
    main()
    plt.show()

