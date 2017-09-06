import colorcet as cc
import matplotlib.pyplot as plt
import matplotlib.lines as plt_lines
import matplotlib.ticker as plt_tick
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import seaborn.apionly as sns
from sklearn.linear_model import LinearRegression
import tifffile
from houghvst.estimation.estimation import compute_blocks_mean_var


def compute_temporal_mean_var(movie):
    means = np.mean(movie, axis=0)
    variances = np.var(movie, axis=0, ddof=1)
    return means.flatten(), variances.flatten()


def ground_truth_pixelwise(dir_name, files, box=None, plt_regression=True):
    colors = sns.color_palette('Pastel1', len(files))
    colors_means = sns.color_palette('Set1', len(files))

    fig, axes_scatter = plt.subplots(1, 1, figsize=(8, 4))
    means_means = []
    variances_means = []
    for k, fn in enumerate(reversed(files)):
        movie = tifffile.imread(dir_name + fn)

        if box is not None:
            movie = movie[:, box[0]:box[1]:2, box[2]:box[3]]
        else:
            movie = movie[:, ::2, :]

        means, variances = compute_temporal_mean_var(movie)

        axes_scatter.scatter(means, variances, marker='.',
                             alpha=1, color=colors[-k-1], edgecolors='none',
                             label='Movie {}'.format(k+1))

        means_means.append(means.mean())
        variances_means.append(variances.mean())

    means_means = list(reversed(means_means))
    variances_means = list(reversed(variances_means))

    if plt_regression:
        mod = LinearRegression()
        mod.fit(np.array(means_means)[:, np.newaxis],
                np.array(variances_means)[:, np.newaxis])
        x = np.array([1200, 2700])[:, np.newaxis]
        axes_scatter.plot(x, mod.predict(x), 'k-')

    for k, (mean, variance) in enumerate(zip(means_means, variances_means)):
        axes_scatter.scatter(mean, variance, marker='+',
                             s=1000, linewidth=2, color=colors_means[k],
                             edgecolor='k', zorder=1000)

    axes_scatter.yaxis.set_major_formatter(plt_tick.FormatStrFormatter('%.0e'))

    axes_scatter.set_xlabel('Mean')
    axes_scatter.set_ylabel('Variance')

    markers1 = [plt_lines.Line2D([], [], marker='o', color=colors[k],
                                 linestyle='None')
                for k in range(len(files))]
    label1 = ['Movie {}'.format(k+1)
              for k in range(len(files))]
    markers2 = [plt_lines.Line2D([], [], marker='+', color=colors_means[k],
                                 mec=colors_means[k], linestyle='None',
                                 markersize=10, markeredgewidth=2)
                for k in range(len(files))]
    label2 = ['Movie {} - average'.format(k+1)
              for k in range(len(files))]
    plt.legend(markers1 + markers2, label1 + label2,
               bbox_to_anchor=(1.01, 1), loc='best')
    fig.tight_layout(rect=(0, 0, 0.78, 1))

    return means_means, variances_means


def ground_truth_patchwise(dir_name, files, gt_means_means, gt_vars_means,
                           box=None):
    colors = sns.color_palette('Pastel1', len(files))
    colors_means = sns.color_palette('Set1', len(files))

    fig, axes = plt.subplots(1, 1, figsize=(8, 4))
    for k, fn in enumerate(files):
        movie = tifffile.imread(dir_name + fn)
        if box is not None:
            movie = movie[:, box[0]:box[1]:2, box[2]:box[3]]
        else:
            movie = movie[:, ::2, :]

        means = []
        variances = []
        for i in range(0, len(movie)):
            means_i, vars_i = compute_blocks_mean_var(movie[i],
                                                      block_size=8, stride=8)
            means.append(means_i)
            variances.append(vars_i)

        means = np.hstack(means)
        variances = np.hstack(variances)
        print(means.shape, variances.shape)

        axes.scatter(means, variances, marker='.', alpha=1, color=colors[k],
                     edgecolors='none')

        axes.scatter(means.mean(), variances.mean(), marker='x', s=1000,
                     linewidth=2, color=colors_means[k],
                     edgecolor=colors_means[k], zorder=1000)

        axes.scatter(gt_means_means[k], gt_vars_means[k], marker='+', s=1000,
                     linewidth=2, color=colors_means[k],
                     edgecolor=colors_means[k], zorder=1000)

    axes.yaxis.set_major_formatter(
        plt_tick.FormatStrFormatter('%.0e'))

    axes.set_xlabel('Mean')
    axes.set_ylabel('Variance')

    markers1 = [plt_lines.Line2D([], [], marker='o', color=colors[k],
                                 linestyle='None')
                for k in range(len(files))]
    label1 = ['Movie {} - patchwise'.format(k+1)
              for k in range(len(files))]
    markers2 = [plt_lines.Line2D([], [], marker='x', color=colors_means[k],
                                 mec=colors_means[k], linestyle='None',
                                 markersize=10, markeredgewidth=2)
                for k in range(len(files))]
    label2 = ['Movie {} - patchwise AVG'.format(k+1)
              for k in range(len(files))]
    markers3 = [plt_lines.Line2D([], [], marker='+', color=colors_means[k],
                                 mec=colors_means[k], linestyle='None',
                                 markersize=10, markeredgewidth=2)
                for k in range(len(files))]
    label3 = ['Movie {} - pixelwise AVG'.format(k+1)
              for k in range(len(files))]
    plt.legend(markers1 + markers2 + markers3, label1 + label2 + label3,
               bbox_to_anchor=(1.01, 1), loc='best')
    fig.tight_layout(rect=(0, 0, 0.7, 1))


def movie_plot(dir_name, files, box=None):
    fig = plt.figure(figsize=(8, 4))
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(2, len(files)),
                     direction="row",
                     axes_pad=0.1,
                     add_all=True,
                     share_all=True)

    for k, fn in enumerate(files):
        movie = tifffile.imread(dir_name + fn)
        if box is not None:
            movie = movie[:, box[0]:box[1]:2, box[2]:box[3]]
        else:
            movie = movie[:, ::2, :]

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

    # plt.tight_layout(h_pad=0, w_pad=0)


def main():
    dir_name = '/Users/mtepper/Dropbox (Simons Foundation)/Noise-data-Lloyd/'
    files = ['20170718_FluoroSlide_t-0{:02d}_Cycle00001_Ch1.tif'.format(k)
             for k in range(6, 11)]

    # cuts_collection = [plotting.Cut('h', 250, 'b'),
    #                    plotting.Cut('h', 270, 'r')]
    # _, axes_transversal = plt.subplots(2, len(files), sharey='row',
    #                                    figsize=(12, 3))
    # for k, fn in enumerate(files):
    #     movie = tifffile.imread(dir_name + fn)
    #     img = movie[0].astype(np.float)
    #     plotting.transversal_cuts(img, cuts_collection, normalize=True,
    #                               axes=axes_transversal[:, k])
    # plt.suptitle(title)

    movie_plot(dir_name, files)

    # box = (0, 512, 0, 100)
    # movie_plot(dir_name, files, box)
    # res = ground_truth_pixelwise(dir_name, files, box=box, plt_regression=False)
    # gt_means_means, gt_vars_means = res
    # ground_truth_patchwise(dir_name, files, gt_means_means, gt_vars_means,
    #                        box=box, plt_regression=False)

    box = (210, 300, 210, 300)
    res = ground_truth_pixelwise(dir_name, files, box=box, plt_regression=True)
    gt_means_means, gt_vars_means = res
    ground_truth_patchwise(dir_name, files, gt_means_means, gt_vars_means,
                           box=box)


if __name__ == '__main__':
    main()
    plt.show()

