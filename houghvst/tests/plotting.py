import colorcet as cc
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def plot_vst_accumulator_space(acc_space, cmap=cc.m_fire, ax=None,
                               plot_estimates=False, plot_focus=False):
    if ax is None:
        ax = plt.gca()

    alpha_step0 = acc_space.alpha_range[1] - acc_space.alpha_range[0]
    alpha_step1 = acc_space.alpha_range[-1] - acc_space.alpha_range[-2]
    sigma_step0 = acc_space.sigma_sq_range[1] - acc_space.sigma_sq_range[0]
    sigma_step1 = acc_space.sigma_sq_range[-1] - acc_space.sigma_sq_range[-2]
    im_plt = ax.imshow(acc_space.score, cmap=cmap,
                       extent=(acc_space.alpha_range[0] - alpha_step0 / 2,
                               acc_space.alpha_range[-1] + alpha_step1 / 2,
                               acc_space.sigma_sq_range[-1] + sigma_step1 / 2,
                               acc_space.sigma_sq_range[0] - sigma_step0 / 2)
                       )
    ax.axis('tight')
    ax.set_xlabel(r'$\alpha$', fontsize='xx-large')
    ax.set_ylabel(r'$\beta$', fontsize='xx-large')
    plt.colorbar(mappable=im_plt, ax=ax)

    if plot_focus:
        len_a = (acc_space.alpha_range[-1] - acc_space.alpha_range[0]) / 4
        len_s = (acc_space.sigma_sq_range[-1]
                 - acc_space.sigma_sq_range[0]) / 10
        rect = mpatches.Rectangle((acc_space.alpha - len_a / 2,
                                   acc_space.sigma_sq - len_s / 2),
                                  len_a, len_s, ec='#0000cd', fc='none',
                                  linewidth=3)
        ax.add_artist(rect)

    if plot_estimates:
        tag_str = r'$\alpha={:.2f}$, $\beta={:.2f}$'
        bbox_props = dict(boxstyle='round', fc='w', ec='#0000cd', alpha=0.5)
        ax.annotate(tag_str.format(acc_space.alpha, acc_space.sigma_sq),
                    xy=(acc_space.alpha, acc_space.sigma_sq), xycoords='data',
                    xytext=(0.95, 0.05), textcoords='axes fraction',
                    va='bottom', ha='right', color='#0000cd', size='xx-large',
                    bbox=bbox_props,
                    arrowprops=dict(facecolor='#0000cd', edgecolor='none',
                                    shrink=0., width=2, headwidth=3,
                                    headlength=3)
                    )
