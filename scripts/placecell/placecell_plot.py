#from environ import *
ROOTDIR = '/tf/workspace/NMPPC'
import os
os.chdir(ROOTDIR)
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'mathtext.fontset': 'cm'})
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import seaborn as sns
sns.set('poster', 'white', 'dark')
pallete = plt.rcParams['axes.prop_cycle'].by_key()['color']

ratnames = ['Gatsby_08282013', 'Gatsby_08022013', 'Achilles_11012013', 'Achilles_10252013']

perms = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
         [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
         [2, 3, 4, 5, 6, 7, 8, 9, 0, 1],
         [3, 4, 5, 6, 7, 8, 9, 0, 1, 2],
         [4, 5, 6, 7, 8, 9, 0, 1, 2, 3],
         [5, 6, 7, 8, 9, 0, 1, 2, 3, 4],
         [6, 7, 8, 9, 0, 1, 2, 3, 4, 5],
         [7, 8, 9, 0, 1, 2, 3, 4, 5, 6],
         [8, 9, 0, 1, 2, 3, 4, 5, 6, 7],
         [9, 0, 1, 2, 3, 4, 5, 6, 7, 8]]

methods = ['JVAE', 'CVAE', 'GMM']

for perm in perms:

    fig = plt.figure(figsize=(32, 20))
    gs = gridspec.GridSpec(nrows=4, ncols=5, width_ratios=[-0.3, 1, 2.5, 2.5, 2.5])

    axs = [[fig.add_subplot(gs[i, j]) for j in range(1, 5)] for i in range(4)]

    fig.subplots_adjust(wspace=0.4, hspace=0.5)
    fig.align_xlabels(axs)
    fig.align_ylabels(axs)

    for i, ratname in enumerate(ratnames):

        dirname = os.path.join(ROOTDIR, 'results', 'JVAE', ratname, ''.join(map(str, perm)))
        sep, ttilde, y, y_smoothed_means, y_smoothed_covs = np.load(os.path.join(dirname, 'reconstruct_test.npy'), allow_pickle=True)

        y_center = 0.5 * (np.nanmax(y, 0) + np.nanmin(y, 0))
        y_max = np.nanmax(np.abs(y - y_center)) + 0.2

        axs[i][0].plot(y[:, 0, 0], y[:, 0, 1])
        axs[i][0].set_aspect('equal')
        axs[i][0].set_xlim(y_center[0, 0] - y_max, y_center[0, 0] + y_max)
        axs[i][0].set_ylim(y_center[0, 1] - y_max, y_center[0, 1] + y_max)

        axs[i][0].set_xlabel(r'$y_1$', fontsize=30)
        axs[i][0].set_ylabel(r'$y_2$', fontsize=30)

        axs[i][0].set_title(ratname, fontsize=25, fontweight='bold', rotation='vertical', x=-0.7, y=-0.3, multialignment='center')

        for l, method in zip(range(1, 4), methods):

            dirname = os.path.join(ROOTDIR, 'results', method, ratname, ''.join(map(str, perm)))
            sep, ttilde, y, y_smoothed_means, y_smoothed_covs = np.load(os.path.join(dirname, 'reconstruct_test.npy'), allow_pickle=True)

            axs[i][l].fill_between(ttilde[:, 0], y_smoothed_means[:, 0, 0] - 2. * y_smoothed_covs[:, 0, 0] ** 0.5, y_smoothed_means[:, 0, 0] + 2. * y_smoothed_covs[:, 0, 0] ** 0.5, alpha=0.5)
            axs[i][l].fill_between(ttilde[:, 0], y_smoothed_means[:, 0, 1] - 2. * y_smoothed_covs[:, 1, 1] ** 0.5, y_smoothed_means[:, 0, 1] + 2. * y_smoothed_covs[:, 1, 1] ** 0.5, alpha=0.5)

            axs[i][l].plot(ttilde[:, 0], y[:, 0, 0])
            axs[i][l].plot(ttilde[:, 0], y[:, 0, 1])

            axs[i][l].set_xlabel('Time [s]')
            axs[i][l].set_ylabel(r'$y_1, \, y_2$', fontsize=30)

            axs[i][l].set_xlim(sep)
            axs[i][l].set_ylim(np.nanmin(y)-0.1, np.nanmax(y)+0.1)

            if i == 0:
                axs[i][l].set_title(method, pad=15, fontsize=30, fontweight='bold')

    plt.savefig(os.path.join(ROOTDIR, 'figures', 'placecell_unsorted_' + ''.join(map(str, perm)) + '.png'),  bbox_inches="tight", dpi=100)
