from environ import *
import os
os.chdir(ROOTDIR)

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'mathtext.fontset': 'cm'})
pallete = plt.rcParams['axes.prop_cycle'].by_key()['color']
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import seaborn as sns
sns.set('poster', 'white', 'dark')

ratnames = ['Lorenz_3D', 'Lorenz_2D']

methods = ['JVAE', 'CVAE', 'GMM']

for batch_num in range(10):

    fig = plt.figure(figsize=(24, 16))
    fig.subplots_adjust(wspace=0., hspace=0.2)

    gs = gridspec.GridSpec(nrows=4, ncols=5, width_ratios=[0.2, 1, 1, 1, 1])

    axs = [[fig.add_subplot(gs[0, j], projection='3d') for j in range(1, 5)],
           [fig.add_subplot(gs[1, j]) for j in range(1, 5)]]

    ratname = ratnames[0]

    dirname = os.path.join(ROOTDIR, 'results', 'JVAE', ratname, 'batch' + str(batch_num))
    sep, ttilde, y, y_smoothed_means, y_smoothed_covs = np.load(os.path.join(dirname, 'reconstruct_test.npy'), allow_pickle=True)
    y_center = 0.5 * (np.nanmax(y, 0) + np.nanmin(y, 0))

    axs[0][0].set_title('Lorenz_3D', fontsize=30, fontweight='bold', rotation='vertical', x=-0.7, y=0.2, multialignment='center')

    axs[0][0].plot3D(y[:, 0, 0], y[:, 0, 1], y[:, 0, 2], lw=2.)

    axs[0][0].set_xlim(y_center[0, 0] - 3, y_center[0, 0] + 3)
    axs[0][0].set_ylim(y_center[0, 1] - 3, y_center[0, 1] + 3)
    axs[0][0].set_zlim(y_center[0, 2] - 3, y_center[0, 2] + 3)

    axs[0][0].set_xticklabels([])
    axs[0][0].set_yticklabels([])
    axs[0][0].set_zticklabels([])

    axs[0][0].set_xlabel(r'$y_1$', fontsize=30)
    axs[0][0].set_ylabel(r'$y_2$', fontsize=30)
    axs[0][0].set_zlabel(r'$y_3$', fontsize=30)

    for l, method in zip(range(1, 4), methods):

        dirname = os.path.join(ROOTDIR, 'results', method, ratname, 'batch' + str(batch_num))
        sep, ttilde, y, y_smoothed_means, y_smoothed_covs = np.load(os.path.join(dirname, 'reconstruct_test.npy'), allow_pickle=True)
        y_center = 0.5 * (np.nanmax(y, 0) + np.nanmin(y, 0))

        axs[0][l].plot3D(y_smoothed_means[:, 0, 0], y_smoothed_means[:, 0, 1], y_smoothed_means[:, 0, 2], lw=2.)

        axs[0][l].set_xlim(y_center[0, 0] - 3, y_center[0, 0] + 3)
        axs[0][l].set_ylim(y_center[0, 1] - 3, y_center[0, 1] + 3)
        axs[0][l].set_zlim(y_center[0, 2] - 3, y_center[0, 2] + 3)

        axs[0][l].set_xticklabels([])
        axs[0][l].set_yticklabels([])
        axs[0][l].set_zticklabels([])

        axs[0][l].set_xlabel(r'$y_1$', fontsize=30)
        axs[0][l].set_ylabel(r'$y_2$', fontsize=30)
        axs[0][l].set_zlabel(r'$y_3$', fontsize=30)

        axs[0][l].set_title(method, pad=15, fontsize=30, fontweight='bold')

    ratname = ratnames[1]

    dirname = os.path.join(ROOTDIR, 'results', 'JVAE', ratname, 'batch' + str(batch_num))
    sep, ttilde, y, y_smoothed_means, y_smoothed_covs = np.load(os.path.join(dirname, 'reconstruct_test.npy'), allow_pickle=True)
    y_center = 0.5 * (np.nanmax(y, 0) + np.nanmin(y, 0))

    axs[1][0].set_title('Lorenz_2D', fontsize=30, fontweight='bold', rotation='vertical', x=-0.7, y=0., multialignment='center')

    axs[1][0].plot(y[:, 0, 0], y[:, 0, 1], lw=2.)
    axs[1][0].set_aspect('equal')
    axs[1][0].set_xlim(y_center[0, 0] - 3, y_center[0, 0] + 3)
    axs[1][0].set_ylim(y_center[0, 1] - 3, y_center[0, 1] + 3)

    axs[1][0].set_xlabel(r'$y_1$', fontsize=30)
    axs[1][0].set_ylabel(r'$y_3$', fontsize=30)

    for l, method in zip(range(1, 4), methods):

        dirname = os.path.join(ROOTDIR, 'results', method, ratname, 'batch' + str(batch_num))
        sep, ttilde, y, y_smoothed_means, y_smoothed_covs = np.load(os.path.join(dirname, 'reconstruct_test.npy'), allow_pickle=True)
        y_center = 0.5 * (np.nanmax(y, 0) + np.nanmin(y, 0))

        axs[1][l].plot(y_smoothed_means[:, 0, 0], y_smoothed_means[:, 0, 1], lw=2.)
        axs[1][l].set_aspect('equal')
        axs[1][l].set_xlim(y_center[0, 0] - 3, y_center[0, 0] + 3)
        axs[1][l].set_ylim(y_center[0, 1] - 3, y_center[0, 1] + 3)

        axs[1][l].set_xlabel(r'$y_1$', fontsize=30)
        axs[1][l].set_ylabel(r'$y_3$', fontsize=30)

    plt.savefig(os.path.join(ROOTDIR, 'figures', 'synthetic_unsorted_' + 'batch' + str(batch_num) + '.png'),  bbox_inches="tight", dpi=100)


ratname = ratnames[0]

for batch_num in range(10):

    data = np.load(os.path.join(ROOTDIR, 'SPIKE_data', ratname, 'data0.npy'), allow_pickle=True)[()]
    spikeTrain, stim, sep = data['spikeTrain'], data['stim'], data['sep']
    clus = data['clu']

    spike_max = 2. * np.max([np.max(np.std(spike[:, 1:], axis=0)) for spike in spikeTrain])

    fig = plt.figure(figsize=(18, 16))
    fig.subplots_adjust(wspace=0.4, hspace=0.2)

    gs = gridspec.GridSpec(nrows=5, ncols=4, height_ratios=[2., 1., 1., 1., 1.])

    axs1 = [fig.add_subplot(gs[0, j], projection='3d') for j in range(4)]
    axs2 = [[fig.add_subplot(gs[i+1, j]) for i in range(4)] for j in range(4)]


    for spike, clu, ax1, ax2 in zip(spikeTrain, clus, axs1, axs2):

        x = stim[:, 1:]
        x_center = 0.5 * (np.nanmax(x, 0) + np.nanmin(x, 0))

        ax1.plot3D(x[:2000, 0], x[:2000, 1], x[:2000, 2], lw=2., alpha=0.1, color=pallete[0])

        ax1.set_xlim(x_center[0] - 3, x_center[0] + 3)
        ax1.set_ylim(x_center[1] - 3, x_center[1] + 3)
        ax1.set_zlim(x_center[2] - 3, x_center[2] + 3)

        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_zticklabels([])

        ax1.set_xlabel(r'$x_1$', fontsize=30)
        ax1.set_ylabel(r'$x_2$', fontsize=30)
        ax1.set_zlabel(r'$x_3$', fontsize=30)


        for i, k in enumerate(np.unique(clu)):

            covariate = stim[(spike[clu==k, 0] // (stim[1, 0] - stim[0, 0])).astype(int), 1:]
            ax1.scatter(covariate[:500, 0], covariate[:500, 1], covariate[:500, 2], alpha=0.2, color=pallete[int(k)%10], marker='o')

            waveform = spike[clu==k, 1:]
            ax2[i].plot(np.arange(32), waveform.mean(0),  color=pallete[int(k)%10])
            ax2[i].fill_between(np.arange(32), waveform.mean(0) - 2. * waveform.std(0), waveform.mean(0) + 2. * waveform.std(0), alpha=0.2, color=pallete[int(k)%10])

            ax2[i].set_xlim(0, 32)
            ax2[i].set_ylim(-spike_max, spike_max)
            ax2[i].set_xticklabels([])
            ax2[i].set_yticklabels([])

    plt.savefig(os.path.join(ROOTDIR, 'figures', 'synthetic_spikes_' + 'batch' + str(batch_num) + '.png'),  bbox_inches="tight", dpi=100)
