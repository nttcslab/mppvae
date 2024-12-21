from environ import *
import os
os.chdir(ROOTDIR)
import numpy as np
np.set_printoptions(suppress = True)
import tensorflow as tf
import pandas as pd
import itertools
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import matplotlib.ticker as ptick
from matplotlib.colors import LogNorm, Normalize
import seaborn as sns
sns.set()
sns.set_style('white')
colors_dict = {'CVAE': 'purple', 'GMM': 'blue', 'RMPP': 'green'}

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

portion = [8, 1, 1]

log_likelihoods_all = {}
mses_all = {}

methods = ['JVAE', 'CVAE', 'GMM']

for ratname in ratnames:

    log_likelihoods_all[ratname] = {}
    mses_all[ratname] = {}

    for method in methods:

        log_likelihoods = []
        mses = []

        for perm in perms:

            log_likelihood = np.load(os.path.join(ROOTDIR, 'results', method, ratname, ''.join(map(str, perm)), 'log_likelihood.npy'))
            log_likelihoods.append(log_likelihood[np.argmax(log_likelihood[:, 0]), 2])

            sep, ttilde, y, y_smoothed_means, y_smoothed_covs = np.load(os.path.join(ROOTDIR, 'results', method, ratname, ''.join(map(str, perm)),  'reconstruct_test.npy'), allow_pickle=True)
            mses.append(np.nanmean(np.square(y - y_smoothed_means)))

        log_likelihoods = np.array(log_likelihoods)
        log_likelihoods_all[ratname][method] = log_likelihoods

        mses = np.array(mses)
        mses_all[ratname][method] = mses


log_likelihoods_table = {key: pd.DataFrame(value) for key, value in log_likelihoods_all.items()}
mses_table = {key: pd.DataFrame(value) for key, value in mses_all.items()}


log_likelihoods_pvalues = {}
mses_pvalues = {}

for ratname in ratnames:

    mse_pvalues = []

    for pairs in itertools.product(mses_table[ratname].columns, repeat=2):
        if pairs[0] != pairs[1]:
            mse_pvalue = wilcoxon(mses_table[ratname][pairs[0]], mses_table[ratname][pairs[1]]).pvalue
        else:
            mse_pvalue = 0.
        mse_pvalues.append(mse_pvalue)

    mse_pvalues = np.array(mse_pvalues).reshape((len(mses_table[ratname].columns), -1))

    mse_pvalues = pd.DataFrame(mse_pvalues)
    mse_pvalues.columns = mses_table[ratname].columns
    mse_pvalues.index = mses_table[ratname].columns
    mses_pvalues[ratname] = mse_pvalues

    log_likelihood_pvalues = []

    for pairs in itertools.product(log_likelihoods_table[ratname].columns, repeat=2):
        if pairs[0] != pairs[1]:
            log_likelihood_pvalue = wilcoxon(log_likelihoods_table[ratname][pairs[0]], log_likelihoods_table[ratname][pairs[1]]).pvalue
        else:
            log_likelihood_pvalue = 0.
        log_likelihood_pvalues.append(log_likelihood_pvalue)

    log_likelihood_pvalues = np.array(log_likelihood_pvalues).reshape((len(log_likelihoods_table[ratname].columns), -1))
    log_likelihood_pvalues = pd.DataFrame(log_likelihood_pvalues)
    log_likelihood_pvalues.columns = log_likelihoods_table[ratname].columns
    log_likelihood_pvalues.index = log_likelihoods_table[ratname].columns

    log_likelihoods_pvalues[ratname] = log_likelihood_pvalues



# Make figure for NLL

fig = plt.figure(figsize=(12, 18))
subfigs = fig.subfigures(nrows=len(ratnames), ncols=1)

for subfig, ratname in zip(subfigs, ratnames):

    subfig.suptitle(ratname, fontweight='bold')

    tb = - log_likelihoods_table[ratname]
    methods = tb.columns

    tb = pd.concat([tb[method].subtract(tb[methods[0]]) for method in methods], axis=1)
    tb = tb[tb.columns[1:]]
    methods = methods[1:]
    tb.columns = methods

    ind = np.arange(len(methods))

    pvalues = log_likelihoods_pvalues[ratname]
    mask = np.zeros_like(pvalues, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    axs = subfig.subplots(nrows=1, ncols=2)

    for pos, method in zip(ind, methods):
        color = colors_dict[method]
        axs[0].scatter(pos, tb[method].mean(), c=color, marker='o')
        axs[0].scatter(pos * np.ones_like(tb[method].values), tb[method].values, c=color, marker='x', alpha=0.2)

    axs[0].axhline(0., c='k', alpha=0.5)

    labels = [item.get_text() for item in axs[0].get_xticklabels()]
    empty_string_labels = ['']*len(labels)
    axs[0].set_xticklabels(empty_string_labels)

    axs[0].set_xticks(ind)
    axs[0].set_xticklabels(methods)
    axs[0].set_xlim([ind[0]-0.5, ind[-1]+0.5])

    ylim = axs[0].get_ylim()
    axs[0].set_ylim([ylim[0], np.percentile(np.array(tb), 95)])
    axs[0].set_ylabel('Differnece of NLL')

    axs[0].yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))
    axs[0].ticklabel_format(style="sci", axis="y", scilimits=(3,3))

    axs[1].set_facecolor((0, 0, 0, 0.2))
    sns_plot = sns.heatmap(pvalues, mask=mask, annot=True, square=True, fmt=".3f", linewidths=.5, cmap="Reds_r", norm=LogNorm(vmin=1e-4, vmax=1), ax=axs[1])


fig.savefig(os.path.join(ROOTDIR, 'figures', 'placecell_nll.png'), bbox_inches='tight')


# Make figure for MSE

fig = plt.figure(figsize=(12, 18))
subfigs = fig.subfigures(nrows=len(ratnames), ncols=1)

for subfig, ratname in zip(subfigs, ratnames):

    subfig.suptitle(ratname, fontweight='bold')

    tb = mses_table[ratname]
    methods = tb.columns

    tb = pd.concat([tb[method].subtract(tb[methods[0]]) for method in methods], axis=1)
    tb = tb[tb.columns[1:]]
    methods = methods[1:]
    tb.columns = methods

    ind = np.arange(len(methods))

    pvalues = mses_pvalues[ratname]
    mask = np.zeros_like(pvalues, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    axs = subfig.subplots(nrows=1, ncols=2, width_ratios=[1., 1.])

    for pos, method in zip(ind, methods):
        color = colors_dict[method]
        axs[0].scatter(pos, tb[method].mean(), c=color, marker='o')
        axs[0].scatter(pos * np.ones_like(tb[method].values), tb[method].values, c=color, marker='x', alpha=0.2)

    axs[0].axhline(0., c='k', alpha=0.5)

    labels = [item.get_text() for item in axs[0].get_xticklabels()]
    empty_string_labels = ['']*len(labels)
    axs[0].set_xticklabels(empty_string_labels)

    axs[0].set_xticks(ind)
    axs[0].set_xticklabels(methods)
    axs[0].set_xlim([ind[0]-0.5, ind[-1]+0.5])

    ylim = axs[0].get_ylim()
    axs[0].set_ylim([ylim[0], np.percentile(np.array(tb), 95)])
    axs[0].set_ylabel('Differnece of MSE')

    axs[1].set_facecolor((0, 0, 0, 0.2))
    sns.heatmap(pvalues, mask=mask, annot=True, square=True, fmt=".3f", linewidths=.5, cmap="Reds_r", norm=LogNorm(vmin=1e-4, vmax=1), ax=axs[1])

fig.savefig(os.path.join(ROOTDIR, 'figures', 'placecell_mse.png'), bbox_inches='tight')
