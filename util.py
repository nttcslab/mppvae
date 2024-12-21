import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.stats import chi
import os

"""
Utility functions
"""

def normalize(spikeTrain, stim, sep):

    spikeTrain_normalized = []

    for spike in spikeTrain:

        spike = spike[(sep[0] <= spike[:, 0]) * (spike[:, 0] <= sep[1])]

        t, fet = spike[:, :1], spike[:, 1:]

        is_not_nan = np.logical_not(np.any(np.isnan(fet), axis=-1))

        fet_mean, fet_cov = np.mean(fet[is_not_nan], axis=0), np.cov(fet[is_not_nan], rowvar=0)
        e, v = np.linalg.eigh(fet_cov)

        fet_normalized = (fet - fet_mean).dot(v) / e ** 0.5
        spike_normalized = np.hstack([t, fet_normalized])
        spikeTrain_normalized.append(spike_normalized)

    stim = stim[(sep[0] < stim[:, 0]) * (stim[:, 0] < sep[1])]
    t, fet = stim[:, :1], stim[:, 1:]

    is_not_nan = np.logical_not(np.any(np.isnan(fet), axis=-1))

    fet_mean, fet_cov = np.mean(fet[is_not_nan], axis=0), np.cov(fet[is_not_nan], rowvar=0)
    e, v = np.linalg.eigh(fet_cov)

    fet_normalized = (fet - fet_mean).dot(v) / e ** 0.5
    stim_normalized = np.hstack([t, fet_normalized])

    return spikeTrain_normalized, stim_normalized, fet_mean, fet_cov


def normalize_crcns(spikeTrain, stim, sep):

    spikeTrain_normalized = []

    for spike in spikeTrain:

        spike = spike[(sep[0] <= spike[:, 0]) * (spike[:, 0] <= sep[1])]

        t, fet = spike[:, :1], spike[:, 1:]
        n, length, channel = fet.shape[0], 32, fet.shape[1] // 32

        const = np.eye(channel)[:, np.newaxis] * np.ones(length)[:, np.newaxis] / np.sqrt(length)
        trend = np.eye(channel)[:, np.newaxis] * np.linspace(-1, 1, length)[:, np.newaxis] / np.linalg.norm(np.linspace(-1, 1, length))
        basis = np.vstack([const, trend]).reshape((-1, length*channel))

        fet -= np.matmul((fet[:, np.newaxis] * basis[np.newaxis]).sum(-1), basis)

        is_not_nan = np.logical_not(np.any(np.isnan(fet), axis=-1))
        fet_mean, fet_cov = np.mean(fet[is_not_nan], axis=0), np.cov(fet[is_not_nan], rowvar=0)
        e, v = np.linalg.eigh(fet_cov)

        fet_normalized = (fet - fet_mean).dot(v[:, 2*channel:]) / e[2*channel:] ** 0.5

        spike_normalized = np.hstack([t, fet_normalized])
        spikeTrain_normalized.append(spike_normalized)

    stim = stim[(sep[0] < stim[:, 0]) * (stim[:, 0] < sep[1])]
    t, fet = stim[:, :1], stim[:, 1:]

    is_not_nan = np.logical_not(np.any(np.isnan(fet), axis=-1))

    fet_mean, fet_cov = np.mean(fet[is_not_nan], axis=0), np.cov(fet[is_not_nan], rowvar=0)
    e, v = np.linalg.eigh(fet_cov)

    fet_normalized = (fet - fet_mean).dot(v) / e ** 0.5
    stim_normalized = np.hstack([t, fet_normalized])

    return spikeTrain_normalized, stim_normalized, fet_mean, fet_cov



def denormalize(y, y_smoothed_means, y_smoothed_covs, stim_mean, stim_cov):
    e, v = np.linalg.eigh(tf.constant(stim_cov, tf.float32))
    y_denormalized, y_smoothed_means_denormalized, y_smoothed_covs_denormalized = tf.matmul(y * e ** 0.5, tf.linalg.matrix_transpose(v)) + stim_mean, tf.matmul(y_smoothed_means * e ** 0.5, tf.linalg.matrix_transpose(v)) + stim_mean, tf.matmul(v, tf.matmul(y_smoothed_covs * (e[:, tf.newaxis] * e) ** 0.5, tf.linalg.matrix_transpose(v)))
    return y_denormalized, y_smoothed_means_denormalized, y_smoothed_covs_denormalized


def makeDataSet(spikeTrain, stim):
    """ Make data set
    """
    shankNumber = len(spikeTrain)
    stimAtSpikes = [interp1d(stim[:, 0], stim[:, 1:], axis=0, kind='previous', fill_value='extrapolate')(spikeTrain[i][:, 0]) for i in range(shankNumber)]
    dataSet = [np.hstack([spikeTrain[i], stimAtSpikes[i]]) for i in range(shankNumber)]
    return dataSet


def splitDataSet(dataSet, stim, sep, portion, perm):
    """ Split data set into train data and test data
    """

    batch_sep = np.linspace(sep[0], sep[1], sum(portion) + 1)
    stimBatch = [stim[(batch_sep[i] <= stim[:, 0]) * (stim[:, 0] < batch_sep[i+1])] for i in range(sum(portion))]
    sepBatch = [[batch_sep[i], batch_sep[i+1]] for i in range(sum(portion))]

    trainInd = perm[:portion[0]]
    valInd = perm[portion[0]:portion[0]+portion[1]]
    testInd = perm[portion[0]+portion[1]:portion[0]+portion[1]+portion[2]]

    trainStimSet, valStimSet, testStimSet = [stimBatch[i] for i in trainInd], [stimBatch[i] for i in valInd], [stimBatch[i] for i in testInd]
    trainSepSet, valSepSet, testSepSet = [sepBatch[i] for i in trainInd], [sepBatch[i] for i in valInd], [sepBatch[i] for i in testInd]

    m_train, T_train = sum([trainStim.shape[0] for trainStim in trainStimSet]), (sep[1] - sep[0]) * portion[0] / sum(portion)
    m_val, T_val = sum([valStim.shape[0] for valStim in valStimSet]), (sep[1] - sep[0]) * portion[1] / sum(portion)
    m_test, T_test = sum([testStim.shape[0] for testStim in testStimSet]), (sep[1] - sep[0]) * portion[2] / sum(portion)

    log_scale_train, log_scale_val, log_scale_test = np.log(m_train / T_train, dtype=np.float32), np.log(m_val / T_val, dtype=np.float32), np.log(m_test / T_test, dtype=np.float32)

    trainDataSet, valDataSet, testDataSet = [], [], []

    for data in dataSet:
        dataBatch = [data[(batch_sep[i] <= data[:, 0]) * (data[:, 0] < batch_sep[i+1])] for i in range(sum(portion))]
        trainDataSet.append([dataBatch[i] for i in trainInd]), valDataSet.append([dataBatch[i] for i in valInd]), testDataSet.append([dataBatch[i] for i in testInd])

    return (trainDataSet, trainStimSet, trainSepSet, log_scale_train, valDataSet, valStimSet, valSepSet, log_scale_val, testDataSet, testStimSet, testSepSet, log_scale_test)


def splitDataBatch(dataSets, stimSet, sepSet, n_batch, n_params):

    batch_ttildes, batch_ys, batch_seps = [], [], []
    batch_num = n_batch // len(dataSets[0])

    for stim, sep in zip(stimSet, sepSet):

        mask = np.logical_not(np.any(np.isnan(stim), 1))
        stim_interp = np.hstack([stim[:, :1], interp1d(stim[mask, 0], stim[mask, 1:], kind='linear', axis=0, fill_value='extrapolate')(stim[:, 0])])

        batch_sep = np.linspace(sep[0], sep[1], batch_num + 1)

        for j in range(batch_num):
            batch_start, batch_end = batch_sep[j], batch_sep[j+1]
            batch_stim = stim[(batch_start <= stim[:, 0]) * (stim[:, 0] < batch_end)]
            batch_stim_interp = stim_interp[(batch_start <= stim[:, 0]) * (stim[:, 0] < batch_end)]

            batch_stim_index = batch_stim[:-1, :1]
            batch_stim_value = batch_stim[:-1, 1:]

            if np.any(np.isnan(batch_stim_value[0])):
                batch_stim_value[0] = batch_stim_interp[0, 1:]
            if np.any(np.isnan(batch_stim_value[-1])):
                batch_stim_value[-1] = batch_stim_interp[-1, 1:]

            batch_stim_index = tf.constant(batch_stim_index, tf.float32)
            batch_stim_value = tf.constant(batch_stim_value[:, np.newaxis], tf.float32)

            batch_ttildes.append(batch_stim_index)
            batch_ys.append(batch_stim_value)

            batch_seps.append([batch_start, batch_end])

    batch_ts, batch_inds, batch_kappas = [], [], []

    for dataSet, n_kappa in zip(dataSets, n_params['n_kappa']):

        batch_ts_sub, batch_inds_sub, batch_kappas_sub = [], [], []

        for data, stim, sep in zip(dataSet, stimSet, sepSet):

            batch_sep = np.linspace(sep[0], sep[1], batch_num + 1)

            for j in range(batch_num):

                batch_start, batch_end = batch_sep[j], batch_sep[j+1]
                batch_stim = stim[(batch_start <= stim[:, 0]) * (stim[:, 0] < batch_end)]
                batch_data = tf.constant(data[(batch_stim[0, 0] <= data[:, 0]) * (data[:, 0] < batch_stim[-1, 0])], tf.float32)

                batch_count = [data[(batch_stim[r, 0] <= data[:, 0]) * (data[:, 0] < batch_stim[r+1, 0])].shape[0] for r in range(batch_stim.shape[0]-1)]
                batch_ind = tf.repeat(tf.range(batch_stim.shape[0]-1, dtype=tf.int32), batch_count)

                batch_ts_sub.append(batch_data[:, :1])
                batch_inds_sub.append(tf.repeat(tf.range(batch_stim.shape[0]-1, dtype=tf.int32), batch_count))

                batch_kappas_sub.append(batch_data[:, 1:-n_params['n_y']])


        batch_ts.append(batch_ts_sub), batch_inds.append(batch_inds_sub), batch_kappas.append(batch_kappas_sub)

    return list(zip(*batch_ts)), list(zip(*batch_inds)), list(zip(*batch_kappas)), batch_ttildes, batch_ys, batch_seps

def padDataBatch(batch_ts, batch_inds, batch_kappas, batch_ttildes, batch_ys):

    R_max = tf.reduce_max([tf.shape(batch_y)[0] for batch_y in batch_ys])
    batch_ttildes_padded, batch_ys_padded = [], []

    for batch_ttilde, batch_y in zip(batch_ttildes, batch_ys):

        R = tf.shape(batch_y)[0]
        batch_ttildes_padded.append(tf.concat([batch_ttilde, tf.repeat(batch_ttilde[-1:], R_max-R, axis=0)], axis=0))
        batch_ys_padded.append(tf.concat([batch_y, tf.repeat(batch_y[-1:], R_max-R, axis=0)], axis=0))

    batch_ttildes_padded, batch_ys_padded = tf.stack(batch_ttildes_padded), tf.stack(batch_ys_padded)

    n_max = tf.reduce_max([[tf.shape(kappa)[0] for kappa in kappas] for kappas in  batch_kappas])
    batch_ts_padded, batch_kappas_padded, batch_masks = [], [], []

    for batch_ts_sub, batch_inds_sub, batch_kappas_sub in zip(list(zip(*batch_ts)), list(zip(*batch_inds)), list(zip(*batch_kappas))):

        batch_is_not_nans_sub = [tf.math.logical_not(tf.math.reduce_any(tf.math.is_nan(batch_kappa), axis=1)) for batch_kappa in batch_kappas_sub]
        batch_ts_padded_sub, batch_kappas_padded_sub, batch_masks_sub = [], [], []

        kappa_sub = tf.concat(batch_kappas_sub, 0)
        n_all = tf.shape(kappa_sub)[0]

        for batch_t, batch_ind, batch_kappa, batch_is_not_nan in zip(batch_ts_sub, batch_inds_sub, batch_kappas_sub, batch_is_not_nans_sub):

            n = tf.shape(batch_ind[batch_is_not_nan])[0]

            batch_ts_padded_sub.append(tf.pad(batch_t[batch_is_not_nan], [[0, n_max-n], [0, 0]]))
            batch_kappas_padded_sub.append(tf.pad(batch_kappa[batch_is_not_nan], [[0, n_max-n], [0, 0]], constant_values=0)[:, tf.newaxis])
            batch_masks_sub.append(tf.pad(tf.cast(batch_ind[batch_is_not_nan][:, tf.newaxis] == tf.range(R_max), tf.float32), [[0, n_max-n], [0, 0]]))

        batch_ts_padded_sub, batch_kappas_padded_sub, batch_masks_sub = tf.stack(batch_ts_padded_sub), tf.stack(batch_kappas_padded_sub), tf.stack(batch_masks_sub)
        batch_ts_padded.append(batch_ts_padded_sub), batch_kappas_padded.append(batch_kappas_padded_sub), batch_masks.append(batch_masks_sub)

    batch_ts_padded, batch_kappas_padded, batch_masks = tf.concat(batch_ts_padded, axis=2), tf.concat(batch_kappas_padded, axis=2), tf.stack(batch_masks, axis=3)

    return  batch_ts_padded, batch_kappas_padded, batch_masks, batch_ttildes_padded, batch_ys_padded


def plotReconstructResult(dirname, filename, sep, ttilde, y, y_smoothed_means, y_smoothed_covs):

    # 1D

    fig = plt.figure(figsize=(12, 6), tight_layout=True)

    plt.plot(ttilde[:, 0], y_smoothed_means[:, 0, 0])
    plt.plot(ttilde[:, 0], y_smoothed_means[:, 0, 1])

    plt.fill_between(ttilde[:, 0], y_smoothed_means[:, 0, 0] - 2. * y_smoothed_covs[:, 0, 0] ** 0.5, y_smoothed_means[:, 0, 0] + 2. * y_smoothed_covs[:, 0, 0] ** 0.5, alpha=0.3)
    plt.fill_between(ttilde[:, 0], y_smoothed_means[:, 0, 1] - 2. * y_smoothed_covs[:, 1, 1] ** 0.5, y_smoothed_means[:, 0, 1] + 2. * y_smoothed_covs[:, 1, 1] ** 0.5, alpha=0.3)

    plt.plot(ttilde[:, 0], y[:, 0, 0])
    plt.plot(ttilde[:, 0], y[:, 0, 1])

    plt.xlim(sep)

    plt.savefig(os.path.join(dirname, filename), bbox_inches='tight')
    plt.close()

    # 2D

    filename = filename.split('.')[0] + '_2D.' + filename.split('.')[1]

    fig = plt.figure(figsize=(6, 6), tight_layout=True)

    plt.plot(y_smoothed_means[:, 0, 0], y_smoothed_means[:, 0, 1], alpha=0.5)
    plt.plot(y[:, 0, 0], y[:, 0, 1], alpha=0.5)

    plt.savefig(os.path.join(dirname, filename))
    plt.close()



def plotTensorBoard(summary_writer, tagname, epoch, sep, ttilde, y, y_smoothed_means, y_smoothed_covs):

    # 1D

    fig = plt.figure(figsize=(12, 6), tight_layout=True)

    plt.plot(ttilde[:, 0], y_smoothed_means[:, 0, 0])
    plt.plot(ttilde[:, 0], y_smoothed_means[:, 0, 1])

    plt.fill_between(ttilde[:, 0], y_smoothed_means[:, 0, 0] - 2. * y_smoothed_covs[:, 0, 0] ** 0.5, y_smoothed_means[:, 0, 0] + 2. * y_smoothed_covs[:, 0, 0] ** 0.5, alpha=0.3)
    plt.fill_between(ttilde[:, 0], y_smoothed_means[:, 0, 1] - 2. * y_smoothed_covs[:, 1, 1] ** 0.5, y_smoothed_means[:, 0, 1] + 2. * y_smoothed_covs[:, 1, 1] ** 0.5, alpha=0.3)

    plt.plot(ttilde[:, 0], y[:, 0, 0])
    plt.plot(ttilde[:, 0], y[:, 0, 1])

    plt.xlim(sep)

    fig.canvas.draw()
    plot_image_array = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]

    with summary_writer.as_default():
        tf.summary.image(tagname, plot_image_array[np.newaxis, :, :, :3], step=epoch+1)

    # 2D

    tagname = tagname + '_2D'

    fig = plt.figure(figsize=(6, 6), tight_layout=True)

    plt.plot(y_smoothed_means[:, 0, 0], y_smoothed_means[:, 0, 1], alpha=0.5)
    plt.plot(y[:, 0, 0], y[:, 0, 1], alpha=0.5)

    fig.canvas.draw()
    plot_image_array = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]

    with summary_writer.as_default():
        tf.summary.image(tagname, plot_image_array[np.newaxis, :, :, :3], step=epoch+1)
