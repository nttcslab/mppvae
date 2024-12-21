import argparse

try:
    parser = argparse.ArgumentParser()
    parser.add_argument('inds', nargs=2)
    args = parser.parse_args()
    name_ind, perm_ind = args.inds
    name_ind, perm_ind = int(name_ind), int(perm_ind)
except:
    name_ind, perm_ind = 0, 0

from environ import *
import os
os.chdir(ROOTDIR)
import numpy as np
np.set_printoptions(suppress=True)
import tensorflow as tf
import time
from tqdm import tqdm
from util import *
from model_JVAE import *

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

ratname, perm = ratnames[name_ind], perms[perm_ind]

portion = [8, 1, 1]

n_batch = 80
n_compile = 10
n_jackknife = 2
n_monte = 10
n_monte_eval = 100
n_repeat = 10

learning_rate=0.001
alpha_1, alpha_2 = 0., 2.
training_epochs=1000
eval_step=50
display_step=200

data = np.load(os.path.join(ROOTDIR, 'SPIKE_data', ratname, 'data.npy'), allow_pickle=True)[()]
spikeTrain, stim, sep = data['spikeTrain'], data['stim'], data['sep']
spikeTrain, stim, stim_mean, stim_cov = normalize_crcns(spikeTrain, stim, sep)

n_params = {'n_site': len(spikeTrain), 'n_y': 2, 'n_kappa': [spike.shape[1] - 1 for spike in spikeTrain]}
n_kappa_max = np.max(n_params['n_kappa'])
spikeTrain = [np.pad(spike, [[0, 0], [0, n_kappa_max - spike.shape[1] + 1]]) for spike in spikeTrain]

dirname = os.path.join(ROOTDIR, 'results', 'JVAE', ratname, ''.join(map(str, perm)))
print('---------------------')
print(dirname)
print('---------------------')
os.makedirs(dirname, exist_ok=True)

dataSet = makeDataSet(spikeTrain, stim)
trainDataSet, trainStimSet, trainSepSet, log_scale_train, valDataSet, valStimSet, valSepSet, log_scale_val, testDataSet, testStimSet, testSepSet, log_scale_test = splitDataSet(dataSet, stim, sep, portion, perm)

batch_ts_train, batch_inds_train, batch_kappas_train, batch_ttildes_train, batch_ys_train, batch_seps_train = splitDataBatch(trainDataSet, trainStimSet, trainSepSet, n_batch, n_params)
batch_ts_val, batch_inds_val, batch_kappas_val, batch_ttildes_val, batch_ys_val, batch_seps_val = splitDataBatch(valDataSet, valStimSet, valSepSet, 1, n_params)
batch_ts_test, batch_inds_test, batch_kappas_test, batch_ttildes_test, batch_ys_test, batch_seps_test = splitDataBatch(testDataSet, testStimSet, testSepSet, 1, n_params)

batch_ts_padded_train, batch_kappas_padded_train, batch_masks_train, batch_ttildes_padded_train, batch_ys_padded_train = padDataBatch(batch_ts_train, batch_inds_train, batch_kappas_train, batch_ttildes_train, batch_ys_train)
batch_ts_padded_val, batch_kappas_padded_val, batch_masks_val, batch_ttildes_padded_val, batch_ys_padded_val = padDataBatch(batch_ts_val, batch_inds_val, batch_kappas_val, batch_ttildes_val, batch_ys_val)
batch_ts_padded_test, batch_kappas_padded_test, batch_masks_test, batch_ttildes_padded_test, batch_ys_padded_test = padDataBatch(batch_ts_test, batch_inds_test, batch_kappas_test, batch_ttildes_test, batch_ys_test)

del trainDataSet, trainStimSet, valDataSet, valStimSet, testDataSet, testStimSet, dataSet
del spikeTrain, stim

n_params.update({'n_recog': [50, 50, 50, 50], 'n_recog_g': [50, 50, 50, 50], 'n_gener': [50, 50, 50, 50], 'n_gener_x': [50, 50, 50, 50], 'n_gener_kappa': [50, 50, 50, 50],
                 'n_recog_y': [50, 50, 50, 50], 'n_gener_y': [50, 50, 50, 50], 'n_recog_kappa': [50, 50, 50, 50],
                 'n_z': 10, 'n_x': 10})

n_params.update({'n_batch': n_batch})
n_params.update({'alpha_1': alpha_1, 'alpha_2': alpha_2})
n_params.update({'learning_rate': learning_rate})
n_params.update({'n_compile': n_compile})
n_params.update({'n_jackknife': n_jackknife})

vae = JVAE(n_params)
vae.assign_log_lambda(batch_kappas_padded_train, batch_masks_train, batch_ys_padded_train, n_monte)
vae.save_weights(os.path.join(dirname, 'model' + str(0)))

log_likelihoods_val, log_x_likelihoods_val = [], []
log_likelihoods_test, log_x_likelihoods_test = [], []

best_log_likelihood_val = - np.inf
avg_cost = 0.

summary_writer = tf.summary.create_file_writer(os.path.join(dirname, 'runs', time.strftime("%Y%m%d-%H%M%S")))

for epoch in tqdm(range(training_epochs)):

    shuffle_ind = tf.random.shuffle(tf.range(n_batch))
    batch_kappas_padded_train, batch_masks_train, batch_ys_padded_train = tf.gather(batch_kappas_padded_train, shuffle_ind), tf.gather(batch_masks_train, shuffle_ind), tf.gather(batch_ys_padded_train, shuffle_ind)
    avg_cost_epoch = vae.train(batch_kappas_padded_train, batch_masks_train, batch_ys_padded_train, n_monte)
    avg_cost += avg_cost_epoch

    if (epoch + 1) % eval_step == 0:

        avg_cost = avg_cost / eval_step
        print('Epoch:', '%04d' % (epoch + 1), '|', 'cost:', '{:.3f}'.format(avg_cost))
        avg_cost = 0.

        if np.isinf(avg_cost_epoch) or np.isnan(avg_cost_epoch):
            vae.load_weights(os.path.join(dirname, 'model' + str(epoch+1-50)))

        kappas, masks, y = batch_kappas_padded_val[0], batch_masks_val[0], batch_ys_padded_val[0]

        log_likelihood_val = np.mean([vae.eval_log_likelihood(kappas, masks, y, log_scale_train, log_scale_val, n_monte_eval) for _ in range(n_repeat)])
        log_x_likelihood_val = np.mean([vae.eval_log_likelihood_x(kappas, masks, y, log_scale_train, log_scale_val, n_monte_eval) for _ in range(n_repeat)])


        log_likelihoods_val.append(log_likelihood_val)
        log_x_likelihoods_val.append(log_x_likelihood_val)

        kappas, masks, y =  batch_kappas_padded_test[0], batch_masks_test[0], batch_ys_padded_test[0]

        log_likelihood_test = np.mean([vae.eval_log_likelihood(kappas, masks, y, log_scale_train, log_scale_test, n_monte_eval) for _ in range(n_repeat)])
        log_x_likelihood_test = np.mean([vae.eval_log_likelihood_x(kappas, masks, y, log_scale_train, log_scale_test, n_monte_eval) for _ in range(n_repeat)])

        log_likelihoods_test.append(log_likelihood_test)
        log_x_likelihoods_test.append(log_x_likelihood_test)

        if log_likelihood_val > best_log_likelihood_val:
            vae.save_weights(os.path.join(dirname, 'best_model'))
            best_log_likelihood_val = log_likelihood_val

        vae.save_weights(os.path.join(dirname, 'model' + str(epoch + 1)))

        print('Log likelihood val:', '{:.3f}'.format(log_likelihood_val), '|',  'Best:', '{:.3f}'.format(best_log_likelihood_val))
        print('Log likelihood x val:', '{:.3f}'.format(log_x_likelihood_val))
        print('Log likelihood test:', '{:.3f}'.format(log_likelihood_test))

        with summary_writer.as_default():
            tf.summary.scalar('log_likelihood_val', log_likelihood_val, step=epoch+1)
            tf.summary.scalar('log_likelihood_test', log_likelihood_test, step=epoch+1)

            tf.summary.scalar('log_likelihood_x_val', log_x_likelihood_val, step=epoch+1)
            tf.summary.scalar('log_likelihood_x_test', log_x_likelihood_test, step=epoch+1)


np.save(os.path.join(dirname, 'log_likelihood.npy'), np.vstack([log_likelihoods_val, log_x_likelihoods_val, log_likelihoods_test, log_x_likelihoods_test]).T)
vae.load_weights(os.path.join(dirname, 'best_model'))

sep, kappas, masks, ttilde, y = batch_seps_val[0], batch_kappas_padded_val[0], batch_masks_val[0], batch_ttildes_padded_val[0], batch_ys_padded_val[0]
y_smoothed_means, y_smoothed_covs, x_smoothed_means, x_smoothed_covs = vae.decode(kappas, masks, log_scale_train, log_scale_val, n_monte_eval, 5, 100)
y, y_smoothed_means, y_smoothed_covs = denormalize(y, y_smoothed_means, y_smoothed_covs, stim_mean, stim_cov)
np.save(os.path.join(dirname, 'reconstruct_val.npy'), np.array([sep, ttilde, y, y_smoothed_means, y_smoothed_covs], dtype=object))
plotReconstructResult(dirname, 'val.png', sep, ttilde, y, y_smoothed_means, y_smoothed_covs)

sep, kappas, masks, ttilde, y = batch_seps_test[0], batch_kappas_padded_test[0], batch_masks_test[0], batch_ttildes_padded_test[0], batch_ys_padded_test[0]
y_smoothed_means, y_smoothed_covs, x_smoothed_means, x_smoothed_covs = vae.decode(kappas, masks, log_scale_train, log_scale_test, n_monte_eval, 5, 100)
y, y_smoothed_means, y_smoothed_covs = denormalize(y, y_smoothed_means, y_smoothed_covs, stim_mean, stim_cov)
np.save(os.path.join(dirname, 'reconstruct_test.npy'), np.array([sep, ttilde, y, y_smoothed_means, y_smoothed_covs], dtype=object))
plotReconstructResult(dirname, 'test.png', sep, ttilde, y, y_smoothed_means, y_smoothed_covs)
