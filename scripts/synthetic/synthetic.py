from environ import *
import os
os.chdir(ROOTDIR)
os.makedirs(os.path.join(ROOTDIR, 'SPIKE_data', 'Lorenz_3D'), exist_ok=True)
os.makedirs(os.path.join(ROOTDIR, 'SPIKE_data', 'Lorenz_2D'), exist_ok=True)
import numpy as np
import tensorflow as tf
from scipy.stats import poisson, uniform, multivariate_normal
from sklearn.decomposition import PCA
import tensorflow_probability as tfp
from tqdm import tqdm

# Calculate discrete solution of Lorenz attractor

def lorenz(x, s, r, b, eps, c):
    x1, x2, x3 = x[0], x[1], x[2]
    dx = c * np.array([s * (x2 - eps * x1), r * x1 - eps * x2 - x1 * x3, x1 * x2 - eps * b * x3])
    return dx

s, r, b, eps, c = 10, 28, 8/3, 1., 2.

n_batch = 10
T = 100
dt = 0.01
num_steps = n_batch * int(T / dt)

xs = np.empty((num_steps + 1, 3))
xs[0] = np.array([0., 1., 1.05])

for i in range(num_steps):

    k1 = lorenz(xs[i], s, r, b, eps, c)
    k2 = lorenz(xs[i] + 0.5 * dt * k1, s, r, b, eps, c)
    k3 = lorenz(xs[i] + 0.5 * dt * k2, s, r, b, eps, c)
    k4 = lorenz(xs[i] + dt * k3, s, r, b, eps, c)

    xs[i + 1] = xs[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)

stim = np.hstack([np.arange(num_steps)[:, np.newaxis] * dt, xs[:-1]])

center = np.array([[np.sqrt((r - 1) * b), - np.sqrt((r - 1) * b)], [np.sqrt((r - 1) * b), - np.sqrt((r - 1) * b)], [r - 1, r - 1]]).T

stim_mean, stim_std = stim[:, 1:].mean(axis=0), stim[:, 1:].std(axis=0)
stim = np.hstack([stim[:, :1], (stim[:, 1:] - stim_mean) / stim_std])
center = (center - stim_mean) / stim_std

batchs_stim = []

for l in range(n_batch):
    batch_stim = stim[int(T / dt)*l:int(T / dt)*(l+1), :]
    batch_stim = np.hstack([batch_stim[:, :1] - batch_stim[0, 0], batch_stim[:, 1:]])

    batchs_stim.append(batch_stim)



# Define the favorable direction of each component

ts_upsampled = np.linspace(stim[0, 0], stim[-1, 0], stim.shape[0] * 10)
stim_upsampled = np.vstack([np.interp(ts_upsampled, stim[:, 0], stim[:, i]) for i in range(stim.shape[-1])]).T

K = 16

ind = [1, 2]

threshold = 0.01
coordinates = []
coordinates_max = []
coordinates_min = []

rad_min = np.linspace(- 0.6 * np.pi, 0.8 * np.pi, K//2)
rad_max = np.linspace(- 0.4 * np.pi, 1.0 * np.pi, K//2)
rad = 0.5 * rad_min + 0.5 * rad_max

circ = np.vstack([np.cos(rad), np.sin(rad)])
circ_min = np.vstack([np.cos(rad_min), np.sin(rad_min)])
circ_max = np.vstack([np.cos(rad_max), np.sin(rad_max)])

mask = stim_upsampled[:, 1] > 0.
stim_masked = stim_upsampled[mask]
stim_masked_centerd = stim_masked[:, 1:] - center[0]

dist = np.linalg.norm((stim_masked_centerd[:, ind] / np.linalg.norm(stim_masked_centerd[:, ind], axis=-1, keepdims=True))[:, :, np.newaxis] - circ, axis=1)
dist_min = np.linalg.norm((stim_masked_centerd[:, ind] / np.linalg.norm(stim_masked_centerd[:, ind], axis=-1, keepdims=True))[:, :, np.newaxis] - circ_min, axis=1)
dist_max = np.linalg.norm((stim_masked_centerd[:, ind] / np.linalg.norm(stim_masked_centerd[:, ind], axis=-1, keepdims=True))[:, :, np.newaxis] - circ_max, axis=1)

for k in range(rad.shape[0]):
    stim_masked_centerd_dir = stim_masked_centerd[dist[:, k] < threshold]
    stim_masked_centerd_dir_min = stim_masked_centerd[dist_min[:, k] < threshold]
    stim_masked_centerd_dir_max = stim_masked_centerd[dist_max[:, k] < threshold]

    coordinates.append(stim_masked_centerd_dir[np.argmax(np.linalg.norm(stim_masked_centerd_dir, axis=-1))] + center[0])
    coordinates_min.append(stim_masked_centerd_dir_min[np.argmax(np.linalg.norm(stim_masked_centerd_dir_min, axis=-1))] + center[0])
    coordinates_max.append(stim_masked_centerd_dir_max[np.argmax(np.linalg.norm(stim_masked_centerd_dir_max, axis=-1))] + center[0])

rad_min = rad_min[::-1]
rad_max = rad_max[::-1]
rad = rad[::-1]

circ = np.vstack([np.cos(np.pi-rad), np.sin(np.pi-rad)])
circ_min = np.vstack([np.cos(np.pi-rad_max), np.sin(np.pi-rad_max)])
circ_max = np.vstack([np.cos(np.pi-rad_min), np.sin(np.pi-rad_min)])

mask = stim_upsampled[:, 1] < 0.
stim_masked = stim_upsampled[mask]
stim_masked_centerd = stim_masked[:, 1:] - center[1]

dist = np.linalg.norm((stim_masked_centerd[:, ind] / np.linalg.norm(stim_masked_centerd[:, ind], axis=-1, keepdims=True))[:, :, np.newaxis] - circ, axis=1)
dist_min = np.linalg.norm((stim_masked_centerd[:, ind] / np.linalg.norm(stim_masked_centerd[:, ind], axis=-1, keepdims=True))[:, :, np.newaxis] - circ_min, axis=1)
dist_max = np.linalg.norm((stim_masked_centerd[:, ind] / np.linalg.norm(stim_masked_centerd[:, ind], axis=-1, keepdims=True))[:, :, np.newaxis] - circ_max, axis=1)

for k in range(rad.shape[0]):
    stim_masked_centerd_dir = stim_masked_centerd[dist[:, k] < threshold]
    stim_masked_centerd_dir_min = stim_masked_centerd[dist_min[:, k] < threshold]
    stim_masked_centerd_dir_max = stim_masked_centerd[dist_max[:, k] < threshold]

    coordinates.append(stim_masked_centerd_dir[np.argmax(np.linalg.norm(stim_masked_centerd_dir, axis=-1))] + center[1])
    coordinates_min.append(stim_masked_centerd_dir_min[np.argmax(np.linalg.norm(stim_masked_centerd_dir_min, axis=-1))] + center[1])
    coordinates_max.append(stim_masked_centerd_dir_max[np.argmax(np.linalg.norm(stim_masked_centerd_dir_max, axis=-1))] + center[1])

coordinates = np.vstack(coordinates)
coordinates = coordinates.reshape((4, -1, 3)).transpose((1, 0, 2)).reshape((-1, 3))
coordinates = coordinates / np.linalg.norm(coordinates, axis=-1, keepdims=True)

coordinates_min = np.vstack(coordinates_min)
coordinates_min = coordinates_min.reshape((4, -1, 3)).transpose((1, 0, 2)).reshape((-1, 3))
coordinates_min = coordinates_min / np.linalg.norm(coordinates_min, axis=-1, keepdims=True)

coordinates_max = np.vstack(coordinates_max)
coordinates_max = coordinates_max.reshape((4, -1, 3)).transpose((1, 0, 2)).reshape((-1, 3))
coordinates_max = coordinates_max / np.linalg.norm(coordinates_max, axis=-1, keepdims=True)

# Load waveform samples from CRCNS dataset

ratname = 'Achilles_11012013'
data = np.load(os.path.join('SPIKE_data', ratname, 'data.npy'), allow_pickle=True)[()]
spikeTrain, stim, sep = data['spikeTrain'], data['stim'], data['sep']
site = 3

waveforms = []

for spike in spikeTrain:


    n_kappa_length, n_kappa_ch = 32, spike.shape[-1] // 32

    waveform = spike[:, 1:].reshape((-1, n_kappa_length, n_kappa_ch))[:, :, site]
    waveform = (waveform - waveform.mean())/ waveform.std()
    np.random.shuffle(waveform)

    waveforms.append(waveform[:5000])

waveforms = np.vstack(waveforms)

# Estimate parameters of decoder for mark distribution


loc = tf.constant(coordinates, tf.float32)
scale = tf.ones_like(coordinates, tf.float32) * 0.2

n_z, n_kappa = 3, waveforms.shape[1]

prior = tfp.distributions.MixtureSameFamily(mixture_distribution=tfp.distributions.Categorical(probs=tf.ones(K) / K), components_distribution=tfp.distributions.MultivariateNormalDiag(loc=loc, scale_diag=scale))

encoder = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=n_kappa),
                          tf.keras.layers.Dense(100, activation=tf.nn.tanh),
                          tf.keras.layers.Dense(100, activation=tf.nn.tanh),
                          tf.keras.layers.Dense(100, activation=tf.nn.tanh),
                          tf.keras.layers.Dense(100, activation=tf.nn.tanh),
                          tf.keras.layers.Dense(tfp.layers.MultivariateNormalTriL.params_size(n_z), activation=None),
                          tfp.layers.MultivariateNormalTriL(n_z)])


decoder = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=[n_z]),
                          tf.keras.layers.Dense(100, activation=tf.nn.tanh),
                          tf.keras.layers.Dense(100, activation=tf.nn.tanh),
                          tf.keras.layers.Dense(100, activation=tf.nn.tanh),
                          tf.keras.layers.Dense(100, activation=tf.nn.tanh),
                          tf.keras.layers.Dense(tfp.layers.IndependentNormal.params_size(n_kappa), activation=None),
                          tfp.layers.IndependentNormal(n_kappa)])

def cost():
    z = encoder(waveforms).sample()
    return  - tf.reduce_sum(decoder(z).log_prob(waveforms)) - tf.reduce_sum(prior.log_prob(z)) + tf.reduce_sum(encoder(waveforms).log_prob(z))

@tf.function(experimental_compile=True)
def update():
    optimizer.minimize(cost, variables)
    return cost()

variables = lambda: encoder.trainable_variables + decoder.trainable_variables
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

training_epochs = 5000

for epoch in tqdm(range(training_epochs)):
    cost_epoch = update()

decoder.save_weights(os.path.join(ROOTDIR, 'SPIKE_data', 'Lorenz3D', 'decoder'))



# Generate unsorted spieks

n_site = 4
order = 100
const = 0.
log_scale = 12

for l, batch_stim in zip(range(10), batchs_stim):

    ratio = batch_stim[:, 0] / T

    coordinates_raw = coordinates_max * ratio[:, np.newaxis, np.newaxis] + coordinates_min * (1 - ratio)[:, np.newaxis, np.newaxis]
    coordinates_raw = coordinates_raw / tf.linalg.norm(coordinates_raw, axis=-1, keepdims=True)

    values = const + order * tf.reduce_sum(coordinates_raw * batch_stim[:, np.newaxis, 1:4], -1) / tf.linalg.norm(batch_stim[:, 1:4], axis=-1, keepdims=True)

    log_lambda = log_scale - tf.reduce_logsumexp(values, 0)
    log_intensity = log_lambda + values

    N = poisson(np.exp(log_intensity)*dt).rvs()

    spikes = []
    covariates = []

    for k in range(K):

        ind = []
        for n in range(1, N[:, k].max()+1):
            ind.append(np.repeat(np.where(N[:, k] == n)[0], n))

        ind = np.sort(np.hstack(ind))
        t = ind * dt + dt * uniform().rvs(ind.shape[0])

        covariate = batch_stim[np.floor(t / dt).astype(int), 1:]
        covariates.append(covariate)

        kappa = multivariate_normal(loc[k], scale[k] ** 2 * np.eye(n_z)).rvs(t.shape[0])
        kappa = decoder(kappa).sample()

        spike = np.hstack([t[:, np.newaxis], kappa])
        spikes.append(spike)

    cs = [np.ones(spikes[k].shape[0]) * k for k in range(K)]

    spikeTrain = []
    clu = []

    for k, spike, c in zip(range(n_site), [spikes[K//n_site*i:K//n_site*(i+1)] for i in range(n_site)], [cs[K//n_site*i:K//n_site*(i+1)] for i in range(n_site)]):

        spike, c = np.vstack(spike), np.hstack(c)
        spike, c = spike[np.argsort(spike[:, 0])], c[np.argsort(spike[:, 0])]

        spikeTrain.append(spike)
        clu.append(c)

    data = {}
    data['spikeTrain'], data['clu'], data['stim'], data['sep'] = spikeTrain, clu, batch_stim, [0, T]
    np.save(os.path.join(ROOTDIR, 'SPIKE_data', 'Lorenz3D', 'data' + str(l) + '.npy'), data)

    data.update({'stim': batch_stim[:, [0, 1, 3]]})
    np.save(os.path.join(ROOTDIR, 'SPIKE_data', 'Lorenz2D', 'data' + str(l) + '.npy'), data)
