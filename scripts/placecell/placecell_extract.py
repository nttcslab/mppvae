from environ import *
import os
os.chdir(ROOTDIR)
import numpy as np
import h5py
from glob import glob

def extract(ratname, sep):

    with h5py.File(os.path.join('SPIKE_data', ratname, ratname + '_sessInfo.mat'), 'r') as f:
        stim = np.hstack([np.array(f['sessInfo']['Position']['TimeStamps']), np.array(f['sessInfo']['Position']['TwoDLocation']).T])
        maze_epoch = f['sessInfo']['Epochs']['MazeEpoch'][()][:, 0]

    spikeTrain = []
    n_group = len(glob(os.path.join('SPIKE_data', ratname, ratname + '.fet.*')))
    n_zkappa = []

    for l in range(1, n_group+1):

        n_kappa_length, n_kappa_ch = 32, (np.loadtxt(os.path.join('SPIKE_data', ratname, ratname + '.fet.' + str(l)), max_rows=1, dtype=int) - 1 - 4) // 3

        res = np.loadtxt(os.path.join('SPIKE_data', ratname, ratname + '.res.' + str(l))) / 20000
        clu = np.loadtxt(os.path.join('SPIKE_data', ratname, ratname + '.clu.' + str(l)))[1:]

        mask = (sep[0] < res) * (res < sep[1]) * (clu != 0)
        print(np.sum(mask), np.sum((sep[0] < res) * (res < sep[1])), ((sep[0] < res) * (res < sep[1]) * (clu == 0)).sum(), ((sep[0] < res) * (res < sep[1]) * (clu == 1)).sum(), n_kappa_ch)

        if np.sum(mask):

            with open(os.path.join('SPIKE_data', ratname, ratname + '.spk.' + str(l)), "rb") as f:
                spk = np.fromfile(f, dtype=np.int16)
                spk = spk.reshape((-1, n_kappa_length, n_kappa_ch))[mask].astype(np.float32)

            spikeTrain.append(np.hstack([res[mask, np.newaxis], spk.reshape((-1, n_kappa_length * n_kappa_ch))]))
            n_zkappa.append(n_kappa_length * n_kappa_ch)

            del spk

    data = {'ratname': ratname, 'spikeTrain': spikeTrain, 'stim': stim, 'sep': sep}

    return data


ratname = 'Gatsby_08282013'
sep = [18200, 18600]
data = extract(ratname, sep)
np.save(os.path.join(ROOTDIR, 'SPIKE_data', ratname, 'data.npy'), data)


ratname = 'Achilles_11012013'
sep = [20780, 21180]
data = extract(ratname, sep)

## Remove contaminated data of 13-th site from Achilles_11012013
spikeTrain = data['spikeTrain']
spikeTrain = [spikeTrain[i] for i in range(len(spikeTrain)) if i != 12]
data['spikeTrain'] = spikeTrain

np.save(os.path.join(ROOTDIR, 'SPIKE_data', ratname, 'data.npy'), data)


ratname = 'Gatsby_08022013'
sep = [18200, 18600]
data = extract(ratname, sep)

## Remove contaminated data of 8-th site from Gatsby_08022013
spikeTrain = data['spikeTrain']
spikeTrain = [spikeTrain[i] for i in range(len(spikeTrain)) if i != 7]
data['spikeTrain'] = spikeTrain

np.save(os.path.join(ROOTDIR, 'SPIKE_data', ratname, 'data.npy'), data)


ratname = 'Achilles_10252013'
sep = [18260, 18660]
data = extract(ratname, sep)

## Remove contaminated data of 6-th site from Achilles_10252013
spikeTrain = data['spikeTrain']
spikeTrain = [spikeTrain[i] for i in range(len(spikeTrain)) if i != 5]
data['spikeTrain'] = spikeTrain

np.save(os.path.join(ROOTDIR, 'SPIKE_data', ratname, 'data.npy'), data)
