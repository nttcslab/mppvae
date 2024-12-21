from environ import *
import os
os.chdir(ROOTDIR)
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

ratnames = ['Gatsby_08282013', 'Gatsby_08022013', 'Achilles_11012013', 'Achilles_10252013']

for ratname in ratnames:

    data = np.load(os.path.join(ROOTDIR, 'SPIKE_data', ratname, 'data.npy'), allow_pickle=True)[()]
    spikeTrain, stim, sep = data['spikeTrain'], data['stim'], data['sep']

    clus = []

    for spike in spikeTrain:

        n_kappa_ch, n_kappa_length = (spike.shape[-1] - 1) // 32, 32

        pca = PCA(n_kappa_ch*3, whiten=True)
        pca = pca.fit(spike[:, 1:])
        fet = pca.fit_transform(spike[:, 1:])

        param_grid = {"n_components": range(1, 21),}
        grid_search = GridSearchCV(GaussianMixture(covariance_type='diag'), param_grid=param_grid, scoring=lambda gmm, X: - gmm.bic(X))
        grid_search = grid_search.fit(fet)

        df = pd.DataFrame(grid_search.cv_results_)[['param_n_components',  'mean_test_score']]
        df['mean_test_score'] = - df['mean_test_score']
        df = df.rename(columns={'param_n_components': 'Number of components', 'mean_test_score': 'BIC'})

        n_components = df.sort_values(by='BIC')['Number of components'].iloc[0]
        gmm = GaussianMixture(n_components=n_components, covariance_type='diag')
        gmm = gmm.fit(fet)
        clu = gmm.fit_predict(fet)

        clus.append(clu)

    np.save(os.path.join(ROOTDIR, 'SPIKE_data', ratname, 'clu.npy'), clus)
