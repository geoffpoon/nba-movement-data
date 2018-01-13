
import numpy as np
from sklearn.decomposition import NMF
import scipy


def run(LL, n_features=10, normalize_what='features'):
    model = NMF(n_components=n_features, init='nndsvda', max_iter=6000, tol=1e-7,
                solver='mu', beta_loss='kullback-leibler')
    W = model.fit_transform(LL)
    H = model.components_    
    
    ## LL ~ W.H
    ## W = weights (n_samples, n_components)
    ## H = features (n_components, n_features)

    if normalize_what == 'features':
        W_norm = np.copy(W)
        H_norm = np.copy(H)
        for i in range(n_features):
            temp = np.sum(H[i,:])
            W_norm[:,i] *= temp
            H_norm[i,:] *= 1./temp
        return W_norm, H_norm
    elif normalize_what == 'weights':
        n_samples = len(W_norm)
        for i in range(n_samples):
            temp = np.sum(W[i,:])
            W_norm[i,:] *= 1./temp
            H_norm[:,i] *= temp
        return W_norm, H_norm
    else:
        return W, H