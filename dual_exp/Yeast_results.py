import sys
import numpy as np
from Utils.load_data import load_yeast
from Methods.IOKR_plus import IOKR_plus

np.random.seed(0)

# Load data
path = 'Data/YEAST_Elisseeff_Weston_2002.csv'
X, Y, n, p, d = load_yeast(path)
X = X[:300]  # reduced size for faster plot
Y = Y[:300]


# Algorithm specific parameters
algo = sys.argv[1]

if algo == 'e_krr':
    params = np.linspace(1e-1, 1, 10)
    n_epochs = 60
    crit = 1e-4

elif algo == 'e_svr':
    params = np.linspace(1e-1, 1, 10)
    n_epochs = 500
    crit = 1e-5

elif algo == 'k_huber':
    params = np.logspace(-3, 1, 16)
    n_epochs = 500
    crit = 1e-4


# Global parameters and initialization
mu = 1e-8
gamma = 1.
n_folds = 3
step_size = 'auto'
lambdas = np.logspace(-8, -1, 20)
n_lambdas = len(lambdas)
n_params = len(params)
MSEs = np.zeros((n_lambdas, n_params + 1))
Spar = np.zeros((n_lambdas, n_params))


# Standard IOKR
clf = IOKR_plus()

for i_L, L in enumerate(lambdas):

    scores = clf.cross_validation_score(X, Y, L=L, gamma=gamma, algo='iokr',
                                        n_folds=n_folds)
    MSEs[i_L, -1] = scores[1]


# Robust IOKR
clf = IOKR_plus()

for i_L, L in enumerate(lambdas):
    for i_param, param in enumerate(params):

        scores = clf.cross_validation_score(
            X, Y, n_folds=n_folds, L=L, gamma=gamma, algo=algo,
            alg_param=param, n_epochs=n_epochs, step_size=step_size, mu=mu,
            crit=crit)

        MSEs[i_L, i_param] = scores[1]
        Spar[i_L, i_param] = scores[-1]


# Save results
if algo == 'e_krr':
    np.save('Results/yeast_krr_MSEs.npy', MSEs)
    np.save('Results/yeast_krr_Spar.npy', Spar)


elif algo == 'e_svr':
    np.save('Results/yeast_svr_MSEs.npy', MSEs)
    np.save('Results/yeast_svr_Spar.npy', Spar)


elif algo == 'k_huber':
    np.save('Results/yeast_huber_MSEs.npy', MSEs)
    np.save('Results/yeast_huber_Spar.npy', Spar)
