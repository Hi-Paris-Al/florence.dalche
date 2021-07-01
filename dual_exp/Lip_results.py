import numpy as np
import pandas as pd
from Methods.FTF_huber import *


# Load data
print('Importing data')
data_filename = 'Data/EMGmatlag.csv'
target_filename = 'Data/lipmatlag.csv'
sampling_filename = 'Data/tfine.csv'

data = pd.read_csv(data_filename, header=None)
target = pd.read_csv(target_filename, header=None)
sampling = pd.read_csv(sampling_filename, header=None)

X = data.values.T
Y = target.values.T
t = sampling.values.reshape(-1)
t = t / 0.64
T_0 = t.size
(N, d) = X.shape


# Outlier creation
n_o = 4
print('Creating outliers')
X_aug = np.zeros((n_o, d))
Y_aug = np.zeros((n_o, d))
ind_out = [7, 9, 18, 26]
X_aug = X[ind_out]
Y_aug = -1.2 * Y[ind_out]

X_tot = np.vstack([X, X_aug])
Y_tot = np.vstack([Y, Y_aug])

M_list = np.array([4, 5, 6, 7, 15])
print('Learning models (%s different values for m)' % M_list.shape[0])
kappa_list = np.linspace(0.05, 2, 50)
n_epochs = 2500
gamma_x = 7
err = np.zeros((len(M_list), kappa_list.shape[0]))


# Loop on m value
for j in range(M_list.shape[0]):

    print('Learning with m=%s' % M_list[j])
    M = M_list[j]
    psi = make_psi_cos_sin(M, t)
    s = np.zeros(2 * M)
    for i in range(M):
        s[2 * i] = 1 / (1 + i)**2
        s[2 * i + 1] = 1 / (1 + i)**2
    eigenv = s

    print('Computing best regularization parameter for plain Ridge Regression')
    # chosen by cross validation for ridge regression with chosen outliers
    reg = best_reg(X_tot, kernel_x_laplace, gamma_x,
                   Y_tot, psi, eigenv, n_epochs)

    # Loop on kappa value
    for k in range(kappa_list.shape[0]):
        if (k % 5 == 0):
            print('Computing for all kappas: ' + str(2 * k) + '%')
        kappa = kappa_list[k]
        err_local = []
        for i in range(N + n_o):
            ma = mask(i, N + n_o)

            beta, losses = ftf_huber(
                X=X_tot[ma], kernel_x=kernel_x_laplace, gamma_x=gamma_x,
                Y=Y_tot[ma], Psi=psi, eigenv=eigenv, kappa=kappa, reg=reg,
                n_epochs=n_epochs)

            pred_i = pred(X_train=X_tot[ma], X_test=X_tot[i], beta=beta,
                          reg=reg, psi=psi, gamma_x=gamma_x, S=np.diag(eigenv))

            err_local.append(((pred_i - Y_tot[i])**2).mean())

        err[j, k] = np.array(err_local).mean()


# Save results
np.save('Results/Lip_err.npy', err)
