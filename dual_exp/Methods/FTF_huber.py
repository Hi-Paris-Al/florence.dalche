import numpy as np


def kernel_laplace(x1, X, gamma_x):
    return(np.exp(-gamma_x * np.abs(x1 - X)).mean(axis=1))


def kernel_x_laplace(X, gamma_x):
    N = X.shape[0]
    K_X = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            K_X[i, j] = np.exp(-gamma_x * np.abs(X[i] - X[j])).mean()
    return(K_X)


def dual_loss(beta, K_X, T, S, reg):
    n = beta.shape[0]
    return 0.5 * np.trace(beta.dot(beta.T)) + 1 / (2 * reg * n) * np.trace(K_X.dot(beta.dot(S.dot(beta.T)))) - np.trace(beta.T.dot(T))


def dual_grad(beta, K_X, T, S, reg):
    n = beta.shape[0]
    return beta + 1 / (reg * n) * K_X.dot(beta.dot(S)) - T


def proj(beta, kappa):
    norm = np.sqrt(np.sum(beta**2, axis=1))
    mask = np.where(norm > kappa)
    beta[mask] *= kappa / norm[mask].reshape((-1, 1))
    return beta


def pgd(K_X, T, S, reg, gamma, kappa, n_epochs=50):
    beta = np.zeros(T.shape)
    losses = []
    for n in range(n_epochs):
        beta -= gamma * dual_grad(beta, K_X, T, S, reg)
        beta = proj(beta, kappa)
        losses.append(dual_loss(beta, K_X, T, S, reg))
    return(beta, losses)


def make_psi_cos_sin(M, t):
    d = t.shape[0]
    psi = np.zeros((2 * M, d))
    for i in range(M):
        for j in range(d):
            psi[2 * i, j] = np.sqrt(2) * np.cos(np.pi * 2 * i * t[j])
            psi[2 * i + 1, j] = np.sqrt(2) * np.sin(np.pi * 2 * i * t[j])
    return(psi)


def pred(X_train, X_test, beta, reg, psi, gamma_x, S):
    n = X_train.shape[0]
    f_x_test = 1 / (reg * n) * kernel_laplace(X_test, X_train,
                                              gamma_x).reshape((1, -1)).dot(np.dot(beta.dot(S), psi))
    return(f_x_test.ravel())


def ftf_huber(X, kernel_x, gamma_x, Y, Psi, eigenv, kappa, reg, n_epochs):
    n, d = X.shape
    K_X = kernel_x(X, gamma_x)
    T = 1 / d * Y.dot(Psi.T)
    S = np.diag(eigenv)
    gamma = 1 / (1 + np.trace(K_X) * eigenv.sum() / reg / n)
    beta, losses = pgd(K_X, T, S, reg, gamma, kappa, n_epochs)
    return(beta, losses)


def mask(j, N):
    res = np.ones(N, dtype=bool)
    res[j] = False
    return(res)

def best_reg(X, kernel_x, gamma_x, Y, Psi, eigenv, n_epochs):
    n = X.shape[0]
    reg_list = np.logspace(-3, 0, 10)
    err = []
    for reg in reg_list:
        err_local = []
        for i in range(n):
            ma = mask(i,n)
            beta, losses = ftf_huber(X=X[ma], kernel_x=kernel_x,
                                     gamma_x=gamma_x, Y=Y[ma], Psi=Psi, eigenv=eigenv,
                                     kappa=100, reg=reg, n_epochs=n_epochs)
            pred_i = pred(X_train=X[ma], X_test=X[i], beta=beta,
                          reg=reg, psi=Psi, gamma_x=gamma_x, S=np.diag(eigenv))
            err_local.append(((pred_i - Y[i])**2).mean())
        err.append(np.array(err_local).mean())
    return(reg_list[np.argmin(np.array(err))])
