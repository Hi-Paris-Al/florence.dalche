import numpy as np
from numba import njit
from scipy import linalg
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split


class IOKR_plus:

    def __init__(self):
        pass

    def fit(self, X, Y, L=1., gamma=1., algo='iokr', alg_param=0., n_epochs=10,
            step_size='auto', mu=1e-8, crit=None):

        # Saving parameters
        self.L = L
        self.gamma = gamma
        self.algo = algo
        self.alg_param = alg_param

        # Training
        self.X_tr = X.copy()
        self.Y_tr = Y.copy()
        self.n_tr = self.X_tr.shape[0]
        K_x = rbf_kernel(self.X_tr, Y=self.X_tr, gamma=self.gamma)

        if self.algo == 'iokr':
            M = K_x + self.n_tr * self.L * np.eye(self.n_tr)
            self.Omega = np.linalg.inv(M)

        elif self.algo in ['e_krr', 'e_svr', 'k_huber']:

            K_y = self.Y_tr.dot(self.Y_tr.T)

            self.W, self.Omega, self.objs = compute_Omega(
                algo, self.alg_param, K_x, K_y, self.L, n_epochs, step_size,
                mu, crit=crit)

            if self.algo in ['e_krr', 'e_svr']:
                self.sparsity = np.mean(np.linalg.norm(self.W, axis=1) < 1e-7)
            elif self.algo == 'k_huber':
                self.sparsity = np.mean(self.alg_param / (self.L * self.n_tr) -
                                        np.linalg.norm(self.W, axis=1) < 1e-7)

    def estimate_output_embedding(self, X_te):

        K_x_te_tr = rbf_kernel(X_te, Y=self.X_tr, gamma=self.gamma)
        Y_pred = K_x_te_tr.dot(self.Omega).dot(self.Y_tr)
        return Y_pred

    def predict_clamp(self, X_te):

        Y_pred = self.estimate_output_embedding(X_te)
        Y_pred[Y_pred > 0.5] = 1
        Y_pred[Y_pred <= 0.5] = 0
        return Y_pred

    def scores(self, X, Y, k=3):

        # Predict
        H = self.estimate_output_embedding(X)
        Y_pred = self.predict_clamp(X)

        # MSE
        n = X.shape[0]
        mse = np.linalg.norm(Y - H) ** 2
        mse /= n

        # Hamming loss
        n, d = Y.shape
        hamming = np.linalg.norm(Y - Y_pred) ** 2
        hamming *= 100 / (n * d)

        # Precision k
        Pk = Y.ravel()[np.argsort(H, axis=1)[:, :k] +
                       k * np.arange(n).reshape(-1, 1)]
        Pk = np.sum(Pk)
        Pk /= (k * n)

        # Sparsity (or saturation)
        try:
            sparsity = self.sparsity
        except AttributeError:
            sparsity = 0.

        return mse, Pk, hamming, sparsity

    def cross_validation_score(self, X, Y, n_folds=5, L=1., gamma=1.,
                               algo='iokr', alg_param=0., n_epochs=10,
                               step_size='auto', mu=1e-8, crit=None):

        res = np.zeros((n_folds, 7))

        for i in range(n_folds):

            # Split
            X_train, X_test, Y_train, Y_test = train_test_split(
                X, Y, test_size=1. / n_folds, random_state=i)

            # Fit
            self.fit(
                X_train, Y_train, L=L, gamma=gamma, algo=algo,
                alg_param=alg_param, n_epochs=n_epochs, step_size=step_size,
                mu=mu, crit=crit)

            res[i, ::2] = np.array(self.scores(X_train, Y_train))      # Train
            res[i, 1::2] = np.array(self.scores(X_test, Y_test))[:-1]  # Test

        res_avg = np.mean(res, axis=0)

        return res_avg


@njit
def objective(UtU, UtV, W, beta):
    A = 0.5 * np.trace(UtU.dot(W).dot(W.T))
    B = - np.trace(UtV.dot(W.T))
    C = beta * np.trace(np.sqrt(W.dot(W.T)))
    return A + B + C


@njit
def BST(x, tau):
    norm_x = np.linalg.norm(x)
    if norm_x >= tau:
        return x - x / norm_x * tau
    else:
        return np.zeros_like(x)


@njit
def proj(x, tau):
    norm_x = np.linalg.norm(x)
    if norm_x < tau:
        return x
    else:
        return x * tau / norm_x


@njit
def compute_W(algo, alg_param, UtU, UtV, L, W_init, n_epochs, step_size,
              crit=None):
    """
    Compute the solution to the dual problem w.r.t. chosen algorithm
    """
    # Initialization
    n_row = W_init.shape[0]
    W = W_init.copy()
    objs = np.zeros(n_epochs)

    # Iterations
    for s in range(n_epochs):

        if algo == 'e_krr':
            obj_param = alg_param
            for i in range(n_row):
                # Block Coordinate Descent step
                W[i, :] += 1 / UtU[i, i] * (UtV[i, :] - UtU[i, :].dot(W))
                W[i, :] = BST(W[i, :], alg_param / UtU[i, i])

        elif algo == 'e_svr':
            obj_param = alg_param
            for i in range(n_row):
                # Block Coordinate Descent step
                W[i, :] += 1 / UtU[i, i] * (UtV[i, :] - UtU[i, :].dot(W))
                W[i, :] = BST(W[i, :], alg_param / UtU[i, i])
                # Projection step
                W[i, :] = proj(W[i, :], 1. / (L * n_row))
                obj_param = alg_param

        elif algo == 'k_huber':
            obj_param = 0.
            # Gradient step
            W -= step_size * (UtU.dot(W) - UtV)
            # Projection step
            for i in range(n_row):
                W[i, :] = proj(W[i, :], alg_param / (L * n_row))

        # Objective and stopping criterion
        objs[s] = objective(UtU, UtV, W, obj_param)
        if crit is not None and s > 0 and np.abs((objs[s - 1] - objs[s]) /
                                                 objs[0]) < crit:
            break

    return W, objs


def compute_Omega(algo, alg_param, K_X, K_Y, L, n_epochs, step_size, mu,
                  crit=None):
    """
    Compute optimal coefficients/outputs matrix Omega w.r.t. chosen algorithm
    """
    n = K_X.shape[0]
    if algo in ['e_krr', 'k_huber']:
        UtU = K_X + n * L * np.eye(n)
    elif algo == 'e_svr':
        UtU = K_X + mu * np.eye(n)

    Q, s, Qt = linalg.svd(K_Y + mu * np.eye(n))
    D = np.diag(s)
    UtV = Q.dot(np.sqrt(D))

    if step_size == 'auto':
        step_size = 8. / (np.trace(K_X) + n * L)

    W_init = np.zeros((n, n))
    W, objs = compute_W(algo, alg_param, UtU, UtV, L, W_init, n_epochs,
                        step_size, crit=crit)

    Omega = W.dot((np.diag(1 / np.sqrt(s))).dot(Qt))

    return W, Omega, objs
