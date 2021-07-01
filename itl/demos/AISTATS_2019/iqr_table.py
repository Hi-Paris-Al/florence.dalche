import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_DYNAMIC'] = 'FALSE'

import operalib as ovk
import matplotlib.pyplot as plt
import numpy as np
import copy
import time
import sys
import GPyOpt
import ctypes
import warnings

from scipy.spatial.distance import pdist, squareform
from scipy.stats.mstats import mquantiles
from scipy.stats import norm, uniform
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split, KFold
from itl import *
from operalib import Quantile
from joblib import Parallel, delayed
from qreg import QRegMTL

class CV(object):

    def __init__(self, est, n_folds, X_train, y_train, T, quantiles_list,
                 lbda_nc, bias, device, split_idx, dataset_idx, ftol):
        self.est = est
        self.n_folds = n_folds
        self.X_train = X_train
        self.y_train = y_train
        self.T = T
        self.quantiles_list = quantiles_list
        self.lbda_nc = lbda_nc
        self.bias = bias
        self.device = device
        self.split_idx = split_idx
        self.dataset_idx = dataset_idx
        self.iter = 0
        self.ftol = ftol

    def sample_loss_iqr(self, params, est, n_folds, X_train, y_train, T,
                        quantiles_list, lbda_nc, bias, device):
        import tensorflow as tf
        #     params[0] gamma_x
        #     params[1] gamma_t
        #     params[2] lbda
        kf = KFold(n_folds)
        Rk = np.zeros(n_folds)
        if est is None:
            est = [None for j in range(n_folds)]
        for j, (train_idx, test_idx) in enumerate(kf.split(X_train)):
            est[j] = QuantileReg(model=ITLModel(Gaussian(10 ** params[0]),
                                                Gaussian(10 ** params[1]),
                                                Gaussian(10 ** params[1]),
                                                derivative=not np.isclose(lbda_nc, 0)),
                                 sampler=SobolUniform_0p1(T),
                                 solver='L-BFGS-B',
                                 lbda={'rkhs': 10 ** params[2],
                                       'crossing': lbda_nc * X_train.shape[0]},
                                 penalty=penalty.Hinge(),
                                 solver_param={'ftol': self.ftol, # speedup
                                               'maxls': 100},
                                 config_param={'intra_op_parallelism_threads': 1,
                                               'inter_op_parallelism_threads': 1},
                                 device=device)
            est[j].fit(X_train[train_idx, :], y_train[train_idx, :])
            result = est[j].predict(X_train[test_idx, :],
                                    quantiles_list.reshape(-1, 1))
            Rk[j] = np.linalg.norm(
                ploss(y_train[test_idx, :],
                      result, np.array([quantiles_list]).reshape(-1, 1)),
                1)
            #print(ploss(y_train[test_idx, :],
                        #result, np.array([quantiles_list]).reshape(-1, 1)), 1)
            self.iter = self.iter + 1
            #est[j].close_session().finalise()
            #print('iqr', params, Rk[j], self.iter // n_folds, j,
                  #self.split_idx, DATASET_NAME_REGRESSION[self.dataset_idx])
        print('iqr', params, (Rk.mean(), Rk.std()), self.iter // n_folds,
              self.split_idx, datasets.NAMES_REGRESSION[self.dataset_idx])
        return Rk.mean(), est

    def sample_loss_jqr(self, params, est, n_folds, X_train, y_train,
                        quantiles_list, lbda_nc):
        kf = KFold(n_folds)
        Rk = np.zeros(n_folds)
        if est is None:
            est = [None for j in range(n_folds)]
        for j, (train_idx, test_idx) in enumerate(kf.split(X_train)):
            est[j] = Quantile(probs=quantiles_list, gamma=10 ** params[0],
                              nc_const=lbda_nc,
                              gamma_quantile=(10 ** params[1]
                                              if params[1] < np.inf
                                              else np.inf),
                              lbda=10 ** params[2])
            est[j].fit(X_train[train_idx, :], y_train[train_idx, :])
            result = est[j].predict(X_train[test_idx, :]).T
            Rk[j] = np.linalg.norm(
                ploss(y_train[test_idx, :],
                      result, np.array([quantiles_list]).reshape(-1, 1)),
                1)
            #print(ploss(y_train[test_idx, :],
                        #result, np.array([quantiles_list]).reshape(-1, 1)), 1)
            self.iter = self.iter + 1
            #print('jqr' if lbda_nc else 'ind', params, Rk[j],
                  #self.iter // n_folds, j,
                  #self.split_idx,
                  #DATASET_NAME_REGRESSION[self.dataset_idx])
        print('jqr' if lbda_nc else 'ind', params, (Rk.mean(), Rk.std()),
              self.iter // n_folds,
              self.split_idx, datasets.NAMES_REGRESSION[self.dataset_idx])
        return Rk.mean(), est

    def objective_iqr(self, params):
        fs = np.zeros((params.shape[0], 1))
        params = np.atleast_2d(params)
        for i in range(params.shape[0]):
            fs[i], _ = self.sample_loss_iqr(params[i, :], self.est,
                                             self.n_folds, self.X_train,
                                             self.y_train, self.T,
                                             self.quantiles_list, self.lbda_nc,
                                             self.bias, self.device)
        return fs

    def objective_jqr(self, params):
        fs = np.zeros((params.shape[0], 1))
        if len(params.ravel()) < 3:
            params = np.insert(params, 1, np.inf)
        params = np.atleast_2d(params)
        for i in range(params.shape[0]):
            fs[i], _ = self.sample_loss_jqr(params[i, :], self.est,
                                            self.n_folds, self.X_train,
                                            self.y_train, self.quantiles_list,
                                            self.lbda_nc)
        return fs


def run_fold(X, y, dataset_idx, methods, i, save_path,
             lbda_nc, T, quantiles_list,
             n_folds, b_iter, b_init, bias, device, ftol):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3,
                                                        random_state=i,
                                                        shuffle=True)

    import tensorflow as tf

    result_jqr = np.empty(0)
    result_iqr = np.empty(0)
    result_ind = np.empty(0)
    result_mtl = np.empty(0)

    if 'jqr' in methods:
        cv = CV(None, n_folds, X_train, y_train, T, quantiles_list,
                True, bias, device, i, dataset_idx, ftol)
        domain =[{'name': 'gamma_x', 'type': 'continuous', 'domain': (-6, 6)},
                 {'name': 'gamma_t', 'type': 'continuous', 'domain': (-6, 6)},
                 {'name': 'lambda',  'type': 'continuous', 'domain': (-6, 6)}]

        np.random.seed(i)
        opt = GPyOpt.methods.BayesianOptimization(f=cv.objective_jqr,
                                                  domain=domain,
                                                  model_type='GP',
                                                  acquisition_type='EI',
                                                  initial_design_numdata=27,
                                                  initial_design_type='sobol',
                                                  save_models_parameters=False,
                                                  normalize_Y=True,
                                                  random_state=i)
        opt.run_optimization(max_iter=b_iter)
        gamma_x_best, gamma_t_best, lbda_best = opt.X[np.argmin(opt.Y)]
        start = time.time()
        est = Quantile(probs=quantiles_list, gamma=10 ** gamma_x_best,
                       nc_const=True, gamma_quantile=10 ** gamma_t_best,
                       lbda=10 ** lbda_best)
        est.fit(X_train, y_train)
        train_time = time.time() - start
        start = time.time()
        result = est.predict(X_test).T
        test_time = time.time() - start
        result_jqr = [
            np.linalg.norm(ploss(y_test, result,
                np.array([quantiles_list]).reshape(-1, 1)), 1) * 100,
            np.sum(np.mean(
                -np.minimum(np.diff(result, axis=1), 0), axis=0)) * 100,
            gamma_x_best, gamma_t_best, lbda_best, train_time, test_time
        ]
        if not os.path.exists(save_path + '/jqr/'):
            os.makedirs(save_path + '/jqr/')
        np.savez_compressed("{}/jqr/{}-{}-{}".format(
            save_path, dataset_idx, datasets.NAMES_REGRESSION[dataset_idx], i),
            **dict(zip(['result', 'cv scores', 'params'],
                       [np.array(result_jqr), opt.Y, opt.X])))

    if 'iqr' in methods:
        cv = CV(None, n_folds, X_train, y_train, T, quantiles_list,
                1., bias, device, i, dataset_idx, ftol)
        domain =[{'name': 'gamma_x', 'type': 'continuous', 'domain': (-6, 6)},
                 {'name': 'gamma_t', 'type': 'continuous', 'domain': (-6, 6)},
                 {'name': 'lambda',  'type': 'continuous', 'domain': (-6, 6)}]

        np.random.seed(i)
        opt = GPyOpt.methods.BayesianOptimization(f=cv.objective_iqr,
                                                  domain=domain,
                                                  model_type='GP',
                                                  acquisition_type='EI',
                                                  initial_design_numdata=27,
                                                  initial_design_type='sobol',
                                                  normalize_Y=True,
                                                  random_state=i)
        opt.run_optimization(max_iter=b_iter)
        gamma_x_best, gamma_t_best, lbda_best = opt.X[np.argmin(opt.Y)]
        start = time.time()
        est = QuantileReg(model=ITLModel(
                            Gaussian(10 ** gamma_x_best),
                            Gaussian(10 ** gamma_t_best),
                            Gaussian(10 ** gamma_t_best),
                            derivative=not np.isclose(lbda_nc, 0)),
                          sampler=SobolUniform_0p1(T),
                          solver='L-BFGS-B',
                          lbda={'rkhs': 10 ** lbda_best,
                                'crossing': lbda_nc * X_train.shape[0]},
                          solver_param={'maxls': 100},
                          config_param={'intra_op_parallelism_threads': 1,
                                        'inter_op_parallelism_threads': 1},
                          penalty=penalty.Hinge(),
                          device=device)
        est.fit(X_train, y_train)
        train_time = time.time() - start
        start = time.time()
        result = est.predict(X_test, quantiles_list.reshape(-1, 1))
        test_time = time.time() - start
        result_iqr = [
            np.linalg.norm(ploss(y_test, result,
                np.array([quantiles_list]).reshape(-1, 1)), 1) * 100,
            np.sum(np.mean(
                -np.minimum(np.diff(result, axis=1), 0), axis=0)) * 100,
            gamma_x_best, gamma_t_best, lbda_best, train_time, test_time
        ]
        if not os.path.exists(save_path + '/iqr/'):
            os.makedirs(save_path + '/iqr/')
        np.savez_compressed("{}/iqr/{}-{}-{}".format(
            save_path, dataset_idx, datasets.NAMES_REGRESSION[dataset_idx], i),
            **dict(zip(['result', 'cv scores', 'params'],
                       [np.array(result_iqr), opt.Y, opt.X])))

    if 'ind' in methods:
        cv = CV(None, n_folds, X_train, y_train, T, quantiles_list,
                False, bias, device, i, dataset_idx, ftol)
        domain =[{'name': 'gamma_x', 'type': 'continuous', 'domain': (-6, 6)},
                 {'name': 'lambda',  'type': 'continuous', 'domain': (-6, 6)}]

        np.random.seed(i)
        opt = GPyOpt.methods.BayesianOptimization(f=cv.objective_jqr,
                                                  domain=domain,
                                                  model_type='GP',
                                                  acquisition_type='EI',
                                                  initial_design_numdata=27,
                                                  initial_design_type='sobol',
                                                  normalize_Y=True,
                                                  random_state=i)
        opt.run_optimization(max_iter=b_iter)
        gamma_x_best, lbda_best = opt.X[np.argmin(opt.Y)]
        start = time.time()
        est = Quantile(probs=quantiles_list, gamma=10 ** gamma_x_best,
                       nc_const=True, gamma_quantile=np.inf,
                       lbda=10 ** lbda_best)
        est.fit(X_train, y_train)
        train_time = time.time() - start
        start = time.time()
        result = est.predict(X_test).T
        test_time = time.time() - start
        result_ind = [
            np.linalg.norm(ploss(y_test, result,
                np.array([quantiles_list]).reshape(-1, 1)), 1) * 100,
            np.sum(np.mean(
                -np.minimum(np.diff(result, axis=1), 0), axis=0)) * 100,
            gamma_x_best, lbda_best, train_time, test_time
        ]
        if not os.path.exists(save_path + '/ind/'):
            os.makedirs(save_path + '/ind/')
        np.savez_compressed("{}/ind/{}-{}-{}".format(
            save_path, dataset_idx, datasets.NAMES_REGRESSION[dataset_idx], i),
            **dict(zip(['result', 'cv scores', 'params'],
                       [np.array(result_ind), opt.Y, opt.X])))

    return {'jqr': result_jqr, 'iqr': result_iqr, 'ind': result_ind}

def run_experiment(dataset_idx, methods=['iqr', 'jqr', 'liq'],
                   n_jobs_split=1, n_jobs_cv=1, save_path=None,
                   lbda_nc=1, T=100,
                   quantiles_list=np.array([.1, .3, .5, .7, .9]),
                   #quantiles_list=np.array([.1, .9]),
                   n_folds=5, b_iter=30, b_init=10, n_iter=20, bias=True,
                   device='cpu', ftol=1e-5):
    X, y, (sX, sY, I) = datasets.import_regression_data(
        datasets.NAMES_REGRESSION[dataset_idx])

    results = Parallel(n_jobs=n_jobs_split,
                       backend="multiprocessing")(delayed(run_fold)(
        X=X, y=y, dataset_idx=dataset_idx, methods=methods, i=i,
        save_path=save_path,
        lbda_nc=lbda_nc, T=T, quantiles_list=quantiles_list, n_folds=n_folds,
        b_iter=b_iter, b_init=b_init, bias=bias, device=device, ftol=ftol)
        for i in range(n_iter))
    results = dict(zip(results[0],zip(*[d.values() for d in results])))
    for i in range(len(methods)):
        np.savez_compressed("{}/{}-{}-{}".format(
            save_path, dataset_idx,
            datasets.NAMES_REGRESSION[dataset_idx], methods[i]),
            results[methods[i]])
    return np.array(results)


def main(argv=None):
    import tensorflow as tf

    prefix = tf.app.flags.FLAGS.output_dir
    method = tf.app.flags.FLAGS.method.split(',')
    n_jobs_dat = tf.app.flags.FLAGS.n_jobs_dat
    n_jobs_split = tf.app.flags.FLAGS.n_jobs_split
    n_folds = tf.app.flags.FLAGS.n_folds
    n_splits = tf.app.flags.FLAGS.n_splits
    datasets_names = tf.app.flags.FLAGS.datasets.split(',')
    verbose = tf.app.flags.FLAGS.verbose
    ftol = tf.app.flags.FLAGS.ftol
    device = tf.app.flags.FLAGS.device
    if 'all' in datasets_names:
        datasets_names = datasets.NAMES_REGRESSION
    datasets_idx = [datasets.NAMES_REGRESSION.index(name)
                    for name in datasets_names]

    if not verbose:
        tf.logging.set_verbosity(tf.logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    try:
        mkl_rt = ctypes.CDLL('libmkl_rt.so')
        mkl_get_max_threads = mkl_rt.mkl_get_max_threads
        def mkl_set_num_threads(cores):
            mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(cores)))

            mkl_set_num_threads(1)
    except:
        print('No MKL found')

    if not os.path.exists(prefix):
        os.makedirs(prefix)

    results = Parallel(n_jobs=n_jobs_dat,
                       backend="multiprocessing")(delayed(run_experiment)(
        dataset_idx=i,
        methods=method,
        save_path=prefix,
        device=device,
        lbda_nc=1,
        n_folds=n_folds,
        b_iter=50,
        n_jobs_split=n_jobs_split,
        n_iter=n_splits,
        bias=True,
        ftol=ftol,
        n_jobs_cv=1) for i in datasets_idx)
    np.savez_compressed("{}/summary".format(prefix),
                        **dict(zip(datasets_names, results)))
    return 0


if __name__ == '__main__':
    # python iqr_table.py --n_splits=20 --n_jobs_split=10 --method=iqr,jqr,ind --datasets=all --ftol=2.220446049250313e-09 --device=/cpu:0 --iqr_crossing=10
    import tensorflow as tf

    warnings.warn("The full experiment might takes multiple days to terminate",
                  UserWarning)
    tf.app.flags.DEFINE_string('output_dir',
                               './iqr_table/',
                               'Path to the directory used to store the '
                               'output files')
    tf.app.flags.DEFINE_string('method',
                               'iqr,jqr,ind' ,
                               'A comma separated list of methods to compare:'
                               'iqr or/and jqr or/and ind')
    tf.app.flags.DEFINE_integer('n_jobs_dat',
                                1,
                                'Number of parallel datasets jobs')
    tf.app.flags.DEFINE_integer('n_jobs_split',
                                8,
                                'Number of parallel splits jobs')
    tf.app.flags.DEFINE_integer('n_folds',
                                5,
                                'Number of folds per cross validation')
    tf.app.flags.DEFINE_integer('n_splits',
                                20,
                                'Number of splits')
    tf.app.flags.DEFINE_string('datasets',
                               'all',
                               'Datasets on which to benchmark')
    tf.app.flags.DEFINE_string('device',
                               'gpu',
                               'device to use')
    tf.app.flags.DEFINE_boolean('verbose',
                                False,
                                'Verbose logging')
    tf.app.flags.DEFINE_float('ftol',
                              1e-5,
                              'Tolerance used during hyperparameter '
                              'optimization')
    tf.app.run()
