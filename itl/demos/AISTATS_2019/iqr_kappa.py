import matplotlib

import sys
import os

import numpy as np
import tensorflow as tf

from itl import *
from sklearn.preprocessing import StandardScaler


def main(argv=None):
    prefix = tf.app.flags.FLAGS.output_dir
    path = tf.app.flags.FLAGS.save_graph
    show = tf.app.flags.FLAGS.show
    if argv is None:
        argv = sys.argv
    if show:
        matplotlib.use('WebAgg')
    else:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap('viridis_r')
    params = {'text.usetex': True,
              'figure.titlesize': 20,
              'legend.fontsize': 20,
              'legend.handlelength': 2,
              'axes.labelsize': 20,
              'axes.titlesize': 20,
              'xtick.labelsize': 20,
              'ytick.labelsize': 20}
    plt.rcParams.update(params)

    np.random.seed(13)

    N = int(argv[1]) if len(argv) > 1 else 10000
    if N < 0:
        raise RuntimeError("N must be greater than 0.")
    Nt = 2000
    d = 1
    p = 1
    quantiles_list = [0.05, 0.25, 0.5, 0.75, 0.95]
    NEW_COLORS = cmap(np.linspace(0, 1, len(quantiles_list)))
    npx_train, npy_train, npz_train = datasets.toy_data_quantile(N)
    npx_test, npy_test, npz_test = datasets.toy_data_quantile(
        Nt, probs=quantiles_list)

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    npx_train = scaler_x.fit_transform(npx_train.reshape(-1, 1))
    npy_train = scaler_y.fit_transform(npy_train.reshape(-1, 1))

    npx_test = scaler_x.transform(npx_test.reshape(-1, 1))
    npy_test = scaler_y.transform(npy_test.reshape(-1, 1))

    fig, axis = plt.subplots(2, 2, figsize=(16, 8))
    if path != '':
        graph = path + '/iqr'
    else:
        graph = path
    for i, kappa in enumerate([1e1, 1e-1, 1e-3, 0]):
        ax = axis[i // 2, i % 2]
        est = QuantileReg(model=ITLModel(Gaussian(1),
                                         Gaussian(), Gaussian(),
                                         derivative=True),
                          solver='L-BFGS-B',
                          lbda={'rkhs': 1e-3, 'crossing': 1.},
                          solver_param={'disp': 1},
                          device='/device:XLA_GPU:0',
                          cost=cost.HuberPinball(kappa=kappa),
                          summary={'path': graph})
        est.fit(npx_train, npy_train)

        pred = scaler_y.inverse_transform(
            est.predict(np.linspace(npx_test.min(),
                                    npx_test.max(),
                                    1000).reshape(-1, 1),
                        np.array(quantiles_list).reshape(-1, 1)))
        fig.gca().set_prop_cycle(None)
        for j in range(len(quantiles_list)):
            ax.plot(scaler_x.inverse_transform(np.linspace(npx_test.min(),
                                                           npx_test.max(),
                                                           1000)),
                    pred[:, j], '--', c=NEW_COLORS[j])
        fig.gca().set_prop_cycle(None)
        for j in range(len(quantiles_list)):
            ax.plot(scaler_x.inverse_transform(npx_test).reshape(-1, 1),
                    np.array(npz_test).reshape(len(quantiles_list),
                                               -1).T[:, j],
                    '-', c=NEW_COLORS[j])
        fig.gca().set_prop_cycle(None)
        ax.scatter(scaler_x.inverse_transform(npx_train),
                   scaler_y.inverse_transform(npy_train),
                   marker='.', c='lightgray')
        fig.gca().set_prop_cycle(None)
        ax.set_title(r'$\kappa={}$'.format(kappa))
        ax.set_xlabel(r'$\mathcal{X}$')
        ax.set_ylabel(r'$\mathcal{Y}$')

    plt.tight_layout()
    if show:
        plt.show()
    else:
        if not os.path.exists(prefix):
            os.mkdir(prefix)
        fig.savefig(prefix + '/iqr_kappa.eps', bbox_inches='tight')
        fig.savefig(prefix + '/iqr_kappa.pdf', bbox_inches='tight')

    return 0

if __name__ == '__main__':
    tf.app.flags.DEFINE_string('output_dir',
                               './icsl_results/',
                               'Path to the directory used to store the output '
                               'files')
    tf.app.flags.DEFINE_string('save_graph',
                               '',
                               'Path to the directory used to store the '
                               'computation graph.')
    tf.app.flags.DEFINE_boolean('show',
                                False,
                                'Diplay on screen the generated figures')
    tf.app.run()
