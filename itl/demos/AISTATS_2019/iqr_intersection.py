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
    N = tf.app.flags.FLAGS.sample_train
    if argv is None:
        argv = sys.argv
    if N < 0:
        raise RuntimeError("sample_train must be greater than 0.")
    if show:
        matplotlib.use('WebAgg')
    else:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
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

    d = 1
    p = 1
    quantiles_list = np.linspace(0, 1, 100)
    npx_train, npy_train, _ = datasets.toy_data_quantile(N)

    scaler_x = IdentityScaler()
    scaler_y = IdentityScaler()

    npx_train = scaler_x.fit_transform(npx_train.reshape(-1, 1))
    npy_train = scaler_y.fit_transform(npy_train.reshape(-1, 1))

    MODELS = ['Piecewise Linear', 'Gaussian']

    fig, axis = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)
    for i in range(2):
        if path != '':
            graph = path + ('/' + str(MODELS[i]))
        else:
            graph = path
        if MODELS[i] == 'Piecewise Linear':
            h = ITLModel(Gaussian(15), Intersection(offset=5))
        else:
            h = ITLModel(Gaussian(15), Gaussian(15), Gaussian(15))
        est = QuantileReg(model=h, solver='L-BFGS-B',
                          lbda={'rkhs': 1e-3,
                                'p_rkhs': 0,
                                'crossing': 0},
                          solver_param={'disp': 1, 'maxls': 50,
                                        'gtol': 1e-8, 'ftol': 1e-11},
                          #sampler=Dirac([.99, .5, .01]),
                          device='/device:GPU_XLA:0',
                          cost=cost.HuberPinball(.0001),
                          penalty=penalty.HuberHinge(.0001),
                          summary={'path': graph})
        est.fit(npx_train, npy_train)

        npx_test = np.linspace(npx_train.min(),
                               npx_train.max(),
                               1000).reshape(-1, 1)
        pred = scaler_y.inverse_transform(
            est.predict(npx_test,
                        np.array(quantiles_list).reshape(-1, 1)))
        #pred = np.tile(pred, (npx_test.shape[0], 1))
        fig.gca().set_prop_cycle(None)
        colors = [ matplotlib.cm.viridis_r(x)
                   for x in np.linspace(0, 1, len(quantiles_list)) ]
        for j in range(len(colors)):
            axis[i].plot(scaler_x.inverse_transform(
                np.linspace(npx_train.min(), npx_train.max(), 1000)),
                pred[:, j], '-', c=colors[j], zorder=1)

        axis[i].scatter(scaler_x.inverse_transform(npx_train),
                        scaler_y.inverse_transform(npy_train),
                        c='black', zorder=2)
        axis[i].set_xlabel('$\mathcal{X}$')
        axis[i].set_ylabel(r'$\mathcal{Y}$')

        axis[i].set_title(r'{}'.format(MODELS[i]))
        sm = plt.cm.ScalarMappable(cmap=matplotlib.cm.viridis_r,
                                   norm=plt.Normalize(vmin=0, vmax=1))
        sm._A = []
        cbar = plt.colorbar(sm, ax=axis[i])
        cbar.ax.set_ylabel(r'$\theta$')

    plt.tight_layout()
    if show:
        plt.show()
    else:
        if not os.path.exists(prefix):
            os.mkdir(prefix)
        fig.savefig(prefix + '/iqr_crossing.eps', bbox_inches='tight')
        fig.savefig(prefix + '/iqr_crossing.pdf', bbox_inches='tight')

    return 0

if __name__ == '__main__':
    tf.app.flags.DEFINE_string('output_dir',
                               './iqr_crossing_results/',
                               'Path to the directory used to store the '
                               'output files')
    tf.app.flags.DEFINE_string('save_graph',
                               '',
                               'Path to the directory used to store the '
                               'computation graph.')
    tf.app.flags.DEFINE_integer('sample_train',
                                40,
                                'Number of samples used to train the model')
    tf.app.flags.DEFINE_boolean('show',
                                False,
                                'Diplay on screen the generated figures')
    tf.app.run()
