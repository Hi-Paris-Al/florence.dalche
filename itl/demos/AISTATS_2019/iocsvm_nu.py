import matplotlib

import sys
import os

import numpy as np
import tensorflow as tf

from itl import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import svm
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.decomposition.pca import PCA


def main(argv=None):
    prefix = tf.app.flags.FLAGS.output_dir
    path = tf.app.flags.FLAGS.save_graph
    show = tf.app.flags.FLAGS.show
    case = tf.app.flags.FLAGS.setting
    if argv is None:
        argv = sys.argv
    if show:
        matplotlib.use('WebAgg')
    else:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm
    params = {'text.usetex': True,
              'figure.titlesize': 20,
              'legend.fontsize': 20,
              'legend.handlelength': 2,
              'axes.labelsize': 20,
              'axes.titlesize': 20,
              'xtick.labelsize': 20,
              'ytick.labelsize': 20}
    plt.rcParams.update(params)
    normalize = mcolors.Normalize(vmin=0, vmax=np.max(2))
    colormap = cm.viridis

    np.random.seed(13)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)
    for j, name in enumerate(['wilt', 'spambase']):
        for i, (X_train, X_test,
                y_train, y_test, _) in enumerate(datasets.import_one_class_data(
                                                 name, 1, .5,
                                                 random_state=0)):
            axis = axes[j]
            nu_list = np.linspace(0, 1, 100)
            gamma_X = 1. / X_train.shape[1]
            gamma_t = pdist_quantile(nu_list, 'sqeuclidean',
                                     quantile=.2, gamma=True)
            if path != '':
                graph = path + '/' + name
            else:
                graph = path

            est = DensityEst(model=ITLModel(ExponentiatedChi2(gamma_X),
                                            Gaussian(gamma_t),
                                            Gaussian(gamma_t)),
                             sampler=GaussLegendreUniform_0p1(),
                             solver='L-BFGS-B',
                             lbda={'rkhs': 1e-3},
                             solver_param={'disp': 1},
                             device='/device:XLA_GPU:0',
                             summary={'path': graph})
            if case.casefold() == 'novelty':
                est.fit(X_train[y_train.ravel() == -1, :])
            elif case.casefold() == 'anomaly':
                est.fit(X_train)
            else:
                raise NotImplementedError('Choose anomaly or novelty')

            if case.casefold() == 'novelty':
                pred_train = est.predict(X_train[y_train.ravel() == -1, :],
                                         nu_list)
                pred_test = est.predict(X_test, nu_list)
            else:
                pred_train = est.predict(X_train, nu_list)
                pred_test = est.predict(X_test, nu_list)

            tpr = np.mean(pred_train > 0, axis=0)
            axis.plot(nu_list, tpr, label='Train',
                      color=colormap(normalize(0)),
                      marker=9)
            tpr = np.mean(pred_test > 0, axis=0)
            axis.plot(nu_list, tpr, label='Test',
                      color=colormap(normalize(2)),
                      marker='.')
            axis.plot(nu_list, 1 - nu_list, marker=',',
                      linestyle='dashed',
                      color=colormap(normalize(1)),
                      label='Oracle Train', c='r')
            axis.set_xlabel(r'$\theta$')
            axis.set_ylabel(r'Proportion of inliers')
            axis.set_title('Dataset: {}'.format(name))
            axis.legend(prop={'size': 24})

    plt.tight_layout()
    if show:
        plt.show()
    else:
        if not os.path.exists(prefix):
            os.mkdir(prefix)
        fig.savefig(prefix + '/iocsvm_nu.eps', bbox_inches='tight')
        fig.savefig(prefix + '/iocsvm_nu.pdf', bbox_inches='tight')

    return 0

if __name__ == '__main__':
    tf.app.flags.DEFINE_string('output_dir',
                               './iocsvm_nu_results/',
                               'Path to the directory used to store the '
                               'output files')
    tf.app.flags.DEFINE_string('save_graph',
                               '',
                               'Path to the directory used to store the '
                               'computation graph.')
    tf.app.flags.DEFINE_string('setting',
                               'anomaly',
                               '[anomaly, novelty], '
                               'Whether to run the experiment in the anomaly '
                               'or novelty detection setting')
    tf.app.flags.DEFINE_boolean('show',
                                False,
                                'Diplay on screen the generated figures')
    tf.app.run()
