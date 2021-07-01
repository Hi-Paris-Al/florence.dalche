import matplotlib

import sys
import os

import numpy as np
import tensorflow as tf

from matplotlib.colors import Normalize
from itl import *
from sklearn.preprocessing import StandardScaler
from scipy.stats import uniform
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.model_selection import train_test_split


def main(argv=None):
    prefix = tf.app.flags.FLAGS.output_dir
    path = tf.app.flags.FLAGS.save_graph
    show = tf.app.flags.FLAGS.show
    n_splits = tf.app.flags.FLAGS.n_splits
    if n_splits < 1:
        raise RuntimeError("n_splits must be greater than 1.")
    if argv is None:
        argv = sys.argv
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
    np.set_printoptions(threshold=np.inf)

    np.random.seed(13)

    #X, y = make_circles(n_samples=1000, noise=.1, random_state=13)
    #X, y = make_moons(n_samples=1000, noise=.4, random_state=13)
    X, y = make_classification(n_samples=1000,
                               n_features=20, n_redundant=4, n_informative=10,
                               random_state=13, class_sep=.5)
    #iris = load_iris()
    #y = iris.target[iris.target > 0] - 1
    #X = iris.data[iris.target > 0, :]  # we only take the first two features for visualization

    #print(gamma)

    # Now we need to fit a classifier for all parameters in the 2d version (we
    # use a smaller set of parameters here because it takes a while to train)
    cost_list = [-.9, 0, .9]

    # Visualization
    #
    # draw visualization of parameter effects
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - .1,
                                     X[:, 0].max() + .1, 200),
                         np.linspace(X[:, 1].min() - .1,
                                     X[:, 1].max() + .1, 200))
    Z = []
    if path != '':
        graph = path + '/iqr'
    else:
        graph = path
    tp = []
    tn = []
    sensitivity = []
    specificity = []
    for i in range(n_splits):
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.5,
                                                            random_state=i)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        gamma1 = pdist_quantile(X_train,
                                gamma=True, quantile=.5)
        gamma2 = pdist_quantile(np.linspace(0, 1, 20),
                                gamma=True, quantile=.2)
        clf = CSSVM(model=ITLModel(Gaussian(gamma1), Gaussian(gamma2)),
                    solver='L-BFGS-B',
                    sampler=GaussLegendreUniform_m1p1(20),
                    lbda={'rkhs': 1e-3, 'p_rkhs': 0},
                    solver_param={'disp': 0},
                    summary={'path': graph})
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test, cost_list)
        tp.append(np.sum((pred > 0)[y_test.ravel() == 1, :], axis=0))
        tn.append(np.sum((pred <= 0)[y_test.ravel() == 0, :], axis=0))
        sensitivity.append(tp[i] / np.sum(y_test == 1))
        specificity.append(tn[i] / np.sum(y_test == 0))
    sensitivity = np.array(sensitivity)
    specificity = np.array(specificity)

    if X.shape[1] == 2:
        for (k, cost) in enumerate(cost_list):
            # evaluate decision function in a grid
            Z.append(clf.decision_function(np.c_[xx.ravel(), yy.ravel()],
                                           cost))
            Z[k] = Z[k].reshape(xx.shape)

        Z = np.array(Z)
        fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey=True, sharex=True)
        for (k, cost) in enumerate(cost_list):
            # visualize decision function for these parameters
            axis = axes[0, k]
            se = np.mean(sensitivity[:, k])
            sp = np.mean(specificity[:, k])
            se_std = np.std(sensitivity[:, k])
            sp_std = np.std(specificity[:, k])
            print('Continuum (se, sp) {}: {} {}'.format(cost,
                                                        sensitivity[:, k],
                                                        specificity[:, k]))
            axis.set_title(r"sensitivity=${0:.2f}\pm{1:.2f}$".format(se, se_std) + "\n" +
                           r"specificity=${0:.2f}\pm{1:.2f}$".format(sp, sp_std))

            # visualize parameter's effect on decision function
            axis.pcolormesh(xx, yy, -Z[k, :, :], cmap=plt.cm.viridis,
                            norm=Normalize(Z.min(), Z.max()))
            mask = (y_train.ravel() == 0)
            axis.scatter(scaler.inverse_transform(X_train)[mask, 0],
                         scaler.inverse_transform(X_train)[mask, 1],
                         c=y_train.ravel()[mask],
                         cmap=plt.cm.viridis_r,
                         edgecolors='k', marker='o')
            mask = (y_train.ravel() == 1)
            axis.scatter(scaler.inverse_transform(X_train)[mask, 0],
                         scaler.inverse_transform(X_train)[mask, 1],
                         c=y_train.ravel()[mask],
                         cmap=plt.cm.viridis,
                         edgecolors='k', marker='^')
            axis.set_xticks(())
            axis.set_yticks(())
            axis.set_ylabel(r'Continuum: $\theta={}$'.format(cost))
    else:
        for (k, cost) in enumerate(cost_list):
            se = np.mean(sensitivity[:, k])
            sp = np.mean(specificity[:, k])
            se_std = np.std(sensitivity[:, k])
            sp_std = np.std(specificity[:, k])
            print(se, se_std, sp, sp_std)

    Z = []
    if path != '':
        graph = path + '/ind'
    else:
        graph = path
    tp = []
    tn = []
    sensitivity = []
    specificity = []
    for i in range(n_splits):
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.5,
                                                            random_state=i)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        gamma1 = pdist_quantile(X_train, gamma=True, quantile=.5)
        gamma2 = pdist_quantile(np.linspace(0, 1, 20), gamma=True, quantile=.2)
        clf = CSSVM(model=ITLModel(Gaussian(gamma1), Impulse()),
                    solver='L-BFGS-B',
                    sampler=Dirac(cost_list),
                    lbda={'rkhs': 1e-3, 'p_rkhs': 0},
                    solver_param={'disp': 0},
                    summary={'path': graph})
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test, cost_list)
        tp.append(np.sum((pred > 0)[y_test.ravel() == 1, :], axis=0))
        tn.append(np.sum((pred <= 0)[y_test.ravel() == 0, :], axis=0))
        sensitivity.append(tp[i] / np.sum(y_test == 1))
        specificity.append(tn[i] / np.sum(y_test == 0))
    sensitivity = np.array(sensitivity)
    specificity = np.array(specificity)

    if X.shape[1] == 2:
        for (k, cost) in enumerate(cost_list):
            # evaluate decision function in a grid
            Z.append(clf.decision_function(np.c_[xx.ravel(), yy.ravel()],
                                           cost))
            Z[k] = Z[k].reshape(xx.shape)

        Z = np.array(Z)
        for (k, cost) in enumerate(cost_list):
            # visualize decision function for these parameters
            axis = axes[1, k]
            se = np.mean(sensitivity[:, k])
            sp = np.mean(specificity[:, k])
            print('Independent (se, sp) {}: {} {}'.format(cost,
                                                          sensitivity[:, k],
                                                          specificity[:, k]))
            se_std = np.std(sensitivity[:, k])
            sp_std = np.std(specificity[:, k])
            axis.set_title(r"sensitivity=${0:.2f}\pm{1:.2f}$".format(se, se_std) + "\n" +
                           r"specificity=${0:.2f}\pm{1:.2f}$".format(sp, sp_std))

            # visualize parameter's effect on decision function
            axis.pcolormesh(xx, yy, -Z[k, :, :], cmap=plt.cm.viridis,
                            norm=Normalize(Z.min(), Z.max()))
            mask = (y_train.ravel() == 0)
            axis.scatter(scaler.inverse_transform(X_train)[mask, 0],
                         scaler.inverse_transform(X_train)[mask, 1],
                         c=y_train.ravel()[mask],
                         cmap=plt.cm.viridis_r,
                         edgecolors='k', marker='o')
            mask = (y_train.ravel() == 1)
            axis.scatter(scaler.inverse_transform(X_train)[mask, 0],
                         scaler.inverse_transform(X_train)[mask, 1],
                         c=y_train.ravel()[mask],
                         cmap=plt.cm.viridis,
                         edgecolors='k', marker='^')
            axis.set_xticks(())
            axis.set_yticks(())
            axis.set_ylabel(r'Independent: $\theta={}$'.format(cost))

        plt.tight_layout()

        if show:
            plt.show()
        else:
            if not os.path.exists(prefix):
                os.mkdir(prefix)
            fig.savefig(prefix + '/icsl_vs.eps', bbox_inches='tight')
            fig.savefig(prefix + '/icsl_vs.pdf', bbox_inches='tight')
    else:
        for (k, cost) in enumerate(cost_list):
            se = np.mean(sensitivity[:, k])
            sp = np.mean(specificity[:, k])
            se_std = np.std(sensitivity[:, k])
            sp_std = np.std(specificity[:, k])
            print(se, se_std, sp, sp_std)

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
    tf.app.flags.DEFINE_integer('n_splits',
                                50,
                                'Number of train-test splits')
    tf.app.run()
