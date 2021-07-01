import sys
import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np

from IPython import embed
from itl import *
from sklearn.preprocessing import StandardScaler

NEW_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']


def main(argv=None):
    if argv is None:
        argv = sys.argv

    if argv is None:
        argv = sys.argv

    prefix = argv[2] if len(argv) > 2 else './results/'

    matplotlib.use('TKagg')
    np.random.seed(13)

    xx, yy = np.meshgrid(np.linspace(-5, 5, 200), np.linspace(-5, 5, 200))
    # Generate train data
    X = 0.3 * np.random.randn(100, 2)
    X_train = np.r_[X + 2, X - 2]
    # Generate some regular novel observations
    X = 0.3 * np.random.randn(20, 2)
    X_test = np.r_[X + 2, X - 2]
    # Generate some abnormal novel observations
    X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))

    scaler_x = StandardScaler()

    npx_train = scaler_x.fit_transform(X_train)
    npx_test = scaler_x.transform(X_test)
    npx_outliers = scaler_x.transform(X_outliers)

    gamma_t = pdist_quantile(GaussLegendreUniform_0p1(), 'sqeuclidean',
                             quantile=.2, gamma=True)
    #print()
    nu_list = np.array([.1])
    est = DensityEst(model=ITLModel(Gaussian(4.),
                                    #Impulse(), Impulse()),
                                    Gaussian(gamma_t), Gaussian(gamma_t)),
                     sampler=GaussLegendreUniform_0p1(),
                     #sampler=Dirac(nu_list),
                     solver='L-BFGS-B',
                     solver_param={'disp': 1}, device='gpu',
                     summary={'path': ''})
    est.fit(npx_train)

    y_pred_train = est.predict(npx_train, nu_list)
    y_pred_test = est.predict(npx_test, nu_list)
    y_pred_outliers = est.predict(npx_outliers, nu_list)
    n_error_train = y_pred_train[y_pred_train == -1].size
    n_error_test = y_pred_test[y_pred_test == -1].size
    n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size
    Z = est.decision_function(scaler_x.transform(np.c_[xx.ravel(),
                                                       yy.ravel()]), nu_list)
    Z = Z.reshape(xx.shape)

    fig = plt.figure()
    xxx = npx_train[0:1, :]
    ttt = np.linspace(0, 1, 100)
    pred = est.decision_function(xxx, ttt)
    print(ttt, pred)
    plt.scatter(ttt.reshape(xxx.shape[0], -1), pred)
    plt.axhline(y=0, c='red')
    plt.show()
    return

    fig = plt.figure()
    plt.title("Novelty Detection")
    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7),
                 cmap=plt.cm.PuBu)
    a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
    plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')

    s = 40
    b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white',
                     s=s, edgecolors='k')
    b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='blueviolet', s=s,
                     edgecolors='k')
    c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='gold', s=s,
                    edgecolors='k')
    plt.axis('tight')
    plt.xlim((-5, 5))
    plt.ylim((-5, 5))
    plt.legend([a.collections[0], b1, b2, c],
               ["learned frontier", "training observations",
                "new regular observations", "new abnormal observations"],
               loc="upper left",
               prop=matplotlib.font_manager.FontProperties(size=11))
    plt.xlabel(
        "error train: %d/200 ; errors novel regular: %d/40 ; "
        "errors novel abnormal: %d/40"
        % (n_error_train, n_error_test, n_error_outliers))
    plt.show()

    return 0

if __name__ == '__main__':
    sys.exit(main())
