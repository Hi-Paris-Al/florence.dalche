import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors2

plt.rcParams['mathtext.fontset'] = "cm"
plt.rcParams['pdf.fonttype'] = 42


# Algorithm specific parameters
algo = sys.argv[1]

if algo == 'e_krr':
    params = np.linspace(1e-1, 1, 10)
    n_params = len(params)
    MSEs = np.load('Results/yeast_krr_MSEs.npy')
    Spar = np.load('Results/yeast_krr_Spar.npy')

    cm = plt.cm.get_cmap('viridis')
    colors = []
    for i in range(n_params):
        colors.append(cm(i / (n_params - 1)))
    idx_mini = 0
    sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=min(params),
                               vmax=max(params)))

    param_str = '$\epsilon$'
    format_str = [' = %.1f'] * n_params
    krr_label = 'KRR ($\epsilon=0$)'
    title_1 = 'Comparison  $\epsilon$-KRR / KRR'
    path_1 = 'Plots/yeast_krr_mse.pdf'
    title_2 = ('Sparsity w.r.t. $\Lambda$ for different $\epsilon$ ' +
               '($\epsilon$-KRR)')
    y_axis_2 = 'Sparsity (% null components)'
    path_2 = 'Plots/yeast_krr_sparsity.pdf'


elif algo == 'e_svr':
    params = np.linspace(1e-1, 1, 10)
    n_params = len(params)
    MSEs = np.load('Results/yeast_svr_MSEs.npy')
    Spar = np.load('Results/yeast_svr_Spar.npy')

    cm = plt.cm.get_cmap('viridis')
    colors = []
    for i in range(n_params):
        colors.append(cm(i / (n_params - 1)))
    idx_mini = 0
    sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=min(params),
                               vmax=max(params)))

    param_str = '$\epsilon$'
    format_str = [' = %.1f'] * n_params
    krr_label = 'KRR'
    title_1 = 'Comparison  $\epsilon$-SVR / KRR'
    path_1 = 'Plots/yeast_svr_mse.pdf'
    title_2 = ('Sparsity w.r.t. $\Lambda$ for different $\epsilon$ ' +
               '($\epsilon$-SVR)')
    y_axis_2 = 'Sparsity (% null components)'
    path_2 = 'Plots/yeast_svr_sparsity.pdf'


elif algo == 'k_huber':
    params = np.logspace(-3, 1, 16)
    n_params = len(params)
    MSEs = np.load('Results/yeast_huber_MSEs.npy')
    Spar = np.load('Results/yeast_huber_Spar.npy')

    cm_reversed = plt.cm.get_cmap('viridis_r')
    colors = []
    for i in range(n_params):
        colors.append(cm_reversed(i / (n_params - 1)))
    idx_mini = 9
    sm = plt.cm.ScalarMappable(cmap=cm_reversed,
                               norm=colors2.LogNorm(vmin=min(params),
                                                    vmax=max(params)))

    param_str = '$\kappa$'
    format_str = [' = %.1f'] * n_params
    krr_label = 'KRR ($\kappa=+\infty$)'
    title_1 = 'Comparison  $\kappa$-Huber / KRR'
    path_1 = 'Plots/yeast_huber_mse.pdf'
    title_2 = ('Saturation w.r.t. $\Lambda$ for different $\kappa$ ')
    y_axis_2 = 'Saturation (%  saturated components)'
    path_2 = 'Plots/yeast_huber_saturation.pdf'


# Commom parameters
lambdas = np.logspace(-8, -1, 20)
n_lambdas = len(lambdas)


# Plot MSEs
plt.figure()

plt.semilogx(lambdas, MSEs[:, -1], label=krr_label, lw=3,
             color='black', zorder=10)

for i in range(n_params):
    param = params[i]
    plt.semilogx(lambdas, MSEs[:, i], color=colors[i])

plt.axhline(y=np.min(MSEs[:, idx_mini]) - 0.0035, color=colors[idx_mini],
            ls='dashed')
plt.ylim(2., 2.6)

cbar = plt.colorbar(sm)
cbar.ax.set_ylabel('   ' + param_str, fontsize=16, rotation=0)

plt.legend()
plt.ylabel('Test MSE', fontsize=16)
plt.xlabel('$\Lambda$', fontsize=16)

plt.title(title_1)
plt.savefig(path_1)


# Plot Sparsity / Saturation
plt.figure()

for i in np.flip(range(n_params)):
    param = params[i]
    plt.semilogx(lambdas, 100 * Spar[:, i], marker='o', color=colors[i])

cbar = plt.colorbar(sm)
cbar.ax.set_ylabel('   ' + param_str, fontsize=16, rotation=0)

plt.ylabel(y_axis_2, fontsize=12)
plt.xlabel('$\Lambda$', fontsize=16)

plt.title(title_2)
plt.savefig(path_2)
