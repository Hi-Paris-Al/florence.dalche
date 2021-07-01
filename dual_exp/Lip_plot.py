import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

plt.rcParams['mathtext.fontset'] = "cm"
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['text.usetex'] = True

err = np.load('Results/Lip_err.npy')
M_list = np.array([4, 5, 6, 7, 15])
kappa_list = np.linspace(0.05, 2, 50)
colors = [cm.viridis(x) for x in np.linspace(0, 1, M_list.shape[0])]

fig = plt.figure()

for j in range(M_list.shape[0]):

    # ridge error
    plt.hlines(err[j][-1], 0, 1.5, colors='grey', linestyles='--')

    # Huber error
    plt.plot(kappa_list[:-10], err[j][:-10], '-',
             color=colors[j], label='m=' + str(M_list[j]))

# for legend
plt.hlines(err[0][-1], 0, 1.5, colors='grey',
           linestyles='--', label='Ridge Regression ($\kappa = +\infty$)')

plt.tick_params(labelsize=16)
plt.xlabel('$\kappa$', fontsize=20)
plt.ylabel('LOO generalization error', fontsize=18)
plt.ylim(np.min(err) - 0.01, np.max(err))
plt.legend(fontsize=10)

fig.savefig('Plots/Lip_huber.pdf', bbox_inches='tight')
