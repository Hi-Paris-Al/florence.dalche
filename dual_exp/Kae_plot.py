import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

plt.rcParams['mathtext.fontset'] = "cm"
plt.rcParams['pdf.fonttype'] = 42

res = np.load('Results/kae.npy')

fig, ax1 = plt.subplots(figsize=(8, 5.6))
ax2 = ax1.twinx()
colors = [cm.viridis(x) for x in [0.1, 0.9]]


# plot epsilon KAE test MSE (purple)
ax1.plot(res[0, :-1], res[2, :-1],
         '-', label='$\epsilon$-KAE', linewidth=2.5, c=colors[0])


# plot standard KAE test MSE (dashed grey)
ax1.axhline(y=res[2, 0], linestyle='--', color='gray', linewidth=2.5,
            label='standard KAE')


# plot W_21 norm of epsilon KAE (red)
ax2.plot(res[0, :-1], res[3, :-1], color='indianred',
         label="W's $\ell_{2, 1}$ norm", linewidth=2.5)


# plot null components (grey bars)
n_comp = (np.array(res[4, :-1]) / 1.5).astype('int')

bar_plot = plt.bar(res[0, :-1], n_comp, width=3e-2, color='lightgrey',
                   label='discarded data')

for i in range(res.shape[1] - 1):
    plt.text(x=res[0, i] - 0.025, y=n_comp[i] + 30, s=n_comp[i], size=10)


# format axes and legend
ax1.set_xlabel('$\epsilon$', fontsize=18)

ax1.set_ylim(0.5, 1)
ax1.set_ylabel('Test reconstruction MSE', fontsize=14, color=colors[0])
ax1.tick_params('y', colors=colors[0])

ax2.set_ylabel('$||W||_{2,1}$', fontsize=14, color='indianred')
ax2.tick_params(colors='indianred')

fig.legend(loc='upper center', fontsize=14, bbox_to_anchor=(0.5, 1),
           bbox_transform=ax1.transAxes)

fig.savefig('Plots/kae_eps.pdf', bbox_inches='tight')
