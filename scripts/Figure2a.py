#!/usr/bin/env python

from netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import AutoMinorLocator
import matplotlib as mpl
import numpy as np

# CNN confidence interval (95%, bootstrap)
conf_level = [0.00142586, 0.0013108,  0.00240254, 0.00472927, 0.00650972, 0.0067122,
              0.00688624, 0.0088501,  0.01059312, 0.01154113, 0.01227826, 0.01225191,
              0.01160514, 0.01276475, 0.0140228,  0.01330945, 0.01300398, 0.014542,
              0.01392549, 0.01294643, 0.01569229, 0.01362468, 0.01339665]

cor_cnn = np.zeros((23, 12))

for i in range(0, 23, 1):
    for j in range(1, 13, 1):

        # Open label
        if i == 0 or i == 23:
            f = Dataset('C:/Users/zetal/PycharmProjects/PythonProject/data/GODAS/GODAS.label.12mn_2mv.1982_2017.nc', 'r')
            lab = f.variables['pr'][2:, int(j-1), 0, 0]
        else:
            f = Dataset('C:/Users/zetal/PycharmProjects/PythonProject/data/GODAS/GODAS.label.12mn_3mv.1982_2017.nc', 'r')
            lab = f.variables['pr'][2:, int(j-1), 0, 0]

        # Open CNN output (Transfer Learning)
        f = open('C:/Users/zetal/PycharmProjects/PythonProject/output/nino34_'+str(i+1)+'month_'+str(j)+'/combination.gdat', 'r')
        cnn = np.fromfile(f, dtype=np.float32)[2:]

        # Compute correlation coefficient
        pick_cnn = np.zeros((34), dtype=np.float32)
        pick_lab = np.zeros((34), dtype=np.float32)

        num = 0
        for k in range(34):
            if cnn[k] != -9.99e+08:
                pick_cnn[num] = cnn[k]
                pick_lab[num] = lab[k]
                num = num + 1

        cor_cnn[int(i), int(j-1)] = np.corrcoef(pick_lab[0:num], pick_cnn[0:num])[0, 1]

mean_cor_cnn = np.mean(cor_cnn, axis=1)

# Confidence interval
upper = mean_cor_cnn + np.array(conf_level)
lower = mean_cor_cnn - np.array(conf_level)

x = np.arange(0, 23, 1)

# Figure 2-(a) - CNN only
fig, ax = plt.subplots(figsize=(10, 4))

line = ax.plot(x, mean_cor_cnn, 'orangered', linewidth=1.4, marker='o', markersize=2, label='CNN')

ax.fill_between(x, lower, upper, facecolor='orangered', alpha=0.3, edgecolor=None)

ax.axhline(0.5, color='black', linewidth=0.5)

ax.legend(['CNN'], loc='upper right', prop={'size': 8})
ax.set_xlabel('Forecast Lead (months)', fontsize=7)
ax.set_ylabel('Correlation Skill', fontsize=7)
ax.set_xticks(np.arange(0, 23, 1))
ax.set_xticklabels(np.arange(1, 24, 1), fontsize=6)
ax.set_yticks(np.arange(0.3, 0.91, 0.1))
ax.set_ylim([0.25, 1.0])
ax.grid(linewidth=0.1, alpha=0.7)
ax.set_title('(a) All-season correlation skills for Nino3.4 (1984-2017)', fontsize=8, pad=10)
ax.tick_params(labelsize=6, direction='in', length=3, width=0.4, color='black')

plt.tight_layout()
plt.savefig('C:/Users/zetal/PycharmProjects/PythonProject/Figure_2a', dpi=1000)
plt.close()