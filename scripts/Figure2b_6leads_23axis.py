#!/usr/bin/env python

from netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.axes as ax
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
import matplotlib as mpl
import matplotlib.patches as patches
from matplotlib.colors import LogNorm

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import numpy as np

# --- CONFIGURACIÓN ---
n_leads = 6    # leads reales con datos
n_total = 23   # extensión del eje X

mon_list = [str(i) for i in range(1, n_leads + 2)]
mon_list_skip = [str(i) for i in range(1, n_leads + 2, 2)]
mon_name = ['JFM', 'FMA', 'MAM', 'AMJ', 'MJJ', 'JJA', 'JAS', 'ASO', 'SON', 'OND', 'NDJ', 'DJF']

cor_cnn = np.zeros((n_leads, 12))

for i in range(0, n_leads, 1):
    for j in range(1, 13, 1):

        # Open label
        if i == 0 or i == n_leads:
            f = Dataset('C:/Users/zetal/PycharmProjects/PythonProject/data/GODAS/GODAS.label.12mn_2mv.1982_2017.nc', 'r')
            lab = f.variables['pr'][2:, int(j - 1), 0, 0]
        else:
            f = Dataset('C:/Users/zetal/PycharmProjects/PythonProject/data/GODAS/GODAS.label.12mn_3mv.1982_2017.nc', 'r')
            lab = f.variables['pr'][2:, int(j - 1), 0, 0]

        # Open CNN output (Transfer Learning)
        f = open('C:/Users/zetal/PycharmProjects/PythonProject/C30H50_output/nino34_' + str(i + 1) + 'month_' + str(j) + '/C30H50/ensmean/result.gdat', 'r')
        cnn = np.fromfile(f, dtype=np.float32)[2:]

        # Compute correlation coefficient  ← SIN CAMBIOS
        pick_cnn = np.zeros((34), dtype=np.float32)
        pick_lab = np.zeros((34), dtype=np.float32)

        num = 0
        for k in range(34):
            if cnn[k] != -9.99e+08:
                pick_cnn[num] = cnn[k]
                pick_lab[num] = lab[k]
                num = num + 1

        cor_cnn[int(i), int(j - 1)] = np.corrcoef(pick_lab[0:num], pick_cnn[0:num])[0, 1]

mean_cor_cnn = np.mean(cor_cnn, axis=1)

cnn_map = np.swapaxes(cor_cnn, 0, 1)  # shape: (12, n_leads)  ← SIN CAMBIOS

# Extender a n_total columnas con NaN (zona en blanco 7-23)
cnn_map_full = np.full((12, n_total), np.nan)
cnn_map_full[:, :n_leads] = cnn_map

zm1 = np.ma.masked_less(np.ma.masked_invalid(cnn_map_full), 0.5)

x = np.arange(-0.5, n_total, 1)
y = np.arange(-0.5, 12, 1)

# Figure 2-(b)
plt.subplot2grid((2, 2), (0, 0), rowspan=1, colspan=1)

mpl.rcParams['hatch.linewidth'] = 0.45

im = plt.imshow(cnn_map_full, cmap='OrRd', vmin=0.0, vmax=1.0,
                extent=[-0.5, n_total - 0.5, -0.5, 11.5], origin='lower',
                aspect='auto')

plt.pcolormesh(x, y, zm1, hatch='/////', facecolor='white', edgecolor='k',
               linewidth=0.0, alpha=0.01, zorder=10, shading='auto')

plt.contourf(np.arange(n_total), np.arange(12), cnn_map_full,
             levels=[0.5, 1.0], colors='none', hatches=['////'])

# Eje Y: sin cambios respecto al original
plt.yticks(np.arange(0, 12, 1), mon_name, fontsize=5)
plt.ylabel('Target season', fontsize=7)
ax = plt.gca()
ax.yaxis.set_label_coords(-0.22, 0.5)

# Eje X: ticks impares 1,3,5,...,23 para cubrir todo el rango
x_ticks  = np.arange(0, n_total, 2)       # índices 0,2,4,...,22
x_labels = np.arange(1, n_total + 1, 2)   # etiquetas 1,3,5,...,23
plt.xticks(x_ticks, x_labels, fontsize=4)
plt.xlabel('Forecast lead (months)', fontsize=7)
ax.xaxis.set_label_coords(0.5, -0.11)

ax.set_xlim(-0.5, n_total - 0.5)
ax.set_ylim(-0.5, 11.5)

plt.title('b', fontsize=9, fontweight='bold', loc='left', pad=4)

plt.subplots_adjust(left=0.18)
plt.tick_params(labelsize=6, direction='in', length=2, width=0.3, color='black', right=True)

cax = plt.axes([0.75, 0.42, 0.015, 0.28])
cbar = plt.colorbar(im, cax=cax, orientation='vertical')
cbar.ax.tick_params(labelsize=6, direction='out', length=2, width=0.4, color='black')

plt.savefig('C:/Users/zetal/PycharmProjects/PythonProject/Figure_2_6leads_23axis', dpi=1000,
            bbox_inches='tight')
plt.close()
