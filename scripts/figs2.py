#!/usr/bin/env python
#from matplotlib.pyplot import figure
from netCDF4 import Dataset
import matplotlib.pyplot as plt
#import matplotlib.ticker as ticker
#import matplotlib.axes as ax
from matplotlib.ticker import (AutoMinorLocator)
import matplotlib as mpl
#import matplotlib.patches as patches
#from matplotlib.colors import LogNorm
#from mpl_toolkits.basemap import Basemap, cm, shiftgrid, addcyclic

import numpy as np

plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.dpi'] = 200


minorLocator = AutoMinorLocator()

deg = u'\xb0'

mon_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
            '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24']
mon_list_skip = ['1', '3', '5', '7', '9', '11', '13', '15', '17', '19', '21', '23']
mon_name = ['JFM', 'FMA', 'MAM', 'AMJ', 'MJJ', 'JJA', 'JAS', 'ASO', 'SON', 'OND', 'NDJ', 'DJF']

cor_cnn = np.zeros((23, 12))

for i in range(0, 23, 1): # LEAD: 0 ... 22
    for j in range(1, 13, 1): # MONTH: 1 ... 12
        print(i, j)
        # Open label
        if i == 0 or i == 23:
            f = Dataset('/home/aruiz/Descargas/Ham_2019_DL_ENSO-v1.0/jeonghwan723-DL_ENSO-bbbdd6d/CNN/Dataset_H19/GODAS/GODAS.label.12mn_2mv.1982_2017.nc', 'r')
            lab = f.variables['pr'][2:, int(j - 1), 0, 0] # desde el 84 (82-83 no) ----> 34 años
            #print(len(lab))
        else:
            f = Dataset('/home/aruiz/Descargas/Ham_2019_DL_ENSO-v1.0/jeonghwan723-DL_ENSO-bbbdd6d/CNN/Dataset_H19/GODAS/GODAS.label.12mn_3mv.1982_2017.nc', 'r')
            lab = f.variables['pr'][2:, int(j - 1), 0, 0]


        # Open CNN output (Transfer Learning)
        g = open('/home/aruiz/Proyectos/CNN_EN_pred/Nino12_pred/trial/output/nino12_lead_' + str(i + 1) + '_month_' + str(j) + '/C30H50/ensmean/pred_ensmean.npy', 'r')
        cnn_2 = np.fromfile(g, dtype=np.float32)[32 + 2:]

        f = open('/home/aruiz/Descargas/Ham_2019_DL_ENSO-v1.0/jeonghwan723-DL_ENSO-bbbdd6d/CNN/output/nino34_' + str(i + 1) + 'month_' + str(j) + '/combination.gdat',
                 'r')
        cnn = np.fromfile(f, dtype=np.float32)[2:]

        # Compute correlation coefficient
        pick_cnn = np.zeros(34, dtype=np.float32)
        pick_lab = np.zeros(34, dtype=np.float32)
        num = 0

        for k in range(34):
            pick_cnn[num] = cnn_2[k]
            pick_lab[num] = lab[k]

            num = num + 1

        cor_cnn[int(i), int(j - 1)] = np.corrcoef(pick_lab[0:num], pick_cnn[0:num])[0, 1]
        print(len(pick_lab[0:num]),len(pick_cnn[0:num]))

    #print(cor_cnn)



mean_cor_cnn = np.mean(cor_cnn, axis=1)

# set confidence interval

cnn_map = np.swapaxes(cor_cnn, 0, 1)



# Figure 2-(a)
plt.subplot2grid((2, 2), (0, 0), rowspan=1, colspan=2)
x = np.arange(0, 23, 1)
y = np.arange(0, 12, 1)
z = np.arange(0, 9, 1)
lines = plt.plot(x, mean_cor_cnn, 'orangered')

my_plot = plt.gca()
line0 = my_plot.lines[0]
#line1 = my_plot.lines[1]


plt.setp(lines, linewidth=1.2, marker='v', markersize=2)
plt.setp(line0, linewidth=1.4, marker='o', markersize=2)
#plt.setp(line1, linewidth=1.4, marker='o', markersize=2)

model_list = ['CNN']

plt.legend(model_list, loc='upper right', prop={'size': 6}, ncol=5)
plt.xlabel('Forecast Lead (months)', fontsize=7)
plt.ylabel('Correlation Skill', fontsize=7)
plt.xticks(np.arange(0, 23, 1), np.arange(1, 24, 1), fontsize=6)
plt.yticks(np.arange(0.3, 0.91, 0.1), fontsize=6)
plt.ylim([0.25, 1.0])
plt.grid(linewidth=0.1, alpha=0.7)
plt.axhline(0.5, color='black', linewidth=0.5)
plt.title('(a) All-season correlation skills for Nino3.4 (1984-2017)', fontsize=8, x=0.3, y=0.97)
plt.tick_params(labelsize=6, direction='in', length=3, width=0.4, color='black')
zm1 = np.ma.masked_less(cnn_map, 0.5)
x = np.arange(-0.5, 22)
#y = np.arange(-0.5, 12)

# Figure 2-(b)
plt.subplot2grid((2, 2), (1, 0), rowspan=1, colspan=1)
plt.pcolor(x, y, zm1, hatch='/////', alpha=0., zorder=4)
mpl.rcParams['hatch.linewidth'] = 0.45
plt.imshow(cnn_map, cmap='OrRd', clim=[0.0, 1.0])
plt.gca().invert_yaxis()
plt.yticks(np.arange(0, 12, 1), mon_name, fontsize=5)
plt.xticks(np.arange(0, 23, 2), np.arange(1, 24, 2), fontsize=4)
plt.ylabel('Target season', fontsize=7)
ax = plt.gca()
ax.yaxis.set_label_coords(-0.11, 0.5)
ax.xaxis.set_label_coords(0.5, -0.11)
plt.xlabel('Forecast lead (months)', fontsize=7)
plt.title('(b) Correlation Skill - CNN', fontsize=8, x=0.29, y=0.96)
plt.tick_params(labelsize=6, direction='in', length=2, width=0.3, color='black', right=True)
cax = plt.axes([0.55, 0.15, 0.015, 0.28])
cbar = plt.colorbar(cax=cax, orientation='vertical')
cbar.ax.tick_params(labelsize=6, direction='out', length=2, width=0.4, color='black')

plt.tight_layout(h_pad=0.5, w_pad=0.2)
#plt.subplots_adjust(left=0.08, right=0.9, bottom=0.12, top=0.94)
#plt.savefig('/home/jhkim/analysis/fig/Figure_2', dpi=1000)
#plt.close()









f = open('/home/aruiz/Descargas/Ham_2019_DL_ENSO-v1.0/jeonghwan723-DL_ENSO-bbbdd6d/CNN/output/nino34_8month_10/combination.gdat','r')
cnn_1 = np.fromfile(f, dtype=np.float32)[2:]


g = open('/home/aruiz/Proyectos/CNN_EN_pred/Nino12_pred/trial/output/nino12_lead_4_month_5/C30H50/EN1/pred_eval.npy', 'r')
cnn_2 = np.fromfile(g, dtype=np.float32)[32+2:]
print(cnn_2)



f = Dataset('/home/aruiz/Descargas/Ham_2019_DL_ENSO-v1.0/jeonghwan723-DL_ENSO-bbbdd6d/CNN/Dataset_H19/GODAS/GODAS.label.12mn_2mv.1982_2017.nc', 'r')
f.variables['pr'][2:, int(j - 1), 0, 0]


















# !/usr/bin/env python
from netCDF4 import Dataset
from tempfile import TemporaryFile
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl
from matplotlib.colors import LogNorm
from mpl_toolkits.basemap import Basemap, cm, shiftgrid, addcyclic
import numpy as np

plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.dpi'] = 200

deg = u'\xb0'

CH_list = ['C30H30', 'C30H50', 'C50H30', 'C50H50']

# Open CNN (1981-2017)
g = open('/home/aruiz/Proyectos/CNN_EN_pred/Nino12_pred/trial/output/nino12_lead_18_month_12/C30H50/ensmean/pred_ensmean.npy', 'r')
cnn_2 = np.fromfile(g, dtype=np.float32)[32:]

#f = open('/home/aruiz/Descargas/Ham_2019_DL_ENSO-v1.0/jeonghwan723-DL_ENSO-bbbdd6d/CNN/output/nino34_18month_12/combination.gdat', 'r')
#cnn = np.fromfile(f, dtype=np.float32)


# Open observation (GODAS, 1981-2017)
f = Dataset('/home/aruiz/Descargas/Ham_2019_DL_ENSO-v1.0/jeonghwan723-DL_ENSO-bbbdd6d/CNN/Dataset_H19/GODAS/GODAS.label.12mn_3mv.1982_2017.nc', 'r')
obs = f.variables['pr'][:, 11, 0, 0]

cnn_2 = cnn_2 / np.std(cnn_2)
obs = obs / np.std(obs)

# open SST & HC map
f = Dataset('/home/aruiz/Descargas/Ham_2019_DL_ENSO-v1.0/jeonghwan723-DL_ENSO-bbbdd6d/CNN/Dataset_H19/GODAS/GODAS.input.36mn.1980_2015.nc', 'r')
osst = np.mean(f.variables['sst'][:, 4:7, :, :], axis=1) # MJJ
ot300 = np.mean(f.variables['t300'][:, 4:7, :, :], axis=1)


# Compute correlation coefficient (1984-2017)
cor_cnn = np.round(np.corrcoef(obs[3:], cnn_2[3:])[0, 1], 2)

# Draw Figure
# Figure 3-(a)
plt.figure()
plt.subplot(1, 1, 1)
x = np.arange(1, 37)
y = np.arange(4, 37)
lines = plt.plot(x, obs, 'black', x, cnn_2, 'orangered')
my_plot = plt.gca()
line0 = my_plot.lines[0]
line1 = my_plot.lines[1]
plt.setp(line0, linewidth=2)
plt.setp(line1, linewidth=0.5, marker='o', markersize=2)
plt.legend(('Observation', 'CNN(Cor=' + str(cor_cnn) + ')'), loc='upper right',
           prop={'size': 7}, ncol=3)
plt.xlim([0, 37])
plt.ylim([-3, 3.5])
plt.xticks(np.arange(2, 39, 2), np.arange(1982, 2019, 2), fontsize=6.5)
plt.tick_params(labelsize=6., direction='in', length=2, width=0.3, color='black')

plt.yticks(np.arange(-3, 3.51, 1), fontsize=6.5)
plt.grid(linewidth=0.2, alpha=0.7)
plt.axhline(0, color='black', linewidth=0.5)
plt.title('(a) 18-month lead prediction for DJF Nino3.4', fontsize=8, x=0.268, y=0.967)
plt.xlabel('Year', fontsize=7)
plt.ylabel('DJF Nino3.4', fontsize=7)

