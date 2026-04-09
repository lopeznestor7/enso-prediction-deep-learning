#!/usr/bin/env python

from netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.axes as ax
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator)
import matplotlib as mpl
import matplotlib.patches as patches
from matplotlib.colors import LogNorm

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import numpy as np
             
mon_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 
            '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24']
mon_list_skip = ['1', '3', '5', '7', '9', '11', '13', '15', '17', '19', '21', '23']
mon_name = ['JFM','FMA','MAM','AMJ','MJJ','JJA','JAS','ASO','SON','OND','NDJ','DJF']

cor_cnn = np.zeros((23,12))


for i in range(0,23,1):
  for j in range(1,13,1):

    # Open label
    if i == 0 or i ==23:
      f = Dataset('C:/Users/zetal/PycharmProjects/PythonProject/data/GODAS/GODAS.label.12mn_2mv.1982_2017.nc','r')
      lab = f.variables['pr'][2:,int(j-1),0,0]
    else:
      f = Dataset('C:/Users/zetal/PycharmProjects/PythonProject/data/GODAS/GODAS.label.12mn_3mv.1982_2017.nc','r')
      lab = f.variables['pr'][2:,int(j-1),0,0]

    # Open CNN output (Transfer Learning)
    f = open('C:/Users/zetal/PycharmProjects/PythonProject/output/nino34_'+str(i+1)+'month_'+str(j)+'/combination.gdat','r')
    cnn = np.fromfile(f, dtype=np.float32)[2:]


    # Compute correlation coefficient
    pick_cnn = np.zeros((34),dtype=np.float32)
    pick_lab = np.zeros((34),dtype=np.float32)
    
    num = 0
    for k in range(34):
      if cnn[k] != -9.99e+08:
        pick_cnn[num] = cnn[k]
        pick_lab[num] = lab[k]
          
        num = num + 1

    cor_cnn[int(i),int(j-1)]    = np.corrcoef(pick_lab[0:num],pick_cnn[0:num])[0,1]
    

mean_cor_cnn =   np.mean(cor_cnn,axis=1)

cnn_map = np.swapaxes(cor_cnn,0,1)

zm1 = np.ma.masked_less(cnn_map,0.5)

# x = np.arange(-0.5,22)
# y = np.arange(-0.5,12)

x = np.arange(-0.5,23,1)
y = np.arange(-0.5,12,1)


# Figure 2-(b)
plt.subplot2grid((2,2),(0,0),rowspan=1,colspan=1)


mpl.rcParams['hatch.linewidth'] = 0.45

# plt.imshow(cnn_map,cmap='OrRd',clim=[0.0,1.0])

# plt.imshow(cnn_map, cmap='OrRd', clim=[0.0,1.0], extent=[-0.5, 22.5, -0.5, 11.5], origin='lower')

im = plt.imshow(cnn_map, cmap='OrRd', vmin=0.0, vmax=1.0, extent=[-0.5, 22.5, -0.5, 11.5], origin='lower')


# plt.pcolor(x, y, zm1, hatch='/////', alpha=0.,zorder=4)

# plt.pcolormesh(x, y, zm1, hatch='/////', alpha=0., zorder=4, shading='auto')

plt.pcolormesh(x, y, zm1, hatch='/////', facecolor='white', edgecolor='k', linewidth=0.0, alpha=0.01, zorder=10, shading='auto')

plt.contourf(cnn_map, levels=[0.5, 1.0], colors='none', hatches=['////'], extent=[-0.5, 22.5, -0.5, 11.5], origin='lower')

# plt.gca().invert_yaxis()

plt.yticks(np.arange(0,12,1),mon_name,fontsize=5)
plt.xticks(np.arange(0,23,2),np.arange(1,24,2), fontsize=4)
plt.ylabel('Target season',fontsize=7)
ax=plt.gca()
ax.yaxis.set_label_coords(-0.11, 0.5)
ax.xaxis.set_label_coords(0.5, -0.11)

plt.xlabel('Forecast lead (months)',fontsize=7)
plt.title('(b) Correlation Skill - CNN', fontsize=8, x=0.29, y=0.96)
plt.tick_params(labelsize=6,direction='in',length=2,width=0.3,color='black',right=True)
cax = plt.axes([0.75, 0.42, 0.015, 0.28])

# cbar = plt.colorbar(cax=cax,orientation='vertical')

cbar = plt.colorbar(im, cax=cax, orientation='vertical')

cbar.ax.tick_params(labelsize=6,direction='out',length=2,width=0.4,color='black')


plt.savefig('C:/Users/zetal/PycharmProjects/PythonProject/Figure_2b', dpi=1000)
plt.close()

