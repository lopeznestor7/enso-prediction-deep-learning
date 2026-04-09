#!/usr/bin/env python
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np

# Open CNN (1981-2017)
f = open('C:/Users/zetal/PycharmProjects/PythonProject/result12.gdat','r')
cnn = np.fromfile(f, dtype=np.float32)
f.close()

# Open observation (GODAS, 1981-2017)
f = Dataset('C:/Users/zetal/PycharmProjects/PythonProject/data/GODAS/GODAS.label.12mn_3mv.1982_2017.nc','r')
obs = f.variables['pr'][:,11,0,0]
f.close()

# Normalize
cnn = cnn/np.std(cnn)
obs = obs/np.std(obs)

# Compute correlation coefficient (1984-2017)
cor_cnn = np.round(np.corrcoef(obs[3:],cnn[3:])[0,1],2)

# Draw Figure 3a
plt.figure(figsize=(10, 4))

x = np.arange(1,37)

plt.plot(x, obs, 'black', linewidth=2)
plt.plot(x, cnn, 'orangered', linewidth=0.5, marker='o', markersize=2)

plt.legend(('Observation','CNN (Cor='+str(cor_cnn)+')'),loc='upper right', prop={'size':7})

plt.xlim([0,37])
plt.ylim([-3,3.5])
plt.xticks(np.arange(2,39,2), np.arange(1982, 2019, 2), fontsize=6.5)
plt.yticks(np.arange(-3,3.51,1), fontsize=6.5)
plt.tick_params(labelsize=6., direction='in', length=2, width=0.3)

plt.grid(linewidth=0.2, alpha=0.7)
plt.axhline(0, color='black', linewidth=0.5)
plt.title('6-month lead prediction for DJF Nino3.4', fontsize=8)
plt.xlabel('Year', fontsize=7)
plt.ylabel('DJF Nino3.4', fontsize=7)

plt.tight_layout()
plt.savefig('C:/Users/zetal/PycharmProjects/PythonProject/Figure_3a_6', dpi=300)
plt.show()