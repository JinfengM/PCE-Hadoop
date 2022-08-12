# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 10:37:25 2022

@author: Dell
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 09:49:38 2021

@author: Administrator
"""


import pandas as pd
import numpy as np
import chaospy
from matplotlib.pyplot import MultipleLocator
x=['2010-1','2010-2','2010-3','2010-4','2010-5','2010-6','2010-7','2010-8','2010-9','2010-10','2010-11','2010-12','2011-1','2011-2','2011-3','2011-4','2011-5','2011-6','2011-7','2011-8','2011-9','2011-10','2011-11','2011-12','2012-1','2012-2','2012-3','2012-4','2012-5','2012-6','2012-7','2012-8','2012-9','2012-10','2012-11','2012-12','2013-1','2013-2','2013-3','2013-4','2013-5','2013-6','2013-7','2013-8','2013-9','2013-10','2013-11','2013-12','2014-1','2014-2','2014-3','2014-4','2014-5','2014-6','2014-7','2014-8','2014-9','2014-10','2014-11','2014-12']

    
import matplotlib
from matplotlib.font_manager import _rebuild
_rebuild()
#################################月均值时间序列#######################################
import matplotlib.pyplot as plt
input_csv=pd.read_csv('PCESA.csv')
ind=np.arange(8)
width=0.4
men_means=(20,23,33,44,4)
women_means=(33,23,33,33,3)

sobol=np.array(input_csv['sobol'])
pce_sobol=np.array(input_csv['pce-sobol'])
morris=np.array(input_csv['morris'])
pce_morris=np.array(input_csv['pce-morris'])
fast=np.array(input_csv['fast'])
pce_fast=np.array(input_csv['pce-fast'])

delta=np.array(input_csv['delta'])
pce_delta=np.array(input_csv['pce-delta'])
dgsm=np.array(input_csv['dgsm'])
pce_dgsm=np.array(input_csv['pce-dgsm'])
rbd=np.array(input_csv['rbd'])
pce_rbd=np.array(input_csv['pce-rbd'])


#with plt.style.context(['science', 'no-latex']):
#matplotlib.rcParams['font.family'] ='sans-serif'
#matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
#matplotlib.rcParams['axes.unicode_minus']=False

fig, ax = plt.subplots(2,3,constrained_layout=True, figsize=(10, 6))

rects1 = ax[0][0].bar(ind, sobol, width, color='#0C5DA5', label='Sobol')
rects2 = ax[0][0].bar(ind+width, pce_sobol, width,color='#00B945', label='PCE')
#plt.xticks(ind,('CANMX','CN2','CH_N2','CH_K2','ALPHA_BNK','SOL_AWC','SOL_K','SOL_BD'))
ax[0][0].set_xticklabels(['','x1','x3','x5','x7','x5','x6','x7'])
ax[0][0].set_ylabel('Normalized sensitivity index')
ax[0][0].legend()

rects3 = ax[0][1].bar(ind, morris, width, color='#0C5DA5', label='Morris')
rects4 = ax[0][1].bar(ind +width, pce_morris, width,color='#00B945', label='PCE')
ax[0][1].set_xticklabels(['','x1','x3','x5','x7','x5','x6','x7'])
ax[0][1].legend()

rects5 = ax[0][2].bar(ind, fast, width, color='#0C5DA5', label='FAST')
rects6 = ax[0][2].bar(ind +width, pce_fast, width,color='#00B945', label='PCE')
ax[0][2].set_xticklabels(['','x1','x3','x5','x7','x5','x6','x7'])
ax[0][2].legend()
# Add some text for labels, title and custom x-axis tick labels, etc.

#第二幅
rects7 = ax[1][0].bar(ind, delta, width, color='#0C5DA5', label='Delta')
rects8 = ax[1][0].bar(ind+width, pce_delta, width,color='#00B945', label='PCE')
#plt.xticks(ind,('CANMX','CN2','CH_N2','CH_K2','ALPHA_BNK','SOL_AWC','SOL_K','SOL_BD'))
ax[1][0].set_xticklabels(['','x1','x3','x5','x7','x5','x6','x7'])
ax[1][0].set_ylabel('Normalized sensitivity index')
ax[1][0].legend()

rects9 = ax[1][1].bar(ind, dgsm, width, color='#0C5DA5', label='DGSM')
rects10 = ax[1][1].bar(ind +width, pce_dgsm, width,color='#00B945', label='PCE')
ax[1][1].set_xticklabels(['','x1','x3','x5','x7','x5','x6','x7'])
ax[1][1].legend()

rects11 = ax[1][2].bar(ind, rbd, width, color='#0C5DA5', label='RBD-FAST')
rects12 = ax[1][2].bar(ind +width, pce_rbd, width,color='#00B945', label='PCE')
ax[1][2].set_xticklabels(['','x1','x3','x5','x7','x5','x6','x7'])
ax[1][2].legend()
#plt.xticks(ind,('x1','','x3','','x5','','x7',''))
#ax[0].autoscale(tight=True)
#plt.tight_layout()
fig.savefig('figures/3-sa-bar.png', dpi=300)
    
#
