#!/usr/bin/env python
# coding: utf-8

# In[1]:


from class_siir_mdl import SIIR
from class_state_vec import state_vector
from class_obs import obs_da
#import matplotlib.pyplot as plt
from class_da_sys import da_system
import numpy as np
import pandas as pd
import math
import sys

sv = state_vector()
sv =  sv.load('x_nature.pkl')
#!!!!!!!!!!!!!!!!!!!!!!!!!!!delete later!!!!!!!!!!!!!!!!!!!!!!
sys.argv = [0, 0.3] #[beta_S initial]




name = 'x_nature'
df = pd.read_csv('%s/data_full_all.csv'%(sv.getCountry()))
date_series = [0, len(df)-1] #the first day in the observion list index as 0, input the index of the last day


N = sv.getPopulation()   #Total number of the population(Wiki)
epsilon = 1 # 1/3
gamma1 = 0.17 * (1/9)
gamma2 = 0.22 * (1/7)
sigma =  0.83 * (1/2)
tauH = np.zeros(len(date_series)-1)
tauH[:] = 0.78 #0.78 * (1/5.2)
gammaH = 0.97 * (1/14)
gammaD = 0.03 * (1/10)
p = 0
beta_before = 0
betas = np.zeros(len(date_series)-1)
betas[:] = sys.argv[1]
dt=0.01
t_overall = np.arange(date_series[0], date_series[-1], dt)


estimate_para = 'betas' 
state_all_0 = [0, 0,0, 0, 
               0, 0, 0, 0, np.log(betas[0])]

#sv = state_vector(al = state_all_0, t = t_overall)  #.....................
sv.setTimeseries(date_series)
#sv.setPopulation(N)
#sv.setCountry(sv.getCountry())
country_name = sv.getCountry()

for i in range(len(date_series)-1):
    tvec = np.arange(date_series[i], date_series[i+1]+1, dt)
    tvec_output = tvec
    smd = SIIR(N = N, epsilon = epsilon, gamma1 = gamma1, gamma2 = gamma2, sigma = sigma, 
               tauH = tauH[i], gammaH = gammaH, gammaD = gammaD, p = 0, beta_before = beta_before)
    history = smd.run(state_all_0, tvec, tvec_output, estimate_para) 
    if i == 0:
        trajectory = history
    else:
        trajectory = np.vstack((trajectory, history))
print('totay days=',int(np.shape(trajectory)[0]/100))
print('totay variables =',np.shape(trajectory)[1])
sv.setTrajectory(trajectory)


for i in range(len(trajectory)):
    if np.exp(trajectory[i,3]) > df['daily_hos_add'][0]:
        print(i)
        print(np.exp(trajectory[i,:]))
        break
sv.setInitial(np.exp(trajectory[i,:]))

outfile = name+'.pkl'
sv.save(outfile)


# In[ ]:




