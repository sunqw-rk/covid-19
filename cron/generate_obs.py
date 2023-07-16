#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from class_state_vec import state_vector
from class_obs import obs_da
import random as rd
import pandas as pd

infile = 'x_nature.pkl'
sv = state_vector()
sv = sv.load(infile)
x_nature = sv.getTrajectory()
maxit,xdim = np.shape(x_nature)

country_name = sv.getCountry()
outfile = 'y_obs.pkl'
obs = obs_da(name = 'observe_three_state')

df = pd.read_csv ('%s/data_full_all_history.csv'%(country_name))
yo = np.zeros((len(df),xdim))


print('Using day-to-day log-transform')

p = df['daily_hos_add']
for i in range(len(yo)):
    if p[i]>=1:
        yo[i,3] = np.log(p[i])
    else:
        yo[i,3] = np.log(p[i]+1)

p=df['acc_recovered']
for i in range(len(yo)):
    if p[i]>=1:
        yo[i,4] = np.log(p[i])
    else:
        yo[i,4] = np.log(p[i]+1)

p=df['acc_death']
for i in range(len(yo)):
    if p[i]>=1:
        yo[i,5] = np.log(p[i])
    else:
        yo[i,5] = np.log(p[i]+1)
      
y_number = np.exp(yo)

print('Observation numbers = ')
print(y_number[:,3:6])


pos = np.zeros_like(yo)
for i in range(int(maxit/100 + 1e-5)):
    pos[i,:] = [0,1,2,3,4,5,6,7,8]

obs.setVal(yo)
obs.setPos(pos)

obs.save(outfile)



# In[ ]:




