#!/usr/bin/env python
# coding: utf-8

# In[1]:


#2021_09_14  I change the setting for H R .. matrix
import numpy as np
from class_siir_mdl import SIIR
from class_state_vec import state_vector
from class_obs import obs_da
from class_da_sys import da_system
from sys import argv
import random as rd

infile = 'x_nature.pkl'
sv = state_vector()
sv = sv.load(infile)
x_nature = sv.getTrajectory()
maxit,xdim = np.shape(x_nature)
sv.setDate(sv.getDate())

sv.setStateDim(xdim-1)
sv.setParaDim(1)
statedim = sv.getStateDim()
paradim = sv.getParaDim()
sv.setParaName('betas')
estimate_para = sv.getParaName()
date_series = sv.getTimeseries()

infile = 'y_obs.pkl'
obs = obs_da()
obs = obs.load(infile)


yp = [3,4,5] # H, RH, D only

if len(yp) < xdim:
  obs.reduceDim(yp)

y_obs = obs.getVal()
y_pts = obs.getPos()

print('y_obs = ')
print(y_obs[0,:])
print('y_pts = ')
print(y_pts[0,:])

yrow,ydim = np.shape(y_obs)

#-----------------------------------------------------------------------
# Initialize the da system
#-----------------------------------------------------------------------
das = da_system()
das.setStateVector(sv)
das.setObsData(obs)
das.xdim = xdim
das.ydim = ydim
das.x0 = x_nature[0,:]
das.t = sv.getTimes()


#-----------------------------------------------------------------------
# Initialize the ensemble
#-----------------------------------------------------------------------
das.edim = 15#150 #np.int(1*xdim)
das.ens_bias_init = 0
das.ens_sigma_init = 0.3

#-----------------------------------------------------------------------
# Initialize the error covariances B and R, and the linearized 
# observation operator H
#-----------------------------------------------------------------------

I = np.identity(xdim)

# Set background error covariance
sigma_b = 1.0
B = I * sigma_b**2

# Set observation error covariance

R = np.zeros((len(y_obs),das.xdim,das.xdim))
sigma_r = 1

for i in range(len(y_obs)):
    R[i,:,:] = I * sigma_r**2

time_varied_R = False


if not time_varied_R:
    for w in yp:
        #R[:,w,w] = (np.log(1.3))**2
        #Qiwen change for 2022-01-01
        R[:,w,w] = (np.log(1.3))**2

        
# Set the linear observation operator matrix as I

H = np.zeros((len(y_obs),das.xdim,das.xdim))
for r in range(len(y_obs)):
    H[r,:,:] = I

# Set constant matrix for nudging
const = 1.0
C = I * const

das.setB(B)
das.setR(R)
das.setH(H)
das.setC(C)

# Update the matrices to fit the reduced observation dimension
if len(yp) < xdim:
  das.reduceYdim(yp)

print('B = ')
print(das.getB())
print('R = ')
print(das.getR())
print('H = ')
print(das.getH())


#-----------------------------------------------------------------------
# Initialize the timesteps
#-----------------------------------------------------------------------
#t_nature = sv.getTimes()
acyc_step = 100                        

fcst_step = acyc_step                


# Store basic timing info in das object
das.acyc_step = acyc_step
das.dt = 0.01
das.maxit = maxit
das.statedim = statedim
das.paradim = paradim

method='EnKF'
das.setMethod(method)

#-----------------------------------------------------------------------
# Store DA object
#-----------------------------------------------------------------------
name = 'x_analysis_init'
outfile=name+'.pkl'
das.save(outfile)


# In[ ]:




