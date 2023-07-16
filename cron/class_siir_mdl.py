# Class S-E-I1-I2-H-R model
import numpy as np
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
#import matplotlib.pyplot as plt
import pickle
import math
from math import e

def siir(state_all,t, N, epsilon, gamma1, gamma2, sigma, tauH, gammaH, gammaD, p, beta_before):
    ex, i1, i2, h, rh, d, r1, r2, betas = state_all
    E  = np.exp(ex) 
    I1 = np.exp(i1) 
    I2 = np.exp(i2) 
    H  = np.exp(h)  
    RH = np.exp(rh) 
    D  = np.exp(d)  
    R1 = np.exp(r1) 
    R2 = np.exp(r2) 
    BETAS = np.exp(betas)
    S  = N - E - I1 - I2 - H - R1 - R2 - RH - D
    dexdt = (np.exp(-ex)) * ((0.58*BETAS/N) * I1 * S + (BETAS/N) * I2 * S - epsilon * E)
    #dexdt = (np.exp(-ex)) * ((0.58*BETAS/100000000) * I1 * S + (BETAS/100000000) * I2 * S - epsilon * E)
    di1dt = (np.exp(-i1)) * (epsilon * E - gamma1 * I1 - sigma * I1)  
    di2dt = (np.exp(-i2)) * (sigma * I1 - gamma2 * I2 - tauH * I2)
    dhdt  = (np.exp(-h))  * (tauH * I2 - gammaH * H - gammaD * H)
    drhdt = (np.exp(-rh)) * (gammaH * H - p * RH)
    dddt  = (np.exp(-d))  * (gammaD * H)
    dr1dt = (np.exp(-r1)) * (gamma1 * I1 - p * R1)
    dr2dt = (np.exp(-r2)) * (gamma2 * I2 - p * R2)
    dbetasdt = beta_before
    return dexdt, di1dt, di2dt, dhdt, drhdt, dddt, dr1dt, dr2dt, dbetasdt

def siir_t1(state_all,t, N, gamma1, gamma2, sigma, tauH, gammaH, gammaD, p, beta_before): #beta1 =  beta2
#def siir_t1(state_all,t, N, beta2, gamma1, gamma2, sigma, tauH, gammaH, gammaD, p, beta_before):
    i1, i2, h, rh, d, r1, r2, beta1 = state_all
    I1 = np.exp(i1) - 1
    I2 = np.exp(i2) - 1
    H  = np.exp(h)  - 1
    R1 = np.exp(r1) - 1
    R2 = np.exp(r2) - 1
    RH = np.exp(rh) - 1
    D  = np.exp(d)  - 1 
    S  = N - I1 - I2 - H - R1 - R2 - RH - D
    di1dt = (np.exp(-i1)) * ((beta1/100000000) * I1 * S + (beta1/100000000) * I2 * S + (beta1/100000000) * H * 0.0141 * S - gamma1 * I1 - sigma * I1) #beta1 =  beta2
    #di1dt = (np.exp(-i1)) * ((beta1/100000000) * I1 * S + 1*(beta2/100000000) * I2 * S + 1*(beta2/100000000) * H * 0.00001 * S - gamma1 * I1 - sigma * I1)
    di2dt = (np.exp(-i2)) * (sigma * I1 - gamma2 * I2 - tauH * I2)
    dhdt  = (np.exp(-h))  * (tauH * I2 - gammaH * H - gammaD * H)
    dr1dt = (np.exp(-r1)) * (gamma1 * I1 - p * R1)
    dr2dt = (np.exp(-r2)) * (gamma2 * I2 - p * R2)
    drhdt = (np.exp(-rh)) * (gammaH * H - p * RH)
    dddt  = (np.exp(-d))  * (gammaD * H)
    #dbeta1dt = 0
    dbeta1dt = beta_before
    return di1dt, di2dt, dhdt, drhdt, dddt, dr1dt, dr2dt, dbeta1dt


class SIIR:
    def __init__(self, N, epsilon, gamma1, gamma2, sigma, tauH, gammaH, gammaD, p, beta_before): # beta1 =  beta2
        self.N = N
        self.epsilon = epsilon
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.sigma = sigma
        self.tauH = tauH
        self.gammaH = gammaH
        self.gammaD = gammaD
        self.p = p
        self.beta_before = beta_before
        self.params = [self.N, self.epsilon, self.gamma1, self.gamma2, self.sigma, self.tauH, self.gammaH, self.gammaD, self.p, self.beta_before]
        #self.params = [self.N, self.beta2, self.gamma1, self.gamma2, self.sigma, self.tauH, self.gammaH, self.gammaD, self.p, self.beta_before]

    
    def run(self, state_all0, t,t_output, whichpara):
        if whichpara == 'betas':
            state_all = odeint(siir, state_all0, t, args = (self.N, self.epsilon, self.gamma1, self.gamma2, self.sigma, self.tauH, self.gammaH, self.gammaD, self.p, self.beta_before))
            #state_all = solve_ivp(siir, (t[0],t[-1]), state_all0, method = 'RK45', t_eval = t_output, dense_output= True, args = (self.N, self.epsilon, self.gamma1, self.gamma2, self.sigma, self.tauH, self.gammaH, self.gammaD, self.p, self.beta_before))
            #state_all = solve_ivp(siir, (t[0],t[-1]), state_all0, t_eval= t, args = (self.N, self.beta2 , self.gamma1, self.gamma2, self.sigma, self.tauH, self.gammaH, self.gammaD, self.p))
        elif whichpara == 'beta2':
            state_all = odeint(siir_b2, state_all0, t)
        elif whichpara == 'gamma1':
            state_all = odeint(siir_g1, state_all0, t)
        elif whichpara == 'gamma2':
            state_all = odeint(siir_g2, state_all0, t)
        elif whichpara == 'sigma':
            state_all = odeint(siir_sigma, state_all0, t)
        elif whichpara == 'p':
            state_all = odeint(siir_p, state_all0, t)
        else:
            print('unrecognized parameter')
            raise SystemExit    
        return state_all
    
    def run_t1(self, state_all0, t, whichpara):
        if whichpara == 'betas':
            state_all = odeint(siir_t1, state_all0, t, args = (self.N, self.epsilon, self.gamma1, self.gamma2, self.sigma, self.tauH, self.gammaH, self.gammaD, self.p, self.beta_before))
        elif whichpara == 'beta2':
            state_all = odeint(siir_b2, state_all0, t)
        elif whichpara == 'gamma1':
            state_all = odeint(siir_g1, state_all0, t)
        elif whichpara == 'gamma2':
            state_all = odeint(siir_g2, state_all0, t)
        elif whichpara == 'sigma':
            state_all = odeint(siir_sigma, state_all0, t)
        elif whichpara == 'p':
            state_all = odeint(siir_p, state_all0, t)
        else:
            print('unrecognized parameter')
            raise SystemExit    
        return state_all
   
            
        
    def plot_state1(self, t, state1, t1, state1_t1, outfile = 'S-I1-I2-R model(log10)', plot_title = 'S-I1-I2-R model(log10)'):
        plt.plot(t, state1[:,0], 'y', linewidth = 3, label = 'log(I1+1)')
        plt.plot(t, state1[:,1], 'r', linewidth = 3, label = 'log(I2+1)')
        plt.plot(t, state1[:,2], 'b', linewidth = 3, label = 'log(H+1)')
        plt.plot(t, state1[:,3], 'g', linewidth = 3, label = 'log(RH+1)')
        plt.plot(t, state1[:,4], 'k', linewidth = 3, label = 'log(D+1)')
        plt.plot(t1, state1_t1[:,0], 'y', linewidth = 3)
        plt.plot(t1, state1_t1[:,1], 'r', linewidth = 3)
        plt.plot(t1, state1_t1[:,2], 'b', linewidth = 3)
        plt.plot(t1, state1_t1[:,3], 'g', linewidth = 3)
        plt.plot(t1, state1_t1[:,4], 'k', linewidth = 3)
        plt.xlabel('time(day)', fontsize=20)
        plt.ylabel('value', fontsize=20)
        plt.xticks(size = 20)
        plt.yticks(size = 20)
        plt.title(plot_title, fontsize=25)
        plt.legend(loc='best', prop={'size': 15}) 
        plt.grid(color = 'grey', linestyle = ':', linewidth = 1)
        plt.rcParams['figure.figsize'] = [10/1, 5/1]
        plt.savefig(outfile)
        plt.show()
        
    def plot_state2(self, t, state2, t1, state2_t1, para_name, outfile = 'para_name', plot_title = 'para_name'):
        plt.plot(t, state2[:,3], 'r', linewidth = 3, label = '%s'%(para_name))
        plt.plot(t1, state2_t1[:,3], 'r', linewidth = 3)
        plt.xlabel('time(day)', fontsize=25)
        plt.ylabel('Value', fontsize=25)
        plt.xticks(size = 25)
        plt.yticks(size = 25)
        plt.title(plot_title, fontsize=25)
        plt.legend(loc='best', prop={'size': 20}) 
        plt.grid(color = 'grey', linestyle = ':', linewidth = 1)
        plt.rcParams['figure.figsize'] = [10/1, 5/1]
        plt.savefig(outfile)
        plt.show()
        
    def plot_state_ori(self, t, state1, t1, state1_t1, outfile = 'S-I1-I2-R model', plot_title = 'S-I1-I2-R model'):
        plt.plot(t, state1[:,0], 'y', linewidth = 3, label = 'asymptomatic infectious(I1)')
        plt.plot(t, state1[:,1], 'r', linewidth = 3, label = 'symptomatic infectious(I2)')
        plt.plot(t, state1[:,2], 'b', linewidth = 3, label = 'hospitalized(H)')
        plt.plot(t, state1[:,3], 'g', linewidth = 3, label = 'hospital recovered(RH)')
        plt.plot(t, state1[:,4], 'k', linewidth = 3, label = 'decease(D)')
        plt.plot(t1, state1_t1[:,0], 'y', linewidth = 3)
        plt.plot(t1, state1_t1[:,1], 'r', linewidth = 3)
        plt.plot(t1, state1_t1[:,2], 'b', linewidth = 3)
        plt.plot(t1, state1_t1[:,3], 'g', linewidth = 3)
        plt.plot(t1, state1_t1[:,4], 'k', linewidth = 3)
        plt.xlabel('time(day)', fontsize=20)
        plt.ylabel('Number', fontsize=20)
        plt.xticks(size = 20)
        plt.yticks(size = 20)
        plt.title(plot_title, fontsize=25)
        plt.legend(loc='best', prop={'size': 15}) 
        plt.grid(color = 'grey', linestyle = ':', linewidth = 1)
        plt.rcParams['figure.figsize'] = [10/1, 5/1]
        plt.savefig(outfile)
        plt.show()
      
    def plot_state4(self, t, state2, outfile = 'Sigma', plot_title = 'Sigma'):
        #plt.plot(t, state2[:,4], 'r', linewidth = 3, label = 'beta1')
        #plt.plot(t, state2[:,5], 'c', linewidth = 3, label = 'beta2')
        #plt.plot(t, state2[:,6], 'm', linewidth = 3, label = 'gamma1')
        #plt.plot(t, state2[:,7], 'y', linewidth = 3, label = 'gamma2')
        plt.plot(t, state2[:,8], 'k', linewidth = 3, label = 'sigma')
        #plt.plot(t, state2[:,9], 'g', linewidth = 3, label = 'p')
        plt.xlabel('time(day)', fontsize=25)
        plt.ylabel('Value', fontsize=25)
        plt.xticks(size = 25)
        plt.yticks(size = 25)
        plt.title(plot_title, fontsize=25)
        plt.legend(loc='best', prop={'size': 20}) 
        plt.grid(color = 'grey', linestyle = ':', linewidth = 1)
        plt.rcParams['figure.figsize'] = [10/1, 5/1]
        plt.savefig(outfile)
        plt.show()


