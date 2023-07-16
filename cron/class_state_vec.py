#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#state_vector
import numpy as np
import pickle

class state_vector:
    def __init__(self, al=[], t=[]):
        self.tdim = np.size(t)
        self.al = al
        self.x0 = al
        self.t = t
        
    def __str__(self):
        print('Initial condition:')
        print(self.x0)
        print('Trajectory:')
        print(self.trajectory)
        return self.name
    
    def setName(self,name):
        self.name = name
    
    def getTrajectory(self):
        return self.trajectory
        
    def setTrajectory(self,states):
        self.trajectory = states
        
    def getEnsTrajectory(self):
        return self.Enstrajectory
    
    def setEnsTrajectory(self,enstates):
        self.Enstrajectory = enstates
        
    def setXaTrajectory(self,Xa_history):
        self.Xatrajectory = Xa_history
       
    def getXaTrajectory(self):
        return self.Xatrajectory
        
    def setParaName(self,paraname):
        self.paraname = paraname
        
    def getParaName(self):
        return self.paraname    
        
    def getTimes(self):
        return self.t
    def setCountry(self, country):
        self.country = country
        
    def getCountry(self):
        return self.country
    
    def setTimeseries(self, timeseries):
        self.timeseries = timeseries
        
    def getTimeseries(self):
        return self.timeseries
    
    def setStateDim(self,statedim):
        self.statedim = statedim
        
    def setParaDim(self,paradim):
        self.pdim = paradim
    def setCondition(self,condition):
        self.condi = condition
    def getCondition(self):
        return self.condi
    
    def getStateDim(self):
        return self.statedim
    
    def getParaDim(self):
        return self.pdim
    def setInitial(self, initial):
        self.initial = initial
    def getInitial(self):
        return self.initial
    
    def setPopulation(self, population):
        self.popu = population
        
    def getPopulation(self):
        return self.popu
    def setDate(self, date):
        self.date = date
    def getDate(self):
        return self.date
    def save(self, outfile):
        with open(outfile, 'wb') as output:
            pickle.dump(self, output)
        
    def load(self, infile):
        with open(infile, 'rb') as input:
            sv = pickle.load(input)
        return sv



