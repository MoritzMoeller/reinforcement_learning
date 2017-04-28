#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 17:47:41 2017

@author: mo
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np

PLT_SLOPES = False
PLT_GAMMATRESH = True
PLT_GAMMA_1 = False

means = []
stds = []

# Specify the gammas that should be included in analysis. File must exist!

gammas = [ 1.0 , .9997, .9994, .9991, .9988, 0.9984999999999999, 0.9982,0.9979, 0.9976, 0.9973, 0.997, 0.9967, 0.9964, 0.9961, 0.9958, 0.9955, 0.9952, 0.9949, 0.9946, 0.9943, 0.994]

# Can only look at the curve for gamma = 1

if PLT_GAMMA_1:
    gammas = [ 1.0 ]
    
# load data

for gamma_ in gammas:
    config = dict(episode_max_length=1000, trajectories_total=100, n_iter=400, gamma=gamma_, stepsize=50)
    data_file = '_'.join('{}={}'.format(key, val) for key, val in config.items())
    data_file = "data/" + data_file
    fileObject = open(data_file, 'rb')
    data = pickle.load(fileObject)
    fileObject.close()
    means.append(data.mean(0))
    stds.append(data.std(0))
    
# analysis

k = 10
tresh = .3
x_treshs = []

for (mean,std,gamma) in zip(means,stds,gammas):
    
    # cut out a approx linear piece
    
    slope = mean[100-k:100+k]    
    x = np.arange(len(slope))
    
    # do linear fit 
    
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, slope)[0]
    
    # can plot slopes and fits for all gammas
    
    if PLT_SLOPES:
        plt.plot(x,slope, label= "gamma = " + str(gamma))
        plt.plot(x, m*x + c)
        plt.legend()
        plt.show()
        
    # determine at which episode learning cruve exceeds treshhold
    
    x_tresh = 1/m*(tresh - c)
    x_treshs.append(x_tresh + 100)
    
# can plot the number of episodes necessary to reach treshhold, with a linear fit

if PLT_GAMMATRESH:
    x = np.array(gammas[1:])
    y = np.array(x_treshs[1:])
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y)[0]
    plt.plot(gammas,x_treshs, "o")
    plt.plot(x,m*x+c)
    plt.show()
    
# can plot learning curve with std for gamma = 0
    
if PLT_GAMMA_1:
    y = means[0]
    std = stds[0]
    x = np.arange(len(y))
    plt.plot(x,y,"r")
    plt.fill_between(x, y - std, y + std)
    plt.show()