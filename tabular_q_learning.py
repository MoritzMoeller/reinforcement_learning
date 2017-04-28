#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 10:44:00 2017

@author: mo
"""

import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

SHOW_MOVIE = True
SHOW_RETURN = False


# pick grid_size = 8 for 'FrozenLake8x8-v0', grid_size = 4 for 'FrozenLake-v0'

grid_size = 8

# functions that convert Q table to dislayed objects

def Q_to_V(Q):
    
    V = np.amax(Q,1)
    
    return V.reshape((grid_size,grid_size))


def Q_to_greedy_quiver(Q):
    
    a_greedy = np.argmax(Q,1)
    
    vert = np.zeros(np.shape(a_greedy))
    hori = np.zeros(np.shape(a_greedy))
    
    for i in range(len(a_greedy)):
        
        a = a_greedy[i]
        
        #down: 0 left: 1 up: 2 right:3
        
        if a == 0:
            vert[i] = -1
            hori[i] =  0
                
        if a == 1:
            vert[i] =  0
            hori[i] = -1
                
        if a == 2:
            vert[i] =  1
            hori[i] =  0
                
        if a == 3:
            vert[i] =  0
            hori[i] =  1

    
    return (hori.reshape(grid_size,grid_size), vert.reshape(grid_size,grid_size))

# set up visualisation

fig, ax = plt.subplots()
im_list = []
quiv_list = []
X, Y = np.meshgrid(np.arange(0, grid_size), np.arange(0, grid_size))

# initialise environment

env = gym.make('FrozenLake8x8-v0')

# parameters of the learning algorithm

gamma = .99
alpha = .85
eps = .05
number_of_episodes = 20000

Q = np.zeros((env.observation_space.n, env.action_space.n))


rets = []
ret = 0
s = env.reset()

i = 0
j = 0

FOUNDGOAL = False

# loop over episodes

while i < number_of_episodes:

    # eps-greedy strategy with decreasing eps
    
    a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
    (s_new, rew, done, _) = env.step(a)
    
    # first encounter of goal state gets reported:
    
    if ((rew != 0) & (FOUNDGOAL == False)) :
        FOUNDGOAL = True
        print("found goal! in episode %i " %j)          
    
    if done:
        Q[s,a] = Q[s,a] + alpha*(rew - Q[s,a])
        
        # paint a picture
        
        if FOUNDGOAL:
            i = i + 1
            quiv_list.append(Q_to_greedy_quiver(Q))
            im_list.append(Q_to_V(Q))
            
            if i%1000 == 0:
                print("Episode count: %i" %i)
        
        s = env.reset()
        rets.append(ret + rew)
        ret = 0
        j = j + 1
        
    else:
        Q[s,a] = Q[s,a] +  alpha*(rew + gamma*np.amax(Q[s_new,:]) - Q[s,a])
        s = s_new
        ret = ret + rew

(U,V) = quiv_list[0]
im = ax.imshow(im_list[0], cmap=plt.get_cmap('CMRmap'), vmin=0, vmax=1)
quiv = ax.quiver(X,Y,U,V)

# making it into an animation

def updatefig(j):
    
    # set the data in the axesimage object

    (U,V) = quiv_list[j]
    quiv.set_UVC(U,V)
    im.set_array(im_list[j])
    
    return im,quiv,

# kick off the animation 

ani = animation.FuncAnimation(fig, updatefig, frames=range(1000), interval=50, blit=False)

if SHOW_MOVIE:
    # Does not work in IPython console!
    plt.show()

# calc ret avg over k epidodes:

k = 50 
avg_rets = np.zeros(len(rets) - k)

for j in range(len(avg_rets)):
    avg_rets[j] = np.mean(rets[j:j+k])

if SHOW_RETURN:
    plt.plot(avg_rets)

