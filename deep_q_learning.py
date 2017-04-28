#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 11:36:33 2017

@author: mo

On the Mountaincar environment: 
    
- The observation space is 2d: I = [-1.2,0.6]x[-0.07,0.07]
- considered solved if avg ret is less than -110

"""

import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation

env = gym.make('MountainCar-v0')

SHOW_RET = False
SHOW_MOVIE = True

#parameters of Q learning algorithm

gamma = .99
alpha = .002
eps = .01
number_of_episodes = 1000
max_steps = 500
ex_buffer_size = 10000
n_hidden = 128
batch_size = 32
train_tresh = 200

# set up lists to capture data

ret_list = []
im_list = []
traj_list = []
cont_list = []

# set up infra-structure for movie

fig, ax = plt.subplots()

gridpts = 100
x = np.linspace(-1.2,.6,gridpts)
x_prime = np.linspace(-.07,.07,gridpts)
grid = np.array(np.meshgrid(x,x_prime))
grid_lin = grid.reshape(2,gridpts*gridpts).transpose()
grid_act = np.ones((gridpts*gridpts,))


# define the experience buffer

class experience_buffer():
    
    def __init__(self, buffersize):
        self.buffersize = buffersize
        self.buffer = []
        
        
    def append_to_buffer(self, a):
        self.buffer.append(a)
        
        if len(self.buffer) > self.buffersize:
            self.buffer = self.buffer[1:] 
        
    def sample_batch(self, size):
        batch = random.sample(b.buffer,size)
        state_batch, action_batch, rew_batch, new_state_batch, done_batch = [],[],[],[],[]
        
        for (s,a,r,ns,d) in batch:
            state_batch.append(s)
            action_batch.append(a)
            new_state_batch.append(ns)
            rew_batch.append(rew)
            
            if d:
                done_batch.append(0)
                
            else:
                done_batch.append(1)
        
        state_batch_a = np.array(state_batch) 
        action_batch_a = np.array(action_batch)
        rew_batch_a = np.array(rew_batch)
        new_state_batch_a = np.array(new_state_batch)
        done_batch_a = np.array(done_batch)
        
        return(state_batch_a,action_batch_a,rew_batch_a,new_state_batch_a, done_batch_a)
        
        
# Construct the Q network

tf.reset_default_graph()

# define input variables, which will contain batches of input

states = tf.placeholder(dtype=tf.float32, shape=[None, env.observation_space.shape[0]])

# define model parameters       

W1 = tf.Variable(tf.random_normal(shape=[env.observation_space.shape[0], n_hidden], stddev = 0.001))
W2 = tf.Variable(tf.random_normal(shape=[n_hidden, env.action_space.n], stddev = 0.001))
b1 = tf.Variable(tf.zeros(shape=[1,n_hidden]))

# define nn model with one hidden layer and tanh activation function

Qout = tf.matmul(tf.nn.tanh(tf.matmul(states, W1) + b1),W2)

# define loss

targetQs = tf.placeholder(shape=[None],dtype=tf.float32)
actions = tf.placeholder(shape=[None],dtype=tf.int32)
actions_onehot = tf.one_hot(actions,env.action_space.n,dtype=tf.float32)
        
Q = tf.reduce_sum(Qout*actions_onehot, reduction_indices=1)
        
error = tf.square(targetQs - Q)
loss = tf.reduce_mean(error)

# define training function

trainer = tf.train.AdamOptimizer(learning_rate=alpha)
updateModel = trainer.minimize(loss)

init = tf.global_variables_initializer()

def update_network():
    
    # sample batch from experience buffer
    
    (s_b, a_b, rew_b, s_new_b, d_b) = b.sample_batch(batch_size)
    
    # calculate the targets using Q-learning update rule
    
    estimate_next_Q = sess.run(Qout, feed_dict={states: s_new_b})
    max_next_Q = np.amax(estimate_next_Q,1)
    targetQ_b = rew_b + gamma*max_next_Q*d_b
    
    # update the parameters of the model
    
    sess.run(updateModel, feed_dict={states: s_b, actions: a_b, targetQs: targetQ_b})
    
def eps_greedy_policy(s):
    
    Qout_ = sess.run(Qout, feed_dict={states: s})

    if np.random.rand(1) < eps:
        a = np.random.randint(0,env.action_space.n)
        
    else:
        a = np.argmax(Qout_)
    
    return a

with tf.Session() as sess:
    
    # init the tf session and the memory buffer
    
    sess.run(init)    
    b = experience_buffer(ex_buffer_size)
    
    # enter into core algorithm
   
    for i in range(number_of_episodes):
        
        if i%10 == 0:
            print("Now doing episode %i" %i)
            
        s = np.array([env.reset()])
        traj = []
        ret = 0
        
        for t in range(max_steps):
            
            # look for the next action
            
            a = eps_greedy_policy(s)
                
            #do transition  
            
            (s_new, rew, done, _) = env.step(a)
            
            #store transition in buffer
            
            b.append_to_buffer((s[0],a,rew,s_new, done))
            ret = ret + rew
            traj.append(s[0])
            
            # As soon as enough transitions are gathered, start to train network
            
            if len(b.buffer) > train_tresh:
                update_network()
            
            # every 40 steps, take a shot for the movie
            
            if t%40 == 0:
                
                # Evaluate current estimate of optimal q function
                
                Q_0 = sess.run(Q, feed_dict={states: grid_lin, actions: 0*grid_act})
                Q_1 = sess.run(Q, feed_dict={states: grid_lin, actions: 1*grid_act})
                Q_2 = sess.run(Q, feed_dict={states: grid_lin, actions: 2*grid_act})
                
                C = np.zeros(np.shape(Q_0))
                
                # construct plateau function encoding greedy choice of action
                
                C[(Q_0 > Q_1) * (Q_0 > Q_2)] = 0
                C[(Q_1 > Q_0) * (Q_1 > Q_2)] = 1
                C[(Q_2 > Q_0) * (Q_2 > Q_1)] = 2
                  
                # from estimate of q, calculate estimate of v
                                
                V = np.maximum(Q_0,Q_1,Q_2).reshape(100,100)
                
                
                im_list.append(V)
                cont_list.append(C.reshape(100,100))
                traj_list.append(traj)
            
            # advance to next step
            
            s = np.array([s_new])
            
            if done or t == max_steps - 1:
                print("Return in last episode: %f" %ret)
                ret_list.append(ret)
                break
            

# hollywood section: making a movie of different quatities   

# plot value function as colour map

max_V = np.max(np.array(im_list))
min_V = np.min(np.array(im_list))    
ax.imshow(im_list[0], cmap=plt.get_cmap('jet'), vmin=min_V, vmax=max_V, extent=(-1.2,.6,-.07,.07), aspect='auto')

# plot trajectory as curve

a = np.array(traj_list[0])
x = a[:,0]
y = a[:,1]
ax.plot(x,y)

# plot contours of plateau function encoding greedy choice of action

ax.contour(grid[0], grid[1], cont_list[0])

def updatefig(j):

    ax.clear()

    # plot valuefunction as colour map  
    
    ax.imshow(im_list[j], cmap=plt.get_cmap('jet'), vmin=min_V, vmax=max_V, extent=(-1.2,.6,-.07,.07), aspect='auto')

    # plot trajectory as curve
    
    a = np.array(traj_list[j])
    x = a[:,0]
    y = a[:,1]
    ax.plot(x,y)
    
    # plot contours of plateau function encoding greedy choice of action
    
    try:
        ax.contour(grid[0], grid[1], cont_list[j])
        
    except ValueError:
        pass

    return ax

ani = animation.FuncAnimation(fig, updatefig, frames=range(len(im_list)), interval=2, blit=False)

if SHOW_MOVIE:
    plt.show()

if SHOW_RET:
    plt.plt(ret_list)
    