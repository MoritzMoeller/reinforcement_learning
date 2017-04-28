#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 17:27:45 2017

@author: mo

based on Code from:
http://rl-gym-doc.s3-website-us-west-2.amazonaws.com/mlss/pg-startercode.py
"""

import numpy as np, os
os.environ["THEANO_FLAGS"]="device=cpu,floatX=float64"
import theano, theano.tensor as T
import gym
import pickle
import os.path

PRINT_STATS = False

# saves learning curve to file in path data/..., which contains parameters in it's name

def save_stats_to_file(config, stats):
    
    data = np.array(stats).reshape(1,len(stats))
    data_file = '_'.join('{}={}'.format(key, val) for key, val in config.items())
    data_file = "data/" + data_file
    
    if os.path.isfile(data_file):
        fileObject = open(data_file, 'rb')
        old_data = pickle.load(fileObject)
        fileObject.close()
        data = np.vstack((old_data,data))    

    fileObject = open(data_file,'wb')     
    pickle.dump(data,fileObject)   
    fileObject.close()

def discount(x, gamma):
    
    # Given vector x, computes a vector y such that  y[i] = x[i] + gamma * x[i+1] + gamma^2 x[i+2] + ...
    
    out = np.zeros(len(x), 'float64')
    out[-1] = x[-1]
    
    for i in reversed(range(len(x)-1)):
        out[i] = x[i] + gamma*out[i+1]
    assert x.ndim >= 1

    return out

def categorical_sample(prob_n):
    

    # Sample from categorical distribution
    
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np.random.rand()).argmax()

def get_traj(agent, env, episode_max_length, render=False):
    

    # Run agent-environment loop for one whole episode, return dictionary of results
    
    ob = agent.one_hot(env.reset())
    obs = []
    acts = []
    rews = []
    
    for _ in range(episode_max_length):
        a = agent.act(ob)
        obs.append(ob)
        (ob, rew, done, _) = env.step(a)
        ob = agent.one_hot(ob)
        acts.append(a)
        rews.append(rew)
        if done: break
        
    return {"reward" : np.array(rews),
            "ob" : np.array(obs),
            "action" : np.array(acts)
            }

def sgd_updates(grads, params, stepsize):
    
    # Create list of parameter updates from stochastic gradient ascent
    
    updates = []
    for (param, grad) in zip(params, grads):
        updates.append((param, param + stepsize * grad))
    return updates 

class REINFORCEAgent(object):

    # Discounted REINFORCE algorithm

    def __init__(self, ob_space, action_space, gamma_, **usercfg):
        
        # Initialize your agent's parameters
        
        self.nO = ob_space.n
        nA = action_space.n
        
        #algorithm parameters
        
        self.config = dict(episode_max_length=1000, trajectories_total=100, n_iter=400, gamma= gamma_, stepsize=50)
        self.config.update(usercfg)

        # Symbolic variables for observation, action, and advantage
        # These variables stack the results from many timesteps--the first dimension is the timestep
        
        ob_no = T.fmatrix() # Observation
        a_n = T.ivector() # Discrete action 
        adv_n = T.fvector() # Advantage
        
        def shared(arr):
            return theano.shared(arr.astype('float64'))
        
        # create weights of neural network with one hidden layer

        self.W = shared(np.random.randn(self.nO,nA)*0.001)
        params = [self.W]

        # define action probabilities
        
        prob_na = T.nnet.softmax(ob_no.dot(self.W))
        N = ob_no.shape[0]
        
        # define loss function that will be differentiate to get the policy gradient
        # Note that we've divided by the total number of timesteps
        # Inlcudes entropy reularisation to prevent premature convergence
        
        beta = -.01
        entropy = T.mean(prob_na*T.log(prob_na))
        loss = T.log(prob_na[T.arange(N), a_n]).dot(adv_n) / N + entropy*beta

        stepsize = T.fscalar()
        grads = T.grad(loss, params)
        
        # Perform parameter updates

        updates = sgd_updates(grads, params, stepsize)
        self.pg_update = theano.function([ob_no, a_n, adv_n, stepsize], [], updates=updates, allow_input_downcast=True)
        self.compute_prob = theano.function([ob_no], prob_na, allow_input_downcast=True)

    def act(self, ob):
        
        # Choose an action
        
        prob = self.compute_prob(ob.reshape(1,-1))
        action = categorical_sample(prob)
        return action
    
    def one_hot(self, ob):
        
        # convert to one-hot
        
        one_hot = np.identity(self.nO)[ob:ob+1]
        return one_hot[0]

    def learn(self, env):

        cfg = self.config
        
        # set up list to collet the average returns, as to get the learning curve 
        
        stat_rets = []
        
        for iteration in range(cfg["n_iter"]):
            
            # Collect trajectories until we get timesteps_per_batch total timesteps 
            
            trajs = []
            
            for trajectories_total in range(cfg["trajectories_total"]):
                traj = get_traj(self, env, cfg["episode_max_length"])
                trajs.append(traj)

            all_ob = np.concatenate([traj["ob"] for traj in trajs])
            all_action = np.concatenate([traj["action"] for traj in trajs]) 
            
            # Compute discounted sums of rewards
            
            rets = [discount(traj["reward"], cfg["gamma"]) for traj in trajs]         
            all_ret = np.concatenate(rets)
            
            # Do policy gradient update step
            
            self.pg_update(all_ob, all_action, all_ret, cfg["stepsize"])
            
            # Calc total rewards and lengths of the episodes
            
            ep_rews = np.array([traj["reward"].sum() for traj in trajs])
            ep_lens = np.array([len(traj["reward"]) for traj in trajs]) 
            
            # Print stats
            
            if PRINT_STATS == True:
                print("-----------------")
                print("Iteration: \t %i"%iteration)
                print("NumTrajs: \t %i"%len(ep_rews))
                print("NumTimesteps: \t %i"%np.sum(ep_lens))
                print("MaxRew: \t %s"%ep_rews.max())
                print("MeanRew: \t %s +- %s"%(ep_rews.mean(), ep_rews.std()/np.sqrt(len(ep_rews))))
                print("MeanLen: \t %s +- %s"%(ep_lens.mean(), ep_lens.std()/np.sqrt(len(ep_lens))))
                print("-----------------")

            stat_rets.append(ep_rews.mean())
            
        return stat_rets

def main():
    
    env = gym.make("FrozenLake-v0")
    
    # iterate over gamma values to investigate dependence of learning on discount
    
    iterations_gamma = 1
    gamma_start = 1
    gamma_stop = .994
    
    # Record learning_runs many learning curves, save all of them to file
    
    learning_runs = 10
    
    for i in range(iterations_gamma):
        if iterations_gamma == 1:
            gamma = gamma_start
            
        else:
            gamma = gamma_start - (gamma_start - gamma_stop)*i/(iterations_gamma -1)
                    
        for j in range(learning_runs):
            agent = REINFORCEAgent(env.observation_space, env.action_space, gamma)
            print("Gamma: %f ,Run %i: Start learning" %(gamma,j))
            stat_rets = agent.learn(env)
            save_stats_to_file(agent.config, stat_rets)

if __name__ == "__main__":
    main()
    