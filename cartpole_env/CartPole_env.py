# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 16:29:34 2020

@author: enric
"""

#%%

from cartpole_env.InvertedPendulum import InvertedPendulum
import gym
from gym import spaces
import numpy as np

#%%

class CartPoleEnv(gym.Env):
    
    def __init__(self, sim_length_max = 100, difficulty = 0, continue_if_fail = False):
        # cartpole params
        
        self.continue_if_fail = continue_if_fail
        
        self.env_type = 'CartPole'
        
        self.done_reward = [-50, -20, 10]
        
        #self.weight = [0.1, 2, .25]  # tracking error, vertical angle, changes in control action
        self.weight = [0.05, 1, .1]  # tracking error, vertical angle, changes in control action
        
        self.sim_length_max = sim_length_max # max simulation length (in seconds)
        self.dt = 0.05
        self.cartpole = InvertedPendulum()
        
        self.difficulty = difficulty
        
        self.phi = np.array([0,0,0,0])
        
        if self.difficulty == 0: 
            self.max_distance = 2  # if actual distance is greater than this threshold, task has "failed"
            self.reference_speed_factor = .5
        elif self.difficulty == 1:
            self.max_distance = 1.5  # if actual distance is greater than this threshold, task has "failed"
            self.reference_speed_factor = .75
        elif self.difficulty == 2:
            self.max_distance = 1  # if actual distance is greater than this threshold, task has "failed"
            self.reference_speed_factor = 1
        elif self.difficulty == 3:
            self.max_distance = .9  # if actual distance is greater than this threshold, task has "failed"
            self.reference_speed_factor = 1.2
        
        # target params
        
        
        
        self.max_force = np.array([self.cartpole.force_max])
        self.max_state = np.append(self.cartpole.state_max_value, self.cartpole.state_max_value[0])

        self.action_space = spaces.Box(low=-self.max_force, high=self.max_force) #, shape=(size__,))
        self.observation_space = spaces.Box(low=-self.max_state, high=self.max_state) # , shape=(len(state_max),))

        # initialize state
        self.state = np.zeros(5) # in gym-like env state has to include target
        self.duration = None

    #####################################################################################################            
    def get_max_iterations(self):
        return int(round(self.sim_length_max / self.dt))

    #####################################################################################################            
    def get_actions_structure(self):
        pass # implemented in wrapper

        
    #####################################################################################################            
    def get_observations_structure(self):
        return len(self.state)

    #####################################################################################################            
    # this function returns a traditional control input to be used instead of the ML one
    def get_controller_input(self):
        self.cartpole.computeControlSignals(self.state[:-1],self.state[-1])
        return self.cartpole.get_ctrl_signal()


    #####################################################################################################            
    def get_complete_sequence(self):
        if hasattr(self, 'stored_states_sequence'):
            return self.stored_states_sequence, self.stored_ctrl_sequence
        else:
            return None


    #####################################################################################################            
    def step(self,action):
        
        self.generate_target()
        if self.duration == 0:
            reset_run = True
        else:
            reset_run = False
        """            
        if self.cartpole.state_archive is not None:
            if self.cartpole.state_archive[-1,2] != self.state[2]:
                stophere = 1
        """    
        state_4 = self.cartpole.step_dt(self.dt, self.state[:-1], action, hold_integrate = True, reset_run =reset_run)
        
       
        self.state = np.append(state_4, self.x_target)
        self.cartpole.store_results(state_4, self.x_target, action)

        ctrl_penalty = self.weight[2]*np.diff(self.cartpole.ctrl_inputs[-2:])**2
        regular_reward = np.maximum(0 , 1+self.duration -( self.weight[0]*(self.state[0]-self.state[4])**2 + self.weight[1]*self.state[2]**2 + ctrl_penalty[0]) ) 


        #evaluate rewards
        if not self.continue_if_fail:
            
            if  np.abs(self.state[2]) > np.pi/2:
                done = True
                reward = self.done_reward[0] # 50   # 
            
            elif np.abs(self.state[0]-self.x_target) > self.max_distance:
                done = True
                reward = self.done_reward[1]
                
            else:
                #print(f'current time = {self.duration}, max time = {self.sim_length_max}')
                #print(reward)
                reward = regular_reward
                # if this becomes negative there miht be an incentive to stop the simulation early!
                if self.duration >= self.sim_length_max:
                    reward += self.done_reward[2]
                    done = True
                else:
                    done = False
                    
        else:
            reward = regular_reward
            #print(reward)
            # if this becomes negative there miht be an incentive to stop the simulation early!
            if self.duration >= self.sim_length_max:
                reward += self.done_reward[2]
                done = True
            else:
                done = False
                    
        
        self.duration += self.dt
        
        if done:
            self.stored_states_sequence = np.append(self.cartpole.solution, self.cartpole.target_pos[:,np.newaxis], axis=1)
            self.stored_ctrl_sequence = self.cartpole.ctrl_inputs
        
        return self.state, reward, done, {}


    #####################################################################################################
    def generate_target(self):
        
        omega_1 = .3*self.reference_speed_factor
        omega_2 = .1*self.reference_speed_factor
        omega_3 = .2*self.reference_speed_factor
        omega_4 = .4*self.reference_speed_factor

        if self.duration == 0:
            phi_1 = np.pi/4*(np.random.random()-0.5)
            phi_2 = np.pi/4*(np.random.random()-0.5)
            phi_3 = np.pi/4*(np.random.random()-0.5)
            phi_4 = np.pi/4*(np.random.random()-0.5)
            self.phi = np.array([phi_1, phi_2, phi_3, phi_4])
        
        def fun_val(t):
            signal = 2*np.sin(omega_1*t+ self.phi[0]) + 0.5*np.sin(omega_2*t + np.pi/7+ self.phi[1]) + \
                0.8*np.sin(omega_3*t - np.pi/12+ self.phi[2])+ 1*np.cos(omega_4*t+np.pi/5+ self.phi[3])
            return signal
    
        alpha = 0.01
        self.x_target = fun_val(self.duration) - fun_val(0)*np.exp(-alpha*self.duration)


        
    #####################################################################################################
    def reset(self, save_history = False, full_random = False, max_val = None):
                
        if max_val is None:
            max_val = self.max_state
        
        self.duration = 0
        self.x_target = 0
        if not full_random:
            self.state = np.array([0,0,0.05*np.random.randn(),0,0])
        else:
            self.state = 2*max_val * np.random.random(max_val.shape[0]) - max_val

        self.reset_update_storage(save_history = save_history)

    #####################################################################################################
    def reset_update_storage(self, save_history = False):
        self.cartpole.reset_store(self.state[:-1], ml_ctrl = True, save_history = save_history)


    #####################################################################################################
    def plot_graphs(self):
        time_line = np.arange(0.0, self.duration, self.dt)
        self.cartpole.plot_graphs(time_line, save = True, no_norm = False, ml_ctrl = True)


    #####################################################################################################    
    # currently copy/paste from robot-env
    def render(self,mode = 'human',iter_params = (0,0,0), action = 0):

        #self.cartpole.store_results(self.state[:-1], self.state[-1], action) # already stored during simulation
        self.cartpole.store_render_data(iter_params)
        
    #####################################################################################################    
    # currently copy/paste from robot-env
    def get_gif(self, fig, save_history = False):
        self.reset_update_storage(save_history= save_history)
        return self.cartpole.generate_gif(self.dt, sample_step=1, fig = fig)




#%%

