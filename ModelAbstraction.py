#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 10:27:43 2020

@author: Enrico Regolin
"""

import os, sys

csfp = os.path.abspath(os.path.dirname(__file__))
if csfp not in sys.path:
    sys.path.insert(0, csfp)

import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import numpy as np
import random
import os
import pickle

from NeuralNetworks.Linear_NNs import LinearModel

from bashplotlib.scatterplot import plot_scatter
from scipy.signal import savgol_filter

from cartpole_env.CartPole_env import CartPoleEnv


#%%
#####
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("-e", "--epochs", dest="n_epochs", type = int, default=40, help="number of epochs per simulation")
parser.add_argument("-l", "--load", dest="load_net_params", type=bool, default=False,
                    help="load net parameters")

parser.add_argument("-bs", "--batch-size", dest="batch_size", type=int, default=32,
                    help="training batch size")

parser.add_argument("-lr", "--learning-rate", dest="lr", type=float, default=0.001,
                    help="optimizer initial learning rate")

parser.add_argument(
  "-sw", "--state-weights",  nargs=4,  # 0 or more values expected => creates a list
  dest = "state_weights", type=float, default=[1.,1.,1.,1.],  # default if nothing is provided
)

parser.add_argument(
  "-ll", "--layers-list",  nargs="*",  # 0 or more values expected => creates a list
  dest = "layers_list", type=int, default=[10,10],  # default if nothing is provided
)


parser.add_argument("-n", "--net-version", dest="net_version", type=int, default=0,
                    help="net version to load")

parser.add_argument("-m", "--memory-size", dest="memory", type=int, default=10000,
                    help="memory size for training set")

args = parser.parse_args()


torch.autograd.set_detect_anomaly(False)
        
#%%
class AbstractModelTrainer():
    
    ##########################################################################
    def __init__(self, batch_size = 128, lr = 0.001, n_epochs = 100, max_samples_stored=20000, layers_width= (5,5), state_weights = (1,1,1,1) ):
        
        # main training parameters
        self.lr = lr
        self.n_epochs = n_epochs
        self.max_samples_stored = max_samples_stored
        self.layers_width = layers_width
        self.batch_size = batch_size
 
        # state weighing (created directly as torch tensor)
        self.torch_weight = torch.from_numpy(np.diag(state_weights)).float().cuda()
 
        # CartPole initialization                
        self.plant = CartPoleEnv(sim_length_max = 10, difficulty = 1, continue_if_fail = True)
        self.dt = self.plant.dt

        # net name definition (for save/load)
        self.net_name = None
        for layer_i in self.layers_width:
            if self.net_name is None:
                self.net_name = 'Net_'+str(layer_i)
            else:
                self.net_name += "_"+str(layer_i)
       
        # net initialization        
        self.obs_net = LinearModel('LinearModel',self.lr ,  (self.plant.state.shape[0]-1) , (self.plant.state.shape[0]-1)+1,*(self.layers_width)).cuda()

        # secondary training parameters
        self.random_frequency = 0.8  # percentage of instances in which control action is taken randomly
        self.training_sequence_max_length = 5  # max number of steps before a training sequence is resetted
        self.training_split = 0.75  # training set/validation set split

        # max values of states and actuation
        self.max_val = np.array([5, 1.5, np.pi/2, 0.35, 5])
        self.act_max = 3

        # saving/visualization parameters
        self.plot_last_n = 5000
        self.visualize_every_n = 2000
        self.save_every_n = 10000

        # initializations
        self.loss_history = np.empty((1,2),dtype = np.float)
        self.storage = []


    ##########################################################################
    def load_net(self, net_name, device, net_version = None, path_log = None, load_history = True ):
        if path_log is None:
            path_log = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'NeuralNetworks' )
        
        # NN and optimizer data is loaded
        self.obs_net.load_net_params(path_log, net_name, device)
    
        # if net is loaded, net name is re-generated
        self.net_name = None
        for i in range(1,self.obs_net.net_depth+1):
            if self.net_name is None:
                self.net_name = 'Net_'+str(getattr(self.obs_net, 'n_layer_'+str(i)))
            else:
                self.net_name += "_"+str(getattr(self.obs_net, 'n_layer_'+str(i)))
        
        # training history is loaded if necessary        
        if load_history:
            self.load_history(net_version)
       

    ##########################################################################
    def load_history(self, net_version):

        tr_loss_filename = self.net_name + "_tr_loss.csv"
        val_loss_filename =self.net_name +  "_val_loss.csv"
       
        with open(tr_loss_filename, 'rb') as a:
            tr_loss_array = np.loadtxt(a, delimiter=",")
            
        with open(val_loss_filename, 'rb') as a:
            val_loss_array = np.loadtxt(a, delimiter=",")
            
        self.loss_history = np.concatenate((tr_loss_array[:,1:], val_loss_array[:,1:]),axis = 1)
        
        if net_version is not None:
            self.loss_history = self.loss_history[:net_version ,:]
        
        # load training and validation sets
        self.load_stored_data()
                

    ##########################################################################
    def load_stored_data(self):
        # load training set
        tr_set_file_name = self.net_name + '_tr_set.obj'
        with open(tr_set_file_name, 'rb') as a:
            self.training_set = pickle.load(a)
        # load validation set
        val_set_file_name = self.net_name + '_val_set.obj'
        with open(val_set_file_name, 'rb') as a:
            self.validation_set = pickle.load(a)
        
        self.generate_datasets()

     
    ##########################################################################               
    # method is called to generate training and validation sets (when Net is not loaded externally)
    def store_data(self):
    
        sim_completed = False
        failed_sims = 0
    
        while not sim_completed:    
            done = False
            self.plant.reset(save_history = True, full_random = True)
            
            steps = 0
            while not done and steps < self.training_sequence_max_length:
                # perform real plant step
                if random.random() > self.random_frequency:
                    action = self.plant.get_controller_input()
                else:
                    action = np.array([np.round((2*random.random()-1),3 )*self.act_max])
        
                st, rew, done, info = self.plant.step(action)
                
                self.storage.append((torch.tensor(self.plant.state)[:-1].unsqueeze(0),\
                                     torch.tensor(action).unsqueeze(0), \
                                         torch.tensor(st[:-1]).unsqueeze(0)))
                steps +=1
                
            if len(self.storage) > self.max_samples_stored:
                sim_completed = True
            
        
        #divide list in training and validation set
        random.shuffle(self.storage)
        idx_split = round(len(self.storage)*self.training_split)
        self.training_set = self.storage[:idx_split]
        self.validation_set = self.storage[idx_split:]
        
        tr_set_file_name = self.net_name + '_tr_set.obj'
        with open(tr_set_file_name, 'wb') as a:
            pickle.dump(self.training_set, a)

        val_set_file_name = self.net_name + '_val_set.obj'
        with open(val_set_file_name, 'wb') as a:
            pickle.dump(self.validation_set, a)

        self.generate_datasets()


    ##########################################################################
    def generate_datasets(self):
        # calculate normalization parameters (will be used after each minibatch extraction)
        # generate normalized validation set (will not be normalized after each iteration)        
        
        state_torch = torch.cat(tuple(d[0] for d in self.training_set)).float().cuda()
        action_torch = torch.cat(tuple(d[1] for d in self.training_set)).float().cuda()
        
        std_st, mean_st = torch.std_mean(state_torch, dim=0)
        std_act, mean_act = torch.std_mean(action_torch, dim=0)
        
        self.norm_values = std_st, mean_st, std_act, mean_act
            
        # normalize validation set (ready for use)
        val_state_torch = torch.cat(tuple(d[0] for d in self.validation_set)).float().cuda()
        val_action_torch = torch.cat(tuple(d[1] for d in self.validation_set)).float().cuda()
        self.val_state_1_torch = torch.cat(tuple(d[2] for d in self.validation_set)).float().cuda()
        
        self.norm_val_state_torch  = (val_state_torch-mean_st)/std_st
        self.norm_val_action_torch = (val_action_torch-mean_act)/std_act


    ##########################################################################
    def update_1step_observer(self):
    
        # extract minibatch from training set
        minibatch = random.sample(self.training_set, self.batch_size)

        state_batch   = torch.cat(tuple(d[0] for d in minibatch)).float().cuda()
        action_batch  = torch.cat(tuple(d[1] for d in minibatch)).float().cuda()
        state_1_batch = torch.cat(tuple(d[2] for d in minibatch)).float().cuda()
        
        # normalize minibatch        
        std_st, mean_st, std_act, mean_act = self.norm_values
        
        norm_state_torch = (state_batch-mean_st)/std_st
        norm_action_torch = (action_batch-mean_act)/std_act

        # estimate next state with NN
        state_est = self.obs_net(torch.cat((norm_state_torch ,norm_action_torch), dim = 1))

        # weights update cycle
        self.obs_net.optimizer.zero_grad()
        loss_est = self.obs_net.criterion(torch.matmul(state_1_batch,self.torch_weight), \
                                          torch.matmul(state_est,self.torch_weight))
        loss_est.backward(retain_graph = False)
        """
        for p in list(filter(lambda p: p.grad is not None, self.obs_net.parameters())):
            print(p.grad.data.norm(2).item())
        """
        self.obs_net.optimizer.step()
        
        # evaluate on validation set (no state weighing)        
        with torch.no_grad():
            val_state_est = self.obs_net(torch.cat((self.norm_val_state_torch ,self.norm_val_action_torch), dim = 1))
            val_loss = self.obs_net.criterion(self.val_state_1_torch,val_state_est)
        
        return loss_est.item(), val_loss.item()


    ##########################################################################
    def updater_routine(self, net_version):
        
        for ii in tqdm(range(1+net_version, 1+net_version+self.n_epochs)):
            loss, val_loss = self.update_1step_observer()
            new_loss = np.array([loss, val_loss])[np.newaxis,:]
            self.loss_history = np.append(self.loss_history, new_loss, axis = 0)
            
            if ii >= self.plot_last_n and not ii % self.visualize_every_n:
                
                print(f'iteration = {ii}')
                print(f'training loss = {loss}')
                print(f'validation loss = {val_loss}')
                loss_array = self.loss_history[-self.plot_last_n:,0]
                val_loss_array = self.loss_history[-self.plot_last_n:,1]
                np.savetxt("loss_hist.csv", np.concatenate((np.arange(1,loss_array.shape[0]+1)[:,np.newaxis],loss_array[:,np.newaxis]), axis = 1), delimiter=",")
                np.savetxt("val_loss_hist.csv", np.concatenate((np.arange(1,val_loss_array.shape[0]+1)[:,np.newaxis],val_loss_array[:,np.newaxis]), axis = 1), delimiter=",")
    
                try:
                    plot_scatter(f = "loss_hist.csv",xs = None, ys = None, size = 30, colour = 'white',pch = '*', title = 'training loss')
                    plot_scatter(f = "val_loss_hist.csv",xs = None, ys = None, size = 30, colour = 'yellow',pch = '*', title = 'validation loss')
                except Exception:
                    pass                    

                if not ii % self.save_every_n:
                    self.obs_net.save_net_params(net_name = self.net_name +'_' +str(ii))

        loss_array = self.loss_history[1:,0]
        val_loss_array = self.loss_history[1:,1]
        
        tr_loss_filename = self.net_name + "_tr_loss.csv"
        val_loss_filename =self.net_name +  "_val_loss.csv"
        
        np.savetxt(tr_loss_filename, np.concatenate((np.arange(1,loss_array.shape[0]+1)[:,np.newaxis],loss_array[:,np.newaxis]), axis = 1), delimiter=",")
        np.savetxt(val_loss_filename, np.concatenate((np.arange(1,val_loss_array.shape[0]+1)[:,np.newaxis],val_loss_array[:,np.newaxis]), axis = 1), delimiter=",")


    ##########################################################################
    # test Abstract Model
    def test_net(self, reset_every = 10,  save_test_result = False):
    
        # generate states sequence
        self.plant.reset()
        done = False    
        steps = 0
    
        while not done:
            # perform real plant step
            action = self.plant.get_controller_input()
            #action = np.array([np.round((2*random.random()-1),3 )*self.act_max])            
            st, rew, done, info = self.plant.step(action)
            steps += 1
        
        # initialize vectors/tensors
        states_stored_np,inputs =  self.plant.get_complete_sequence()
        
        #states_stored_np = np.concatenate((self.cartpole.cartpole.state_archive,self.cartpole.cartpole.target_pos[:,np.newaxis]),axis = 1 )
        states_stored = torch.tensor(states_stored_np ).float().cuda()
        
        std_st, mean_st, std_act, mean_act = self.norm_values
        
        states_stored_norm = (states_stored[:,:-1]-mean_st)/std_st
        
        states_sequence_obs = np.zeros(states_stored.shape)
        states_sequence_obs[0,:-1] = states_stored_norm[0,:].cpu().numpy()
        states_sequence_obs_norm = states_sequence_obs.copy()
        
        state_norm = states_stored_norm[0,:].unsqueeze(0)
        
        #inputs = self.cartpole.cartpole.ctrl_inputs
        inputs_norm = (inputs-mean_act.item())/std_act.item()
        
        # evaluate NN and store values
        for i in range(states_sequence_obs.shape[0]-1):
            
            if i>1 and not i % reset_every:
                state_norm = states_stored_norm[i,:].unsqueeze(0)
            
            action_norm = torch.tensor(inputs_norm[i]).float().cuda().unsqueeze(0).unsqueeze(0)
        
            with torch.no_grad():
                next_state = self.obs_net(torch.cat((state_norm,action_norm), dim = 1))
            states_sequence_obs[i+1,:-1] = next_state.cpu().detach().numpy().copy()
            
            state_norm = ((next_state.clone()- mean_st)/std_st).detach().clone()
            states_sequence_obs_norm[i+1,:-1] = state_norm.cpu().detach().numpy().copy()

        # figure generation
        fig1 = plt.figure()
        
        ax1 = fig1.add_subplot(511)
        ax2 = fig1.add_subplot(512)
        ax3 = fig1.add_subplot(513)
        ax4 = fig1.add_subplot(514)
        ax5 = fig1.add_subplot(515)
        
        ax1.plot(states_stored_np[:,0])
        ax1.plot(states_sequence_obs[:,0])
        
        ax2.plot(states_stored_np[:,1])
        ax2.plot(states_sequence_obs[:,1])
        
        ax3.plot(states_stored_np[:,2])
        ax3.plot(states_sequence_obs[:,2])
        
        ax4.plot(states_stored_np[:,3])
        ax4.plot(states_sequence_obs[:,3])
        
        ax5.plot(inputs)
    
        fig1.show()
        
        if save_test_result:
            fig1.savefig("result.png", dpi = 300) # save figure
        

    ##########################################################################
    def plot_training_history(self, save_history = False):
        vert_zoom = np.minimum(0.005 , self.loss_history)
        
        fig = plt.figure()
        ax1 = fig.add_subplot(411)
        ax2 = fig.add_subplot(412)
        ax3 = fig.add_subplot(413)
        ax4 = fig.add_subplot(414)
        start_sample = 1
    
        ax1.plot(self.loss_history[start_sample:,0])
        ax2.plot(vert_zoom[start_sample:,0])
        ax3.plot(self.loss_history[start_sample:,1])
        ax4.plot(vert_zoom[start_sample:,1])
        
        if save_history:
            fig.savefig("training_hist.png", dpi = 300) # save figure

#%%

def train_net(batch_size = 256, lr = 0.0002, n_epochs = 5000 , max_samples_stored = 20000, \
         layers_width = (5,5), state_weights = (1,1,1,1), load_net_params = False, \
            net_version = 0):
    

    trainer = AbstractModelTrainer(batch_size = batch_size, lr = lr, n_epochs = n_epochs, \
                      max_samples_stored = max_samples_stored, layers_width = layers_width, \
                          state_weights = state_weights)
    
    if load_net_params:
        print('loading data...')        
        device = torch.device("cuda")
        net_name = trainer.net_name + "_" + str(net_version)
        trainer.load_net(net_name, device, net_version)
        
        trainer.load_stored_data()
        print('loading complete')
        
    else:
        trainer.store_data()
    
    
    trainer.updater_routine(net_version*load_net_params)
    
    trainer.plot_training_history()
    
    trainer.test_net()
    
    return trainer

#%%

def test_external(net_name):

    device = torch.device("cuda")
    
    trainer = AbstractModelTrainer()
    
    pathlog = os.path.join(os.getcwd(), 'NeuralNetworks')
    trainer.load_net(net_name, device, path_log = pathlog)
    
    trainer.test_net()
    
    return trainer

#%%

def remove_versions_other_than(net_name):
    
    device = torch.device("cpu")
    
    trainer = AbstractModelTrainer()
    
    pathlog = os.path.join(os.getcwd(), 'NeuralNetworks')
    trainer.load_net(net_name, device, path_log = pathlog, load_history = False)
    
    NN_dir = os.listdir(pathlog)
    iteration_string = net_name.replace(trainer.net_name,'')
    for fname in NN_dir:
        if fname.startswith(trainer.net_name):
            iteration_keeper = fname.endswith(iteration_string+'.pt') 
            if not iteration_keeper:
                os.remove(os.path.join(pathlog, fname))


#%%
if __name__ == "__main__":
    
    train_net(batch_size = args.batch_size, lr = args.lr, n_epochs = args.n_epochs, \
         max_samples_stored = args.memory, layers_width=args.layers_list, \
             state_weights = args.state_weights, load_net_params = args.load_net_params, \
            net_version = args.net_version)

#%%
#remove_versions_other_than('Net_30_30_10_100000')

#%%
#trainer = test_external('Net_50_50_10_100000')