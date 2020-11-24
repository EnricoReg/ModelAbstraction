#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 13:57:39 2020

@author: rodpod21
"""

# efficiency_map approximation

import torch
import numpy as np
from tqdm import tqdm

from NeuralNetworks.Linear_NNs import LinearModel

from platooning_env.model_platooning_energy import ElMotor


emot = ElMotor()
emot.plotEffMap()   

NN_map = LinearModel('LinearModel',0.0002, 1, 2, 10,15, 15, 5 )


from torch.utils.data import DataLoader, TensorDataset




memory_size = 1000000
batch_size = 1024
n_epochs = 20000
train_split = 0.75
sample_split = round(train_split*memory_size)

def train_net():

    loss_hist = []

    
    speed_data = np.random.random(memory_size)[:,np.newaxis]*emot.max_speed
    torque_data = np.random.random(memory_size)[:,np.newaxis]*emot.max_torque
    
    eff_data = emot.getEfficiency(speed_data, torque_data)
    
    idx = np.logical_not(np.isnan(eff_data))[:,0]
    
    fig = plt.figure()
    ax = plt.scatter(speed_data[idx],torque_data[idx])
    

    
    
    X = torch.tensor(np.concatenate((speed_data[idx]/emot.max_speed,torque_data[idx]/emot.max_torque),axis = 1)).float()
    Y = torch.tensor(eff_data[idx]).float()
    
    train_dataset = TensorDataset(X[:sample_split,:], Y[:sample_split])
    val_dataset = TensorDataset(X[sample_split:,:], Y[sample_split:])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #val_dataloader = DataLoader(val_dataset, batch_size=512, shuffle=True)    
    dataiter = iter(train_dataloader)
    for _ in tqdm(range(n_epochs)):
    #for batch_idx, (x, y) in enumerate(train_dataloader):
        #print(batch_idx, x.shape, y.shape)
        
        
        try:
            x,y = dataiter.next()
        except:
            dataiter = iter(train_dataloader)
            x,y = dataiter.next()
        
        
        y_est = NN_map(x)
        NN_map.optimizer.zero_grad()
        
        loss = NN_map.criterion(y, y_est)
        loss.backward()
        NN_map.optimizer.step()
        
        with torch.no_grad():
            y_est_val = NN_map(val_dataset.tensors[0])
            val_loss = NN_map.criterion(y_est_val, val_dataset.tensors[1])
            
        loss_hist.append([loss.item(), val_loss.item()])
    
    #print(f'last idx {batch_idx}')
    loss_hist = np.array(loss_hist)
    return loss_hist    
    
    
loss_hist = train_net()
    
fig = plt.figure()
ax_1 = fig.add_subplot(211)
ax_2 = fig.add_subplot(212)
ax_1.plot(loss_hist[-5000:,0])
ax_2.plot(loss_hist[-5000:,1])


xx,yy = np.meshgrid(emot.speed_vect, emot.torque_vect)
xx_norm = xx/emot.max_speed
yy_norm = yy/emot.max_torque

test_data = torch.stack((torch.tensor(xx_norm),torch.tensor(yy_norm)),dim = 2)
with torch.no_grad():
    test_y = NN_map(test_data.float()).squeeze(2).cpu().detach().numpy()

test_y[yy >  emot.f_max_rq(xx) ] = np.nan

fig1 = plt.figure()
ax1 = plt.contourf(emot.speed_vect, emot.torque_vect, test_y, levels = 30 ,cmap = 'jet')
plt.plot(emot.EM_w_list,emot.EM_T_max_list , 'k')
plt.colorbar(ax1)
plt.show()


fig2 = plt.figure()
ax2 = plt.contourf(emot.speed_vect, emot.torque_vect, test_y-emot.eff_matrix, levels = 30 ,cmap = 'jet')
plt.plot(emot.EM_w_list,emot.EM_T_max_list , 'k')
plt.colorbar(ax2)

#%%
