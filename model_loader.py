#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 17:49:17 2020

@author: rodpod21
"""

# model loader
from NeuralNetworks.Linear_NNs import LinearModel
import torch
import os
import numpy as np

model = LinearModel()

device = torch.device("cuda")
net_name = 'Net_50_50_15_150000'
path_log = os.path.join(os.getcwd(), 'NeuralNetworks')

# create input tensor
model.load_net_params(path_log, net_name, device)
state = np.random.randn(1,4)
action = np.random.randn(1,1)
tensor_in = torch.tensor(np.concatenate((state,action), axis = 1)).float().to(device)

# evaluate net
state_out = model(tensor_in)