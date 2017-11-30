# plotting
import numpy as np
import matplotlib.pyplot as plt

# pytorch
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

# VAE
from vae import *


# load data
savepath = 'C:\\Users\\Kevin\\Documents\\Classes\\cs281\\vae\\coders.dat'
autoencoder = torch.load(savepath)

filepath = 'C:/Users/Kevin/Documents/Classes/cs281/DynamicVAE/gen_data/acrobot/'
dataset = dict()
dataset['time'] = np.load(filepath + 'acrobot_trajectory_time.npy')
dataset['state'] = np.load(filepath + 'acrobot_trajectory_state.npy')
dataset['control'] = np.load(filepath + 'acrobot_trajectory_control.npy')
T = dataset['state'].shape[2] # knots
N = dataset['state'].shape[0] # observations
N = 1000

xdata = np.concatenate((dataset['state'],dataset['control']),axis=1)
xdata = xdata[:N,:,:]
xdata = np.transpose(xdata,(1,0,2))
xdata = np.reshape(xdata,(5,T*N)).T
xdata = preprocessing.scale(xdata)


# generate new points and interpolate


