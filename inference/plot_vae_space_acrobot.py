# plotting
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import math

# pytorch
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

# VAE
from vae import *
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def plotAcrobot(ax, state):
    # state is 4x1 vector (5x1? include control?)
    # ax is current axes to plot on in subplots
    l1,l2 = 1,1
    r = 0.05
    theta1 = state[0]
    theta2 = state[1]
    x1 = l1 * np.sin(theta1)
    y1 = -l1 * np.cos(theta1)
    x2 = x1 + l2 * np.sin(theta2)
    y2 = y1 - l2 * np.cos(theta2)
    ax.plot([0, x1, x2], [0, y1, y2], lw=2, c='k')
    c0 = Circle((0, 0), r/2, fc='k', zorder=10)
    c1 = Circle((x1, y1), r, fc='b', ec='b', zorder=10)
    c2 = Circle((x2, y2), r, fc='r', ec='r', zorder=10)
    ax.add_patch(c0)
    ax.add_patch(c1)
    ax.add_patch(c2)
    
    # plot rotation vectors    
    u1,v1 = -np.cos(theta2),np.sin(theta2)
    u2,v2 = -np.cos(theta1),np.sin(theta1)
    
    # me messing around...
    u1,v1 = np.sin(theta1),np.cos(theta1)
    u2,v2 = np.sin(theta2),np.cos(theta2)
    u1,v1 = np.sin(theta1+np.pi/2),np.cos(theta1+np.pi/2)
    u2,v2 = np.sin(theta2+np.pi/2),np.cos(theta2+np.pi/2)
    u1,v1 = y1,-x1
    u2,v2 = y2-y1,-x2-x1
    
    u1,v1 = u1*state[2],v1*state[2]
    u2,v2 = u2*state[3],v2*state[3]    
    
    ax.quiver([x1,x2],[y1,y2],[u1,u2],[v1,v2],scale=1000)
    
    # fix axes (to l1+l2)
    ax.set_xlim(-l1-l2-r, l1+l2+r)
    ax.set_ylim(-l1-l2-r, l1+l2+r)
    ax.set_aspect('equal')
    ax.axis('off')
    
    

# load data
filepath = 'C:/Users/Kevin/Documents/Classes/cs281/DynamicVAE/gen_data/acrobot/'

dataset = dict()
dataset['time'] = np.load(filepath + 'acrobot_trajectory_time.npy')
dataset['state'] = np.load(filepath + 'acrobot_trajectory_state.npy')
dataset['control'] = np.load(filepath + 'acrobot_trajectory_control.npy')

T = dataset['state'].shape[2] # knots
N = dataset['state'].shape[0] # observations
N = 4000

# reshape data, remove time component
xdata = np.concatenate((dataset['state'],dataset['control']),axis=1)
xdata = xdata[:N,:,:]
xdata = np.transpose(xdata,(1,0,2))
xdata = np.reshape(xdata,(5,T*N)).T

# load model
modelpath = 'C:/Users/Kevin/Documents/Classes/cs281/vae/coders_4000_noscale.dat'
autoencoder = torch.load(modelpath)


# make vae plots
unit = 1
nx,ny = (12,12)
x = np.linspace(-unit,unit,nx)
y = np.linspace(-unit,unit,ny)
xv,yv = np.meshgrid(x,y)
xv = np.reshape(xv,nx*ny)
yv = np.reshape(yv,nx*ny)

x_grid = np.vstack((xv,yv))
x_grid = torch.from_numpy(x_grid.T).type(torch.FloatTensor)
x_square = autoencoder.decoder(Variable(x_grid,requires_grad=False))
x_square = x_square[0].data.numpy()

fig,axarr = plt.subplots(nx,ny)
axarr = axarr.ravel()
for j in range(nx*ny):
    plotAcrobot(axarr[j],x_square[j,:4])


fig.subplots_adjust(wspace=0, hspace=0)
plt.show()



