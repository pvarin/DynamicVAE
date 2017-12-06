# EM version of PPCA...

# to optimize, just use weights found W and multiply them by Z



# plotting
import numpy as np
import matplotlib.pyplot as plt

# VAE
from vae import *
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import copy
from sklearn.decomposition import FactorAnalysis


#*************** load acrobot data
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

#xdata = preprocessing.scale(xdata)
# split data
xdata_train, xdata_test = train_test_split(xdata,test_size=.25)


#************* run VAE
N = xdata.shape[1] # data size
M = 100 # minibatch size
Dx = 5
Dz = 2

# create W matrix
W = np.random.normal(0, 1, (Dx,Dz))
Z = (np.linalg.inv(W.T.dot(W)).dot( W.T.dot(xdata.T))).T

# run EM on PCA (sigma^2 = 0)
maxiter = 1000
it = 0
Wdiff = np.inf
tol = 1e-10
Wdifftrack = np.array([])
while (it < maxiter) & (Wdiff > tol):
    Wold = copy.deepcopy(W)
    # E step
    Z = (np.linalg.inv(W.T.dot(W)).dot( W.T.dot(xdata.T))).T
    # M step
    W = (xdata.T.dot(Z)).dot(np.linalg.inv(Z.T.dot(Z)))
    # change
    Wdiff = np.sum((W-Wold)**2)
    Wdifftrack = np.append(Wdifftrack,Wdiff)
    # log likelihood
    #C = W.dot(W.T)
    #S = xdata.dot(xdata.T)
    #loglik = -N/2 * np.log(C) - 

Z = (np.linalg.inv(W.T.dot(W)).dot( W.T.dot(xdata.T))).T
plt.scatter(Z[:,0],Z[:,1])
plt.show()


# version 2 from SKlearn
estimator = FactorAnalysis(n_components=Dz)
estimator.fit(xdata)
Wp = estimator.components_
loglik = estimator.loglik_
