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
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


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

xdata = preprocessing.scale(xdata)
# split data
xdata_train, xdata_test = train_test_split(xdata,test_size=.25)


#************* run VAE
N = xdata.shape[1] # data size
M = 100 # minibatch size
Dx = 5
Dz = 2
xdata_train = torch.from_numpy(xdata_train).type(torch.FloatTensor)
xdata_test = torch.from_numpy(xdata_test).type(torch.FloatTensor)
train_dataloader = DataLoader(xdata_train, batch_size=M)
test_dataloader = DataLoader(xdata_test, batch_size=M)

# setup the autoencoder
encoder = nn.Sequential(
      nn.Linear(Dx, 100),
      nn.ReLU(),
      MultiOutputLinear(100, [Dz, Dz]),
    )

decoder = nn.Sequential(
      nn.Linear(Dz, 100),
      nn.ReLU(),
      MultiOutputLinear(100, [Dx, Dx]),
    )

autoencoder = GaussianVAE(encoder, decoder, L=10)

# setup the optimizer
#learning_rate = 3e-2
optimizer = Adam(autoencoder.parameters())#, lr=learning_rate)


# optimize
num_epochs = 30
elbo_train = np.zeros(num_epochs)
elbo_test = np.zeros(num_epochs)
MSE_train = np.zeros(num_epochs)
MSE_test = np.zeros(num_epochs)
for epoch in range(num_epochs):

    # compute test ELBO
    for batch_i, batch in enumerate(test_dataloader):
        data = Variable(batch, requires_grad=False)
        elbo_test[epoch] += autoencoder.elbo(data).data[0]
        
        latent_data = autoencoder.encoder(data)
        recover_data = autoencoder.decoder(latent_data[0])
        MSE_test[epoch] += (data-recover_data[0]).pow(2).mean().data[0]  
    elbo_test[epoch] /= len(test_dataloader)
    MSE_test[epoch] /= len(test_dataloader)

    # compute training ELBO
    for batch_i, batch in enumerate(train_dataloader):
        data = Variable(batch, requires_grad=False)

        autoencoder.zero_grad()
        loss = -autoencoder.elbo(data)
        loss.backward()
        elbo_train[epoch] += -loss.data[0]
        
        latent_data = autoencoder.encoder(data)
        recover_data = autoencoder.decoder(latent_data[0])
        MSE_train[epoch] += (data-recover_data[0]).pow(2).mean().data[0]
        optimizer.step()
    elbo_train[epoch] /= len(train_dataloader)
    MSE_train[epoch] /= len(train_dataloader)
    print('Epoch [{}/{}]\
           \n\tTrain ELBO: {}\n\tTest ELBO:  {}\
           \n\tTrain MSE: {}\n\tTestMSE: {}'.format(\
            epoch+1, num_epochs, \
            elbo_train[epoch], elbo_test[epoch],\
            MSE_train[epoch], MSE_test[epoch]))


#************ plotting
#elbo
plt.plot(elbo_train, label='training Lower Bound')
plt.plot(elbo_test, label='test Lower Bound')
plt.legend()
plt.show()

#reconstruction
x_latent = autoencoder.encoder(Variable(xdata_test,requires_grad=False))
x_recover = autoencoder.decoder(x_latent[0])[0]
plt.plot(xdata_test[:,0].numpy(),label='1st dim data')
plt.plot(x_recover[:,0].data.numpy(),label='1st dim decoded')
plt.legend()
plt.show()

#mse
plt.plot(MSE_train,label='training recovered MSE')
plt.plot(MSE_test, label='testing recovered MSE')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.show()

# visualize embedded space
xdata = torch.from_numpy(xdata).type(torch.FloatTensor)
x_latent = autoencoder.encoder(Variable(xdata,requires_grad=False))
x_latent = x_latent[0] # take means, ignore variances
x_latent = x_latent.data.numpy()
plt.scatter(x_latent[:,0],x_latent[:,1])
plt.xlabel('VAE 1')
plt.ylabel('VAE 2')
plt.title('VAE embedding 1000 simulations')
plt.show()

# visualize embedded space for each knot point
import matplotlib.cm as cm
N = 4000
x_latent_time = np.reshape(x_latent,(T,N,Dz))
colors = cm.rainbow(np.linspace(0,1,T))
for i in range(T):
    plt.scatter(x_latent_time[i,:,0],x_latent_time[i,:,1],color=colors[i],alpha=0.3,label=str(i))
plt.legend()
plt.xlabel('VAE 1')
plt.ylabel('VAE 2')
plt.title('VAE embedding 1000 simulations')
plt.show()

# visualize each knot separately
figs, axs = plt.subplots(3,7)
figs.subplots_adjust(hspace = .5, wspace=.001)
axs = axs.ravel()
colors = cm.rainbow(np.linspace(0,1,T))
for i in range(T):

    axs[i].scatter(x_latent_time[i,:,0],x_latent_time[i,:,1],color=colors[i],alpha=0.3)
    axs[i].set_title(str(i))
plt.show()

#**** plot some individual paths (by initial state?)
xdata_time = np.concatenate((dataset['state'],dataset['control']),axis=1)
xdata_time = xdata_time[:N,:,:]
xdata_time = np.transpose(xdata_time,(2,0,1)) #obs x knots x dim
# get  distance metric of initial conditions
xdata_x0 = np.squeeze(xdata_time[1,:,0:4]) # drop control
from scipy.spatial.distance import pdist,squareform
y = squareform(pdist(xdata_x0,'euclidean'))

# now get examples from one point, and plot sorted by distance to example point
import random
ind = random.randint(0,N)
dist = y[ind,:]
idx = np.argsort(dist)

# plot vae examples
plt.scatter(x_latent[:,0],x_latent[:,1])
plt.xlabel('VAE 1')
plt.ylabel('VAE 2')

numex = 10
colors = cm.rainbow(np.linspace(0,1,numex))
for i in range(numex):
    plt.plot(x_latent_time[:,idx[i],0],x_latent_time[:,idx[i],1],color=colors[i])
plt.show()


# plot only 3
plt.scatter(x_latent[:,0],x_latent[:,1])
plt.xlabel('VAE 1')
plt.ylabel('VAE 2')
plt.plot(x_latent_time[:,idx[0],0],x_latent_time[:,idx[0],1],color='r',label='sample')
plt.plot(x_latent_time[:,idx[5],0],x_latent_time[:,idx[5],1],color='g',label='nearby')
plt.plot(x_latent_time[:,idx[500],0],x_latent_time[:,idx[500],1],color='b',label='far')
plt.legend()
plt.show()
 
#**** save data
savepath = 'C:\\Users\\Kevin\\Documents\\Classes\\cs281\\vae\\coders_5000.dat'
torch.save(autoencoder,savepath)
    