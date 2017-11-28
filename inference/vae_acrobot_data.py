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


# reshape data, remove time component
xdata = np.concatenate((dataset['state'],dataset['control']),axis=1)
xdata = np.transpose(xdata,(1,0,2))
xdata = np.reshape(xdata,(5,T*N)).T
xdata = xdata[:50000,:]
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

plt.plot(elbo_train, label='training Lower Bound')
plt.plot(elbo_test, label='test Lower Bound')
plt.legend()
plt.show()

x_latent = autoencoder.encoder(Variable(xdata_test,requires_grad=False))
x_recover = autoencoder.decoder(x_latent[0])[0]
plt.plot(xdata_test[:,0].numpy(),label='1st dim data')
plt.plot(x_recover[:,0].data.numpy(),label='1st dim decoded')
plt.legend()
plt.show()


