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

def gen_data(Dx, Dz, data_size):
    '''
    Generates random data according to a nonlinear generative model
    Z->X
    '''
    # global parameter
    W = torch.randn(Dz, Dx)
    
    # training data
    Z_train = torch.randn(N, Dz).type(torch.FloatTensor)
    eps = 0.1*torch.randn(N, Dx).type(torch.FloatTensor)
    X_train = torch.sin(Z_train).mm(W) + eps
    
    # test data
    Z_test = torch.randn(N, Dz).type(torch.FloatTensor)
    eps = 0.1*torch.randn(N, Dx).type(torch.FloatTensor)
    X_test = torch.sin(Z_test).mm(W) + eps

    return X_train, Z_train, X_test, Z_test

# generate data
N = 10000 # data size
M = 100 # minibatch size
Dx = 10
Dz = 3
x_train, z_train, x_test, z_test = gen_data(Dx, Dz, data_size=N)
#train_dataset = TensorDataset(x_train,z_train)
#test_dataset = TensorDataset(x_test,z_test)
#train_dataloader = DataLoader(train_dataset, batch_size=M)
#test_dataloader = DataLoader(test_dataset, batch_size=M)
train_dataloader = DataLoader(x_train, batch_size=M)
test_dataloader = DataLoader(x_test, batch_size=M)


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
learning_rate = 3e-4
optimizer = Adam(autoencoder.parameters(), lr=learning_rate)

# optimize
num_epochs = 50
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


# look at MSE loss in encoding

x_latent = autoencoder.encoder(Variable(x_test,requires_grad=False))
x_recover = autoencoder.decoder(x_latent[0])[0]
plt.plot(x_test[:,0].numpy(),label='1st dim data')
plt.plot(x_recover[:,0].data.numpy(),label='1st dim decoded')
plt.legend()
plt.show()


