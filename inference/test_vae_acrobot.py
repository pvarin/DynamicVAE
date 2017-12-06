# plotting
import numpy as np
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

# pytorch
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

# VAE
from vae import *

# load data and seperate into training and test data
X = np.load('../gen_data/acrobot/acrobot_trajectory_state.npy')
X = X.transpose([0,2,1])
X = X.reshape((-1,X.shape[-1]))
X = normalize(X, axis=0, norm='l2')

mask = np.random.binomial(1,.9,size=X.shape[0]).astype(bool)
idx_train = np.where(mask)[0]
idx_test = np.where(mask==False)[0]


x_train = torch.Tensor(X[idx_train,:])
x_test = torch.Tensor(X[idx_test,:])
# dummy variables so we can use the Tensor Dataset object

# generate data
N = x_train.size(0)
M = 100 # minibatch size
Dx = x_train.size(1)
Dz = 2

z_train = torch.zeros(x_train.size(0),Dz)
z_test = torch.zeros(x_test.size(0),Dz)
# x_train, z_train, x_test, z_test = gen_data(Dx, Dz, data_size=N)
train_dataset = TensorDataset(x_train,z_train)
test_dataset = TensorDataset(x_test,z_test)
train_dataloader = DataLoader(train_dataset, batch_size=M)
test_dataloader = DataLoader(test_dataset, batch_size=M)

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
optimizer = Adam(autoencoder.parameters())
# optimizer = SGD(autoencoder.parameters(),lr=1e-5)

# optimize
num_epochs = 50
elbo_train = np.zeros(num_epochs)
elbo_test = np.zeros(num_epochs)
for epoch in range(num_epochs):
    # compute test ELBO
    for batch_i, batch in enumerate(test_dataloader):
        data = Variable(batch[0], requires_grad=False)
        elbo_test[epoch] += autoencoder.elbo(data).data[0]
    elbo_test[epoch] /= len(test_dataloader)

    # compute training ELBO
    for batch_i, batch in enumerate(train_dataloader):
        data = Variable(batch[0], requires_grad=False)

        autoencoder.zero_grad()
        loss = -autoencoder.elbo(data)
        loss.backward()
        elbo_train[epoch] += -loss.data[0]
        optimizer.step()
    elbo_train[epoch] /= len(train_dataloader)
    print('Epoch [{}/{}]\
           \n\tTrain ELBO: {}\n\tTest ELBO:  {}'.format(\
            epoch+1, num_epochs, \
            elbo_train[epoch], elbo_test[epoch]))

    torch.save(autoencoder.state_dict(),'acrobot_vae_2dim/acrobot_vae_parameters_epoch_{:03d}_adam.pt'.format(epoch))

plt.plot(elbo_train, label='training Evidence Lower Bound')
plt.plot(elbo_test, label='test Evidence Lower Bound')
plt.legend()
plt.show()