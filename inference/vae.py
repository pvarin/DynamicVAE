import math
import itertools
import torch
import torch.nn as nn
from torch.autograd import Variable

class MultiOutputLinear(nn.Module):
    '''
    A module that maps a single input to multiple outputs through linear layers
    This is helpful when the network returns multple parameters, such as the mean and covariance of a Gaussian
    '''
    def __init__(self, d_in, d_out):
        super(MultiOutputLinear, self).__init__()
        for i, d in enumerate(d_out):
            self.add_module('{}'.format(i), nn.Linear(d_in, d))

    def forward(self, x):
        return [m(x) for m in self.children()]

class GaussianVAE(nn.Module):
    '''
    A module that 
    '''
    def __init__(self, encoder, decoder, L=1):
        super(GaussianVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.L = L

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        # encode
        z_mean, z_logvar = self.encode(x)
        
        # reparameterize
        eps = Variable(torch.randn(self.L, *z_mean.size()), requires_grad=False)
        z_l = z_mean + (.5*z_logvar).exp()*eps

        # decode
        x_mean, x_logvar = self.decode(z_l)

        return z_mean, z_logvar, x_mean, x_logvar

    def elbo(self, x):
        '''
        Encodes then decodes the output
        ''' 
        return GaussianVAE_ELBO(x, *self(x))

    @property
    def input_dimension(self):
        return next(encoder.parameters()).size()[1]

    @property
    def latent_dimension(self):
        return next(decoder.parameters()).size()[1]

def GaussianVAE_ELBO(x, z_mean, z_logvar, x_mean, x_logvar):
    Dx = x_mean.size()[-1]

    # divergence between unit Gaussian and diagonal Gaussian
    divergence = - .5*torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp(),1)
    divergence = torch.mean(divergence,0) # sum over all of the data in the minibatch

    # likelihood approximated by L samples of z
    log_likelihood = -torch.sum((x - x_mean)*((x - x_mean)/x_logvar.exp()),2) - Dx/2.0*(math.log(2*math.pi) + torch.sum(x_logvar,2))
    log_likelihood = torch.mean(log_likelihood,0) # average over all of the samples to approximate expectation
    log_likelihood = torch.mean(log_likelihood,0) # sum over all of the data

    return -divergence + log_likelihood
    