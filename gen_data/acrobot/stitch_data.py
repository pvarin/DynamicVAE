import csv
import numpy as np
import json
import os

# load filenames
path = "./data"
prefix = "acrobot_trajectory_"
filenames = [os.path.join(path,f) for f in os.listdir(path) if os.path.isfile(os.path.join(path,f)) and prefix in f]

# initialize dataset
traj = np.loadtxt(filenames[0],dtype=np.float64,delimiter=',')
Nx, Nt = traj.shape
N = len(filenames)
dataset = np.zeros((N,Nx,Nt),dtype=np.float64)

# populate dataset
for i,f in enumerate(filenames):
	dataset[i,:,:] = np.loadtxt(f,dtype=np.float64,delimiter=',')

datafile = "acrobot_trajectory"
np.save(datafile,dataset) 