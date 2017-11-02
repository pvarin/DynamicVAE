import csv
import numpy as np
import json
import os

# load filenames
path = "."
prefix = "acrobot_trajectory_"
filenames = [f for f in os.listdir(path) if os.path.isfile(f) and prefix in f]

# initialize dataset
traj = np.loadtxt(filenames[0],dtype=np.float64,delimiter=',')
Nx, Nt = traj.shape
N = len(filenames)
dataset = np.zeros(N,dtype="int32, " + str((Nx,Nt)) + "float64")

# populate dataset
for i,f in enumerate(filenames):
	idx = range(i*4,(i+1)*4)
	dataset[i][0] = i
	# print dataset[i][1]
	# print np.loadtxt(f,dtype=np.float64,delimiter=',')
	dataset[i][1] = np.loadtxt(f,dtype=np.float64,delimiter=',')

datafile = "acrobot_trajectory"
np.save(datafile,dataset) 