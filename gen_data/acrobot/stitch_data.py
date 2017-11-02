import csv
import numpy as np
import json
import os

def stitch_data(path,prefix,datafile):
	# load filenames
	filenames = [os.path.join(path,f) for f in os.listdir(path) if os.path.isfile(os.path.join(path,f)) and prefix in f]

	# initialize dataset
	traj = np.loadtxt(filenames[0],dtype=np.float64,delimiter=',',ndmin=2)
	Nx, Nt = traj.shape
	N = len(filenames)
	dataset = np.zeros((N,Nx,Nt),dtype=np.float64)

	# populate dataset
	for i,f in enumerate(filenames):
		dataset[i,:,:] = np.loadtxt(f,dtype=np.float64,delimiter=',')

	# save dataset
	np.save(datafile,dataset)

if __name__ =='__main__':
	# parse the control, time, and state data
	stitch_data("./data","acrobot_trajectory_time_","acrobot_trajectory_time")
	stitch_data("./data","acrobot_trajectory_control_","acrobot_trajectory_control")
	stitch_data("./data","acrobot_trajectory_state_","acrobot_trajectory_state")