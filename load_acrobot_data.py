import numpy as np

filepath = 'C:/Users/Kevin/Documents/Classes/cs281/DynamicVAE/gen_data/acrobot/'

dataset = dict()
dataset['time'] = np.load(filepath + 'acrobot_trajectory_time.npy')
dataset['state'] = np.load(filepath + 'acrobot_trajectory_state.npy')
dataset['control'] = np.load(filepath + 'acrobot_trajectory_control.npy')
i = 0 # the index of the trajectory
t = 0 # the time index
state_traj = dataset['state'][i,:,:]
control_traj = dataset['control'][i,:,:]
state = state_traj[:,t]
control = control_traj[:,t]