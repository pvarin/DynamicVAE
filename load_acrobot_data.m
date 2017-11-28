addpath(genpath('C:\Users\Kevin\Documents\Classes\cs281\DynamicVAE\npy-matlab-master'));
datapath = 'C:\Users\Kevin\Documents\Classes\cs281\DynamicVAE\gen_data\acrobot\';

time =readNPY([datapath 'acrobot_trajectory_time.npy']);
control = readNPY([datapath 'acrobot_trajectory_control.npy']);
state = readNPY([datapath 'acrobot_trajectory_state.npy']);