import numpy as np
from utils import random_split

import torch

def load_torch_state_dataset(train_idx=None, test_idx=None):
    X_train = load_numpy_states()

    X_train, X_test = load_numpy_dataset(train_idx, test_idx)

def load_numpy_states(train_idx=None, test_idx=None):
    X = load_numpy()['states']
    if train_idx is None:
        train_idx, test_idx = random_split(np.arange(X.shape[0]))

    return X[], X_test

def load_numpy():
    dataset = dict()
    dataset['states'] = np.load('../gen_data/acrobot/acrobot_trajectory_state.npy')
    dataset['time'] = np.load('../gen_data/acrobot/acrobot_trajectory_time.npy')
    dataset['control'] = np.load('../gen_data/acrobot/acrobot_trajectory_control.npy')
    return dataset

def load_acrobot_torch():
    pass