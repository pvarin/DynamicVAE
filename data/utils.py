import numpy as np

def random_split(X, axis=0, p=.9):
    '''
    Randomly selects a fraction p of elements in X along a given axis
    '''
    mask = np.random.binomial(1,p,size=X.shape[axis]).astype(bool)
    X1 = X.compress(mask, axis=axis)
    X2 = X.compress(np.logical_not(mask), axis=axis)
    return X1, X2