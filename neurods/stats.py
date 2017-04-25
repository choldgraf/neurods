import numpy as np
from numpy.linalg import inv
from scipy.stats import zscore

def standard_error(data, axis=0):
    """Calculate the standard error over an axis in data.

    Parameters
    ----------
    data : ndarray
        The data to use for standard error calculation.
    axis : int
        The axis to calculate standard error across.

    Returns
    -------
    ste : array
        The standard error of the data along the axis specified.
    """
    std = np.std(data, axis=0)
    ste = std / np.sqrt(data.shape[0])
    return ste

def ols(X, Y):
    """Estimate weights for X given data Y

    Parameters
    ----------
    X : array
        Stimulus design matrix; should be (time [TRs] x conditions [or features])
    Y : array
        Data to use to fit weights; size is (time [TRs] x voxels)

    Returns
    -------
    B : array
        weights for each condition or feature in X; size is (conditions [or features] x voxels)
    """
    B = np.dot(inv(np.dot(X.T, X)), np.dot(X.T, Y))
    return B

def _rand_index(n_datapts, block_size=0.2, replace=True):
    """Helper function to get random index for data of a given size, 

    """
    if block_size < 1:
        # Block size is given as a fraction of the full timecourse    
        n_blocks = int(block_size * n_datapts)
        block_index = np.arange(n_datapts).reshape([n_blocks, -1])
        sample_index = np.random.choice(n_blocks, n_blocks, replace=replace)
        sample_index = block_index[sample_index].reshape([-1])
    else:
        # Block size is given in absolute units
        n_blocks = int(np.ceil(n_datapts / block_size))
        block_bounds = np.ceil(np.linspace(0, n_datapts, n_blocks + 1)).astype(np.int32)
        block_index = [np.arange(st, fin) for st, fin in zip(block_bounds[:-1], block_bounds[1:])]
        sample_index = np.random.choice(n_blocks, n_blocks, replace=replace)
        sample_index = np.hstack([block_index[si] for si in sample_index])
    return sample_index

def randomize_ols(X, Y, block_size=0.2):
    """compute Ordinary Least Squares regression with a random sample of the original data


    """
    sample_index = _rand_index(X.shape[0], block_size=block_size)
    return ols(X[sample_index], Y[sample_index])

def compute_correlation(matrix_1, matrix_2):
    """Compute correlation between columns of two matrices
    
    `matrix_1` and `matrix_2` can be any type of 2D array. Correlations are computed
    between columns of each matrix (i.e. between the first column of `matrix_1` and
    the first column of `matrix_2`, the second column of `matrix_1` and the second column 
    of `matrix_2`, etc). Often time will be the first dimension, but it does not have 
    to be. 

    Parameters
    ----------
    matrix_1 : array
        2D array (e.g., fMRI data)
    matrix_2 : array
        2D array #2 (e.g., predicted responses)
    
    Returns
    -------

    """
    matrix_1_norm  = zscore(matrix_1, axis=0)
    matrix_2_norm  = zscore(matrix_2, axis=0)
    corr = np.mean(matrix_1_norm * matrix_2_norm, axis=0)
    return corr