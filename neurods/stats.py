import numpy as np


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

