import numpy as np
from mne.utils import _time_mask


def time_mask(times, tmin, tmax):
    """Return a boolean mask of times according to a tmin/tmax.

    Parameters
    ----------
    times : array, shape (n_times,)
        The times (in seconds) to mask.
    tmin : float or int
        The minimum time to include in the mask
    tmax : float or int
        The maximum time to include in the mask

    Returns
    -------
    mask : array, dtype bool, shape (n_times,)
        A boolean mask with values True between tmin and tmax.
    """
    return _time_mask(times, tmin, tmax)
