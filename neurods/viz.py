"""Functions for data visualization"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import itertools
import mne


def set_figsize(dims):
    """Set the default figure size.

    Parameters
    ----------
    dims : list, length 2
        The width and height of the default figsize (in inches)
    """
    if len(dims) != 2:
        raise ValueError('dims should be (n_in_width, n_in_height)')
    plt.rcParams['figure.figsize'] = dims

