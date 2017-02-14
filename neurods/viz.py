"""Functions for data visualization"""
from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
import itertools


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

# --- Created in week 3 ---
def find_squarish_dimensions(n):
    """Get balanced (approximately square) numbers of rows & columns for n elements

    Returns (nearly) sqrt dimensions for a given number. e.g. for 23, will
    return [5, 5] and for 26 it will return [6, 5]. Useful for creating displays of
    sets of images. Always sets x greater than y if x != y

    Returns
    -------
    x : int
       larger dimension (if not equal)
    y : int
       smaller dimension (if not equal)
    """
    sq = np.sqrt(n)
    if round(sq)==sq:
        # If n is a perfect square
        x, y = sq, sq
    else:
        # Take either bigger square (first entries) or asymmetrical square (second entries)
        x_poss = [np.ceil(sq), np.ceil(sq)]
        y_poss = [np.ceil(sq), np.floor(sq)]
        n_elements = [x*y for x, y in zip(x_poss, y_poss)]
        err = np.array([n_e-n for n_e in n_elements])
        # Make sure negative values will not be chosen as the minimum
        err[err<0] = 1000 
        best_idx = np.argmin(err)
        x = x_poss[best_idx]
        y = y_poss[best_idx]
    return x, y

def slice_3d_array(volume, axis=0, nrows=None, ncols=None, 
                   fig=None, vmin=None, vmax=None, **kwargs):
    """Slices 3D array along arbitrary axis

    Parameters
    ----------
    volume : np.array (3D)
        Data to be shown
    axis : int | 0, 1, [2] (optional)
        axis along which to divide the array into slices
    nrows : int (optional)
        number of rows
    ncols : int (optional)
        number of columns
    fig : plt.figure
        figure in which to plot array slices

    Other Parameters
    ----------------
    vmin, vmax : float or None
        Color axis minimum / maximum for all plots. If either is 
        `None`, defaults to min or max for `volume`
    kwargs : keyword arguments
        Other arguments are passed to imshow()
    """
    if fig is None:
        fig = plt.figure()
    if nrows is None or ncols is None:
        ncols, nrows = find_squarish_dimensions(volume.shape[axis])
    if vmin is None:
        vmin = np.nanmin(volume)
    if vmax is None:
        vmax = np.nanmax(volume)
    left_edges = np.linspace(0, 1, ncols+1)[:-1]
    bottom_edges = np.linspace(1, 0, nrows+1)[1:]
    width = 1 / ncols
    height = 1 / nrows
    bottoms, lefts = zip(*list(itertools.product(bottom_edges, left_edges)))
    slices = np.split(volume, volume.shape[axis], axis=axis)
    for i, sl in enumerate(slices):
        ax = fig.add_axes((lefts[i], bottoms[i], width, height))
        ax.imshow(sl.squeeze(), vmin=vmin, vmax=vmax, **kwargs)
        ax.set_axis_off()
    return fig