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


# --- Created in week 8 ---
def slice_3d_array(volume, axis=2, fig=None, vmin=None, vmax=None,
                   cmap=plt.cm.gray, nr=None, nc=None):
    """Slices 3D array along arbitrary axis

    Parameters
    ----------
    volume : np.array (3D)
        Data to be shown
    axis : int | 0, 1, [2] (optional)
       axis along which to divide the array into slices
    fig : plt.figure
        figure in which to plot array slices

    Other Parameters
    ----------------
    vmin : float [max(volume)] (optional) 
       color axis minimum
    vmax : float [min(volume)] (optional)
       color axis maximum
    cmap : matplotlib colormap instance [plt.cm.gray] (optional)
        color map for data
    nr : int (optional)
       number of rows
    nc : int (optional)
       number of columns
    """
    if fig is None:
        fig = plt.figure()
    if nr is None or nc is None:
        nc, nr = find_squarish_dimensions(volume.shape[axis])
    if vmin is None:
        vmin = np.nanmin(volume)
    if vmax is None:
        vmax = np.nanmax(volume)
    l_edges = np.linspace(0, 1, nc+1)[:-1]
    b_edges = np.linspace(1, 0, nr+1)[1:]
    width = 1/float(nc)
    height = 1/float(nr)
    bottoms, lefts = zip(*list(itertools.product(b_edges, l_edges)))
    for ni, sl in enumerate(np.split(volume, volume.shape[axis], axis=axis)):
        ax = fig.add_axes((lefts[ni], bottoms[ni], width, height))
        ax.imshow(sl.squeeze(), vmin=vmin, vmax=vmax, interpolation="nearest",
                  cmap=cmap)
        ax.set_xticks([])
        ax.set_yticks([])
    return fig


# --- Created in week 9 ---
def show_design_matrix(X, conditions=None, time_unit='s'):
    """Simple display of design matrix for an experiment, as an image"""
    n_trs, n_channels = X.shape
    plt.imshow(X.T)
    plt.xlabel('Time ({})'.format(time_unit))
    plt.ylabel('Condition')
    if conditions is None:
        conditions = np.arange(1, n_channels+1)
    plt.yticks(range(n_channels), conditions)


def plot_activity_on_brain(x, y, act, im, smin=10, smax=100, vmin=None,
                           vmax=None, alpha=None, ax=None, cmap=None,
                           name=None):
    """Plot activity as a scatterplot on a brain.

    Parameters
    ----------
    x : array, shape (n_channels,)
        The x positions of electrodes
    y : array, shape (n_channels, )
        The y positions of electrodes
    act : array, shape (n_channels,)
        The activity values to plot as size/color on electrodes
    im : ndarray, passed to imshow
        An image of the brain to match with x/y positions
    smin : int
        The minimum size of points
    smax : int
        The maximum size of points
    vmin : float | None
        The minimum color value / size cutoff
    vmax : float | None
        The maximum color value / size cutoff
    alpha : float | ndarray | None
        The alpha value for each channel.
    ax : axis | None
        An axis object to plot to
    cmap : matplotlib colormap | None
        The colormap to plot
    name : string | None
        A string name for the plot title.

    Returns
    -------
    ax : axis
        The axis object for the plot
    """
    # Handle defaults
    if ax is None:
        _, ax = plt.subplots()
    if cmap is None:
        cmap = plt.cm.coolwarm
    alpha = 1. if alpha is None else alpha
    if isinstance(alpha, (int, float, np.int)):
        alpha = np.repeat(alpha, act.shape[0])
    vmin = act.min() if vmin is None else vmin
    vmax = act.max() if vmax is None else vmax

    # Define colors + sizes
    act_norm = (act - vmin) / float(vmax - vmin)
    colors = cmap(act_norm)
    colors[:, -1] = alpha
    sizes = np.clip(np.abs(act) / float(vmax), 0, 1)  # Normalize bw 0 and 1
    sizes = MinMaxScaler((smin, smax)).fit_transform(sizes[:, np.newaxis])

    # Plotting
    ax.imshow(im)
    ax.scatter(x, y, s=sizes, c=colors, cmap=cmap)
    ax.set_title(name, fontsize=20)
    return ax


def xy_to_layout(xy, image, ch_names=None, flipy=True):
    """Convert xy coordinates to an MNE layout, normalized by an image

    Parameters
    ----------
    xy : array, shape (n_points, 2)
        xy coordinates of electrodes
    image : ndarray
        The image of the brain that xy positions plot on top of
    ch_names : None | list of strings, length n_points
        Channel names for each xy point
    flipy : bool
        Whether or not to flip the y coordinates of xy points after
        creating the layout (use if xy points are from the bottom-up,
        because imshow will plot from the top-down)

    Returns
    -------
    lt : instance of MNE layout
        The layout of these xy positions
    """
    if ch_names is None:
        ch_names = ['%s' % ii for ii in range(xy.shape[0])]
    lt = mne.channels.generate_2d_layout(xy, bg_image=image,
                                         ch_names=ch_names)

    if flipy is True:
        # Flip the y-values so they plot from top to bottom
        lt.pos[:, 1] = 1 - lt.pos[:, 1]
    return lt


def plot_tfr(times, freqs, tfr, ax=None, use_log=True,
             baseline=None, linear_y=True):
    """Plot a time-frequency representation.

    Parameters
    ----------
    times : array, shape (n_times,)
        The timepoints (in seconds) for the TFR.
    freqs : array, shape (n_freqs,)
        The frequencies for each frequency band of the TFR.
    tfr : array, shape (n_freqs, n_times)
        The TFR to plot.
    ax : matplotlib axis | None
        Plot the TFR on a pre-created axis. If None, one will be created.
    use_log : bool
        Whether to take the log of each value before plotting.
    baseline : array, shape (2,) | None
        The tmin/tmax of an (optional) baseline.
    linear_y : bool
        Whether to force y-values in the plot to be linear. Useful if
        frequencies are log-spaced.

    Returns
    -------
    ax : matplotlib axis
        The axis with the TFR plot.
    """
    if ax is None:
        f, ax = plt.subplots()
    if use_log is True:
        tfr = np.log(tfr)
    if baseline is not None:
        tfr = mne.baseline.rescale(tfr, times, baseline, mode='zscore')
        vmin, vmax = (-6, 6)
    else:
        vmin, vmax = (None, None)
    if linear_y is True:
        freqs = np.arange(freqs.shape[0])
    ax.pcolormesh(times, freqs, tfr, cmap=plt.cm.coolwarm,
                  vmin=vmin, vmax=vmax)
    ax.set_xlim([times.min(), times.max()])
    ax.set_ylim([freqs.min(), freqs.max()])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    return ax


def plot_tfr_topo(atfr, im, lt, baseline=(None, 0), mode='zscore',
                  vmin=-5, vmax=5, cmap=None, show=True):
    """Plot a topographic layout of an `AverageTFR` object.

    Parameters
    ----------
    atfr : instance of MNE `AverageTFR` object.
        The TFR data to plot
    im : ndarray of an image, shape (n_pix_h, n_pix,w, {3|4})
        The image to plot behind the TFR images.
    lt : instance of MNE `layout` object
        The layout corresponding to x/y locations of each electrode.
    baseline : None (default) or tuple of length 2
        The time interval to apply baseline correction.
        If None do not apply it. If baseline is (a, b)
        the interval is between "a (s)" and "b (s)".
        If a is None the beginning of the data is used
        and if b is None then b is set to the end of the interval.
        If baseline is equal ot (None, None) all the time
        interval is used.
    mode : None | 'ratio' | 'zscore' | 'mean' | 'percent' | 'logratio' | 'zlogratio' # noqa
        Do baseline correction with ratio (power is divided by mean
        power during baseline) or zscore (power is divided by standard
        deviation of power during baseline after subtracting the mean,
        power = [power - mean(power_baseline)] / std(power_baseline)), mean
        simply subtracts the mean power, percent is the same as applying ratio
        then mean, logratio is the same as mean but then rendered in log-scale,
        zlogratio is the same as zscore but data is rendered in log-scale
        first.
        If None no baseline correction is applied.
    vmin : float | None
        The mininum value an the color scale. If vmin is None, the data
        minimum value is used.
    vmax : float | None
        The maxinum value an the color scale. If vmax is None, the data
        maximum value is used.
    cmap : matplotlib colormap | str
        The colormap to use. Defaults to 'RdBu_r'.
    show : bool
        Whether to show the figure after creating it.

    Returns
    -------
    fig : matplotlib `Figure`
        The figure with TFR plots embedded
    """
    cmap = plt.cm.coolwarm if cmap is None else cmap
    fig = atfr.plot_topo(picks=range(len(atfr.ch_names)), layout=lt,
                         baseline=(None, 0), mode='zscore', vmin=-5, vmax=5,
                         cmap=plt.cm.coolwarm, show=False)
    mne.viz.add_background_image(fig, im)
    if show is True:
        fig.show()
    return fig
