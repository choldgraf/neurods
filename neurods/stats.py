import numpy as np

def standard_error(data, axis=0):
    """Calculate the standard error over an axis in data.

    Parameters
    ----------
    data : ndarray
        The data to use for standard error calculation.
    axis : int
        The axis to calculate standard error across.
    """
    std = np.std(data, axis=0)
    ste = std / np.sqrt(data.shape[0])
    return ste


def bootstrap_func(data, func=np.mean, ci=[2.5, 97.5], n_boot=1000, axis=0):
    """Bootstrap a function across the axis of a dataset.

    This is useful for calculating the statistics (e.g., the mean) without
    making as many parametric assumptions.

    Parameters
    ----------
    data : ndarray
        The data to use for calculating a bootstrap statistics.
    func : callable
        The function to compute across bootstraps. Must accept an ndarray as
        input, and have an "axis" parameter that it will compute the function
        across. Must output a single value for the dimension it is collapsing
        across.
    ci : list of floats, shape (n_percentiles,)
        The percentiles to return for the confidence interval. Normally this is
        a low/high limit of a 95/99% confidence interval. It defaults to 95%.
    n_boot : int
        The number of bootstraps to carry out.
    axis : int
        The axis across which to perform the bootstrap.

    Returns
    -------
    boot_results : array, shape (n_percentiles, ~data.shape)
        The results of the bootstrap. The dimension in data across which the
        statistic is calculated will be gone. The first dimension will be the
        percentiles of the bootstrap results.
    """
    len_axis = data.shape[axis]
    boot_data = np.zeros([n_boot, len_axis])
    rand_ixs = np.array([np.random.randint(0, len_axis, len_axis)
                         for ii in range(n_boot)])
    boot_results = []
    for boot_ixs in rand_ixs:
        boot_data = data.take(boot_ixs, axis=axis)
        boot_results.append(func(boot_data, axis=axis))
    boot_results = np.asarray(boot_results)
    return np.percentile(boot_results, ci, axis=0)
        