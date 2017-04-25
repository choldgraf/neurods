import numpy as np
import mne


def get_onsets(cond):
    """Convert a set of indicators for when a condition is on to onset indices for that condition
    
    Parameters
    ----------
    cond : array
        A 1D array of 1s and 0s (or a boolean array of Trues and Falses), indicating which 
        time indices of an experimental timecourse were part of a single given condition
    
    Returns
    -------
    onset_times : array
        onset time indices for `cond`
    """
    # (Note fancy syntax from above to pull out first element of a tuple)
    on_times, = np.nonzero(cond)
    # Choose 
    keepers = np.diff(np.hstack([-2, on_times]))>1
    onset_times = on_times[keepers]
    return onset_times

def extract_epochs(data, onset_indices, tmin=-0.1, tmax=0.6, sfreq=2048, is_verbose=False,
                   baseline_times=(None, 0), baseline_type='mean'):
    """Extract event time periods (epochs) in a given a time window around event onsets
    
    Parameters
    ----------
    data : array (2D)
        dimensions of of data should be (channels, time)
    onset_indices : array-like
        onset indices for a given condition or event type
    tmin, tmax : scalar
        values for time at start and end of condition, relative to condition onset
    sfreq : scalar
        sampling frequency for data
    is_verbose : boolean
        flag for 
    baseline_times : tuple
        time points to use for baseline computation (start, end). If start or end is None,
        
    Returns
    -------
    epochs : array
        array containing event epochs, of size (events, channels, time)
    times : array
        1D array of time points from `tmin` to `tmax`

    """
    epochs = []    
    for onset_index in onset_indices:
        # Find event onset / offset index
        if is_verbose:
            print('Onset index = %d'%onset_index)
        event_start_idx = onset_index + int(sfreq * tmin)
        event_end_idx = onset_index + int(sfreq * tmax)
        if is_verbose:
            print("Extracting indices from %d to %d"%(event_start_idx, event_end_idx))
        # Calculate window
        data_slice = data[:, event_start_idx:event_end_idx]
        #data_slice = zscore(data_slice, axis=1)
        if is_verbose:
            print("Data slice is of shape {}".format(data_slice.shape))
        epochs.append(data_slice)
    # Concatenate together all epochs we have extracted
    epochs = np.array(epochs)
    times = np.linspace(tmin, tmax, num=epochs.shape[-1])
    # Baseline by pre-stimulus times
    if baseline_type is not None:
        epochs = mne.baseline.rescale(epochs, times, baseline_times, baseline_type, verbose=is_verbose)
    return epochs, times
