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
    keepers = np.diff(np.hstack([-1, on_times]))>1
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

def url_to_interact(url, url_type='data8'):
    """Create an interact link from a URL in github or data-8.org.

    Parameters
    ----------
    url : string
        The URL of the file/folder you want to convert to an interact link.
    url_type : one of 'ds8' | 'data8'
        Whether the output URL should be attached to ds8 or data8.
    """
    # First define the repo name
    if not any([i in url for i in ['data-8', 'data8.org']]):
        raise ValueError('Provide a URL attached to a data-8 repository')
    if 'github.com' in url:
        repo_split = 'data-8/'
    elif 'data8.org' in url:
        repo_split = 'data8.org/'
    else:
        raise ValueError('Provide a URL for github.com or data8.org')
    repo = url.split(repo_split)[-1].split('/')[0]

    # Now pull file path/name
    name_split = 'gh-pages/' if 'github.com' in url else repo + '/'
    name = url.split(name_split)[-1]

    url_int = 'https://{2}.berkeley.edu/hub/interact?repo={0}&path={1}'.format(
        repo, name, url_type)
    print('Your interactive URL is:\n---\n{0}\n---'.format(url_int))
    return url_int
