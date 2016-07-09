from mne.time_frequency import cwt_morlet
from mne.utils import warn
import numpy as np
from tqdm import tqdm


def tfr_morlet(data, sfreq, freqs, kind='amplitude', n_cycles=3.,
               use_fft=False, decimate=1, average=False):
    """Calculate the time-frequency representation of a signal.

    Parameters
    ----------
    data : array, shape (n_signals, n_times) | (n_epochs, n_signals, n_times)
        The data to calculate the TFR.
    sfreq : float | int
        The sampling frequency of the data
    freqs : array, shape (n_frequencies)
        The frequencies to calculate for the TFR.
    kind : string, 'filter', 'amplitude'
        What kind of TFR to output. If "filter", then the output will be a
        band pass filter of the signal. If "amplitude", the output will be
        the amplitude at each frequency band.
    n_cycles : int | array, shape (n_frequencies)
        The number of cycles for each frequency to include.
    use_fft : bool
        Whether to use an FFT to calculate the wavelet transform
    decimate : int
        The amount to decimate the output. If 1, no decimation will be done.
    average : bool
        Whether to average across the first dimension before returning results.
    """
    # Loop through our data
    if average is True:
        if data.ndim < 3:
            raise ValueError('If averaging, data should have at least 3 dims')
        n_ep, n_sig, n_times = data.shape[-3:]
        n_freqs = len(freqs)
        tfr = np.zeros([n_sig, n_freqs, int(np.round(n_times / decimate))])
    else:
        tfr = []
    for i_data in tqdm(data):
        # Calculate the wavelet transform for each iteration and stack
        i_data = np.atleast_2d(i_data)
        this_tfr = cwt_morlet(i_data, sfreq,
                              freqs, n_cycles=n_cycles,
                              use_fft=use_fft)
        if kind == 'filter':
            # In this case, we'll just take the real values
            this_tfr = np.real(this_tfr)

            if decimate != 1:
                warn('Decimating band-passed data may cause artifacts.')
        elif kind == 'amplitude':
            # Now take the absolute value so it's only amplitude
            this_tfr = np.abs(this_tfr)
        else:
            raise ValueError('kind must be one of "filter" | "amplitude"')
        this_tfr = this_tfr[..., ::decimate]

        if average is True:
            tfr += this_tfr
        else:
            tfr.append(this_tfr)

    if average is True:
        tfr /= n_ep
    else:
        tfr = np.asarray(tfr)
    return tfr.squeeze()
