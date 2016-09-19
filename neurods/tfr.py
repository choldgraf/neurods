from mne.time_frequency import cwt_morlet
from mne.utils import warn
import numpy as np
from tqdm import tqdm
import warnings


def tfr_morlet(data, sfreq, freqs, kind='amplitude', n_cycles=3.,
               use_fft=True, decimate=1, average=False):
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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


def extract_amplitude(inst, freqs, n_cycles=7, normalize=False):
    """Extract the time-varying amplitude for a frequency band.

    If multiple freqs are given, the amplitude is calculated at each frequency
    and then averaged across frequencies.

    Parameters
    ----------
    inst : instance of Raw
        The data to have amplitude extracted
    freqs : array of ints/floats, shape (n_freqs)
        The frequencies to use. If multiple frequency bands given, amplitude
        will be extracted at each and then averaged between frequencies. The
        structure of each band is fmin, fmax.
    n_cycles : int
        The number of cycles to include in the filter length for the wavelet.
    normalize : bool
        Normalize the power of each band by its mean before combining.

    Returns
    -------
    inst : mne instance, same type as input 'inst'
        The MNE instance with channels replaced with their time-varying
        amplitude for the supplied frequency range.
    """

    # Data checks
    freqs = np.atleast_1d(freqs)
    picks = range(len(inst.ch_names))
    if inst.preload is False:
        raise ValueError('Data must be preloaded.')

    # Filter for HFB and extract amplitude
    bands = np.zeros([len(picks), inst.n_times])
    for ifreq in tqdm(freqs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Extract power for this frequency + modify in place to save mem
            band = np.abs(cwt_morlet(inst._data, inst.info['sfreq'], [ifreq]))
        band = band[:, 0, :]

        if normalize is True:
            # Scale frequency band so that the ratios of all are the same
            band /= band.mean()
        bands += np.abs(band)

    # Convert into an average
    bands /= len(freqs)
    inst = inst.copy()
    inst._data = bands
    return inst
