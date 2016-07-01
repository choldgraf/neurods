import datascience as ds
import numpy as np


def mne_to_table(data):
    """Convert an MNE Raw object into a datascience table."""
    data_values = []
    for i_ch, i_data in zip(data.ch_names, data._data):
        data_values.append((i_ch, i_data))
    table = ds.Table().with_columns(data_values)
    table['time'] = np.arange(data._data.shape[-1]) / data.info['sfreq']
    return table
