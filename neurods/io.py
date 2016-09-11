import datascience as ds
import numpy as np
import shutil as sh
import os.path as op
import os
from zipfile import ZipFile
from mne.utils import _fetch_file


path_data = '/home/shared/cogneuro-connector/data/'


def mne_to_table(data):
    """Convert an MNE Raw object into a datascience table."""
    data_values = []
    for i_ch, i_data in zip(data.ch_names, data._data):
        data_values.append((i_ch, i_data))
    table = ds.Table().with_columns(data_values)
    table['time'] = np.arange(data._data.shape[-1]) / data.info['sfreq']
    return table


def _convert_url_to_downloadable(url):
    """Convert a url to the proper style depending on its website."""

    if 'drive.google.com' in url:
        raise ValueError('Google drive links are not currently supported')
        # For future support of google drive
        file_id = url.split('d/').split('/')[0]
        base_url = 'https://drive.google.com/uc?export=download&id='
        out = '{}{}'.format(base_url, file_id)
    elif 'www.dropbox.com' in url:
        out = url.replace('www.dropbox.com', 'dl.dropboxusercontent.com')
    else:
        out = url
    return out


def download_file(url, name, root_destination='~/data/', zipfile=False,
                  replace=False):
    """Download a file from dropbox, google drive, or a URL.

    This will download a file and store it in a '~/data/` folder,
    creating directories if need be. It will also work for zip
    files, in which case it will unzip all of the files to the
    desired location.

    Parameters
    ----------
    url : string
        The url of the file to download. This may be a dropbox
        or google drive "share link", or a regular URL. If it
        is a share link, then it should point to a single file and
        not a folder. To download folders, zip them first.
    name : string
        The name / path of the file for the downloaded file, or
        the folder to zip the data into if the file is a zipfile.
    root_destination : string
        The root folder where data will be downloaded.
    zipfile : bool
        Whether the URL points to a zip file. If yes, it will be
        unzipped to root_destination + name.
    replace : bool
        If True and the URL points to a single file, overwrite the
        old file if possible.
    """
    # Make sure we have directories to dump files
    home = op.expanduser('~')
    tmpfile = home + '/tmp/tmp'
    if not op.isdir(home + '/data/'):
        print('Creating data folder...')
        os.makedirs(home + '/data/')

    if not op.isdir(home + '/tmp/'):
        print('Creating tmp folder...')
        os.makedirs(home + '/tmp/')

    download_path = _convert_url_to_downloadable(url)

    # Now save to the new destination
    out_path = root_destination.replace('~', home) + name
    if not op.isdir(op.dirname(out_path)):
        print('Creating path {} for output data'.format(out_path))
        os.makedirs(op.dirname(out_path))

    if zipfile is True:
        _fetch_file(download_path, tmpfile)
        myzip = ZipFile(tmpfile)
        myzip.extractall(out_path)
        os.remove(tmpfile)
    else:
        if len(name) == 0:
            raise ValueError('Cannot overwrite the root data directory')
        if replace is False and op.exists(out_path):
            raise ValueError('Path {} exists, use `replace=True` to '
                             'overwrite'.format(out_path))
        _fetch_file(download_path, out_path)
    print('Successfully moved file to {}'.format(out_path))
