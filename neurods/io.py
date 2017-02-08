import datascience as ds
import numpy as np
import pandas as pd
import shutil as sh
import os.path as op
import os
import nbformat
import sys
from glob import glob
from zipfile import ZipFile
from mne.utils import _fetch_file
from subprocess import check_output
from importlib import reload


path_data = '/home/shared/cogneuro-connector/data/'
data_list = {'eeg': path_data + 'eeg/',
             'ecog': path_data + 'ecog/',
             'fmri': path_data + 'fmri/'}


def mne_to_table(data):
    """Convert an MNE Raw object into a datascience table.

    Parameters
    ----------
    data : instance of MNE raw object.
        The data to be converted to a table.

    Returns
    -------
    table : instance of datascience Table.
        The data in table format.
    """
    df = pd.DataFrame(data._data.T, columns=data.ch_names)
    table = ds.Table().from_df(df)
    table['time'] = np.arange(df.shape[0]) / data.info['sfreq']
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
    elif 'github.com' in url:
        out = url.replace('github.com', 'raw.githubusercontent.com')
        out = out.replace('blob/', '')
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


def strip_answers(nbpath, strip_string='### STUDENT ANSWER',
                  output_suffix='student', keep_strip_string=True,
                  remove_cell=False, clean_outputs=True, save=True,
                  path_save=None):
    """Clean inputs / outputs of notebooks to send to students.

    Parameters
    ----------
    nbpath : str
        The path to a jupyter notebook file.
    strip_string : str
        A string to search for in input cells. If the string is
        found, then anything in the cell AFTER the string is removed.
    output_suffix : str
        An output to be appended to a saved jupyter notebook.
    save : bool
        Whether to save the output notebook.
    clean_outputs : bool
        Whether to clear outputs for the cells before saving.
    path_save : string
        An optional path to a folder where the output notebook will
        be saved. If None, save in the same directory as nbpath.

    Returns
    -------
    nb : instance of NotebookNode
        The NotebookNode corresponding to the cleaned notebook.
    """
    nb = nbformat.read(nbpath, 4)
    for ii, cell in enumerate(nb['cells']):
        # Only check code cells
        if cell['cell_type'] != 'code':
            continue

        # Replace some input cells
        ix = cell['source'].find(strip_string)
        if ix != -1:
            if remove_cell is True:
                # Remove the whole cell and move on
                _ = nb['cells'].pop(ii)
                continue
            newstr = cell['source'][:ix + len(strip_string)] + '\n'
            cell['source'] = newstr

        # Clean outputs
        if clean_outputs is True:
            cell['outputs'] = []

        # Clear prompt numbers
        cell['execution_count'] = None
        cell.pop('prompt_number', None)

    if save is True:
        filename = os.path.basename(nbpath)
        path_save = os.path.dirname(nbpath) if path_save is None else path_save
        name, ext = filename.split('.')
        outname = '{}_{}.{}'.format(name, output_suffix, ext)
        save_file = '{}{}{}'.format(path_save, os.sep, outname)
        print('Saving to {}'.format(save_file))
        if not os.path.isdir(path_save):
            os.makedirs(path_save)
        nbformat.write(nb, save_file)
    return nb


def update_neurods():
    """Use a shell command to update neurods."""
    s = ('pip install git+https://github.com/choldgraf/neurods.git@student --user '
         '--upgrade')
    s = check_output(s.split(' '))
    s = s.decode('utf-8')
    print(s)
