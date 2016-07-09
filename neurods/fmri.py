"""Utilities for fMRI section of CogNeuro Connector for Data8"""

import numpy as np
import scipy.stats
import warnings
import numpy as np

def hrf(shape='twogamma', sf=1, pttp=5, nttp=15, pnr=6, ons=0, pdsp=1, ndsp=1, t=None, d=0):
    """Create canonical hemodynamic response filter
    
    TODO: fix this docstring to be pep8 / numpy or whatever
    
    Parameters
    ----------
    shape : 
        HRF general shape {'twogamma' [, 'boynton']}
    sf : 
        HRF sample frequency (default: 1s/16, OK: [1e-3 .. 5])
    pttp : 
        time to positive (response) peak (default: 5 secs)
    nttp : 
        time to negative (undershoot) peak (default: 15 secs)
    pnr : 
        pos-to-neg ratio (default: 6, OK: [1 .. Inf])
    ons : 
        onset of the HRF (default: 0 secs, OK: [-5 .. 5])
    pdsp : 
        dispersion of positive gamma PDF (default: 1)
    ndsp : 
        dispersion of negative gamma PDF (default: 1)
    t : 
        sampling range (default: [0, ons + 2 * (nttp + 1)])
    d : 
        derivatives (default: 0)
    
    Returns
    -------
    h : HRF function given within [0 .. onset + 2*nttp]
    t : HRF sample points
    
    Notes
    -----
    The pttp and nttp parameters are increased by 1 before given
    as parameters into the scipy.stats.gamma.pdf function (which is a property
    of the gamma PDF!)

    Converted to python from BVQXtools 
    Version:  v0.7f
    Build:    8110521
    Date:     Nov-05 2008, 9:00 PM CET
    Author:   Jochen Weber, SCAN Unit, Columbia University, NYC, NY, USA
    URL/Info: http://wiki.brainvoyager.com/BVQXtools
    """

    # Input checks
    if sf > 5:
        sf = 1/16
    elif sf < 0.001:
        sf = 0.001
    if not shape.lower() in ('twogamma', 'boynton'):
        warnings.warn('Shape can only be "twogamma" or "boynton"')
        shape = 'twogamma'
    if t is None:
        t = np.arange(0,(ons + 2 * (nttp + 1)), sf) - ons
    else:
        t = np.arange(np.min(t),np.max(t),sf) - ons;

    # computation (according to shape)
    h = np.zeros((len(t), d + 1));
    if shape.lower()=='boynton':
        # boynton (single-gamma) HRF
        h[:,0] = scipy.stats.gamma.pdf(t, pttp + 1, pdsp)
        if d > 0:
            raise NotImplementedError('Still WIP - code just needs porting.')
            """# Matlab code, partly translated:
            h[:, 1] = h[:, 1] - scipy.stats.gamma.pdf(t + 1, pttp + 1, pdsp);
            hi = find(h[:, 1] ~= 0);
            h[hi, 1] = h[hi, 1] - ((pinv(h[hi, 1]' * h[hi, 1]) * h[hi, 1]' * h[hi, 1])' * h[hi, 1]')';
            if d > 1:
                h[:,2] = h[:, 1] - scipy.stats.gamma.pdf(t, pttp + 1, pdsp / 1.01);
                hi = find(h[:,2] ~= 0);
                h[hi,2] = h[hi,2] - ((pinv(h[hi, [1, 2]).T * h[hi, [1, 2])) * h[hi, [1, 2]).T * h[hi,2]).T * h[hi, [1, 2]).T).T;"""
    elif shape.lower()=='twogamma':
        gpos = scipy.stats.gamma.pdf(t, pttp + 1, pdsp)
        gneg = scipy.stats.gamma.pdf(t, nttp + 1, ndsp) / pnr
        h[:,0] =  gpos-gneg             
        if d > 0:
            raise NotImplementedError('Still WIP. Sorting through matlab multiplications is annoying.')
            """gpos = scipy.stats.gamma.pdf(t-1, pttp + 1, pdsp)
            gneg = scipy.stats.gamma.pdf(t-1, nttp + 1, ndsp) / pnr
            h[:, 1] = h[:, 0] - gpos - gneg
            hi = h[:, 1] != 0
            h[hi, 1] = h[hi, 0] - ((np.linalg.pinv(h[hi, 0].T * h[hi, 0]) * h[hi, 0].T * h[hi, 0]).T * h[hi, 1].T).T
            if d > 1:
                h[:,2] = (h[:, 1] - (scipy.stats.gamma.pdf(t, (pttp + 1) / 1.01, pdsp / 1.01) - scipy.stats.gamma.pdf(t, nttp + 1, ndsp) / pnr)) / 0.01;
                hi = h[:,2] != 0
                h[hi,2] = h[hi,2] - ((pinv(h[hi, [1, 2]).T * h[hi, [1, 2])) * h[hi, [1, 2]).T * h[hi,2]).T * h[hi, [1, 2]).T).T;
            """
    # normalize for convolution
    if d < 1:
        h /= np.sum(h)
    else:
        h /= np.tile(np.sqrt(np.sum(h**2)), h.shape[0], 1)
    return t, h

def compute_event_avg(data, events, time_per_event):
    """Columns of `events` are markers for different types of events
    events are event onsets OR 
    data should be columns
    """
    # 1D data for now
    if events.dtype in (np.bool,):
        event_start = np.nonzero(events)[0]
    else:
        event_start = events
    event_stack = []
    for st, fin in zip(event_start, event_start+time_per_event):
        tmp = data[st:fin]
        if fin > len(data):
            tmp = np.hstack([tmp,np.zeros(fin-len(data))])
        event_stack.append(tmp)
    event_stack = np.nanmean(event_stack, axis=0)
    return event_stack