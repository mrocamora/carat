# encoding: utf-8
# pylint: disable=C0103
"""
Tempo
=====

Tempo related functions
-----------------------

.. autosummary::
    :toctree: generated/

    compute_tempo_values

"""

import numpy as np

from scipy.signal import savgol_filter


__all__ = ['compute_tempo_values']


def compute_tempo_values(beat_times):
    """ Compute tempo values from beat time instants.

    Parameters
    ----------
    beat_times : np.ndarray
        time instants of the beats

    Returns
    -------
    bpms : np.ndarray
        tempo values as beats per minute (bpm)

    Examples
    --------

    """

    durs = beat_times[1:] - beat_times[:-1]
    bpms = np.round(60 / durs)

    return bpms


def smooth_tempo_curve(bpms, win_len=15, poly_ord=3):
    """ Smooth tempo curve using savgol filter from `scipy.signal`.

    Parameters
    ----------
    bpms : np.ndarray
        tempo values as beats per minute (bpm)
    window_length : int
        length of the filter window
    poly_order : int
        order of the polynomial used to fit the samples

    Returns
    -------
    bpms_smooth : np.ndarray
        smoothed tempo values as beats per minute (bpm)

    Examples
    --------
    
    TODO: implement other kinds of smoothing (move to utils)

    """
    
    bpms_smooth = savgol_filter(bpms, win_len, poly_ord)

    return bpms_smooth


