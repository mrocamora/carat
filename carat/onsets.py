# encoding: utf-8
# pylint: disable=C0103
"""
Onsets
======

Onsets detection
----------------

.. autosummary::
    :toctree: generated/

    detection

"""

from . import util
from . import features

__all__ = ['detection']


def detection(signal, fs=22050, **kwargs):
    """Onset detection from audio signal. 

    Parameters
    ----------
    signal : np.array
        input audio signal
    fs : int
        sampling rate
    **kwargs :  (check)
        keyword arguments passed down to each of the functions used

    Returns
    -------
    onset_times : np.ndarray
        time instants of the onsets
    feature_values: np.ndarray
        feature values at the onsets

    """

    # valid keywords for peak_detection (other ones are passed to accentuation feature)
    peaks_kw, acce_kw = util.getValidKeywords(kwargs, features.peak_detection)

    # accentuation feature computation
    acce, times, _ = features.accentuation_feature(signal, fs, **acce_kw)

    # peak picking in the feature function
    ons_indx, _, _ = features.peak_detection(acce, **peaks_kw)
    
    # time instants of the onsets
    onset_times = times[ons_indx]
    
    # feature values at the onsets
    feature_values = acce[ons_indx]

    return onset_times, feature_values
