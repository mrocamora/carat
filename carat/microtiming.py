# encoding: utf-8
# pylint: disable=C0103
# pylint: disable=too-many-arguments
"""
Microtiming
===========

Microriming analysis
--------------------
.. autosummary::
    :toctree: generated/

    beats_from_onsets
    downbeats_from_onsets
    onsets_to_metrical_grid
"""

import numpy as np
from . import util

__all__ = ['beats_from_onsets']#, 'downbeats_from_onsets'] #, 'onsets_to_metrical_grid']

def beats_from_onsets(beat_annotations, onsets, tolerance=0.125, method='mean'):
    """Estimation of the location of beats from the onsets.

    Based on the 'democratic' estimation of the beginning of each rhythm cycle introduced in [1].
    This is the method applied in the study of entrainment in candombe music presented in [2].

    [1] Polak, London, Jacobi
           "Both Isochronous and Non-Isochronous Metrical Subdivision Afford Precise and Stable
           Ensemble Entrainment: A Corpus Study of Malian Jembe Drumming"
           Frontiers in Neuroscience. 10:285. 2016. doi: 10.3389/fnins.2016.00285

    [2] Rocamora, Jure, Polak, Jacobi
           "Interpersonal music entrainment in Afro-Uruguayan candombe drumming"
           44th International Council for Traditional Music (ICTM) World Conference,
           Limerick, Irleand. 2017.


    Parameters
    ----------
    beat_annotations : np.ndarray
        location of annotated beats (in seconds)
    onsets : np.ndarray
        onsets (in seconds)
    method : str
        method used to combine onsets from different instruments ('mean', 'median')

    Returns
    -------
    beats : np.ndarray
        location of beats estimated from onsets
    """

    # number of beats to estimate
    num_beats = beat_annotations.shape[0]
    # number of ensemble parts (for recordings with multiple instruments)
    num_parts = 1

    # onsets at beats (initialize with nan)
    onsets_at_beats = np.zeros((num_parts, num_beats))
    onsets_at_beats.fill(np.nan)

    # for each instrument of the ensemble
    for ind_ins in range(num_parts):
        # for each annotated beat
        for ind_beat in range(num_beats):

            # current beat
            beat = beat_annotations[ind_beat]
            # find the index of the closest onsets to current annotated beat
            ind_onset = util.find_nearest(onsets, beat)

            # compute beat duration in seconds
            if ind_beat < num_beats-1:
                beat_dur = beat_annotations[ind_beat+1] - beat_annotations[ind_beat]
            else:
                beat_dur = beat_annotations[ind_beat] - beat_annotations[ind_beat-1]

            # check if onset is close enough
            if abs(beat-onsets[ind_onset]) / beat_dur < tolerance:
                # save onset at beat
                onsets_at_beats[ind_ins, ind_beat] = onsets[ind_onset]

    # estimate location of beats using onsets closest to annotated beat
    if method == 'mean':
        # mean of the onsets, i.e. 'democratic' vote
        beats = np.nanmean(onsets_at_beats, axis=0)

    elif method == 'median':
        # median of the onsets, i.e. one of the onsets
        beats = np.nanmedian(onsets_at_beats, axis=0)

    else:
        raise AttributeError(" method not implemented.")

    return beats
