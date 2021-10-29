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
import scipy.stats as sp
from . import util

__all__ = ['beats_from_onsets', 'normalize_onsets', 'define_metrical_grid',
           'onsets_to_metrical_grid', 'onsets_to_normal_dist']
#, 'downbeats_from_onsets'] #, 'onsets_to_metrical_grid']

def beats_from_onsets(beat_annotations, onsets, tolerance=0.125, method='mean'):
    """Estimation of the location of beats from the onsets.

    Parameters
    ----------
    beat_annotations : np.ndarray
        location of annotated beats (in seconds)
    onsets : np.ndarray
        onsets (in seconds)
    tolerance : float
        tolerance value for an onset to be close enough to the annotated beat
        (as a percentage of the beat duration)
    method : str
        method used to combine onsets from different instruments ('mean', 'median')

    Returns
    -------
    beats : np.ndarray
        location of beats estimated from onsets

    Notes
    -----
    Based on the 'democratic' estimation of the beginning of each rhythm cycle introduced in [1].
    This is the method applied in the study of entrainment in candombe music presented in [2].

    References
    ----------
    .. [1] Polak, London, Jacobi
           "Both Isochronous and Non-Isochronous Metrical Subdivision Afford Precise and Stable
           Ensemble Entrainment: A Corpus Study of Malian Jembe Drumming"
           Frontiers in Neuroscience. 10:285. 2016. doi: 10.3389/fnins.2016.00285

    .. [2] Rocamora, Jure, Polak, Jacobi
           "Interpersonal music entrainment in Afro-Uruguayan candombe drumming"
           44th International Council for Traditional Music (ICTM) World Conference,
           Limerick, Irleand. 2017.

    """

    # number of beats to estimate
    num_beats = beat_annotations.shape[0]

    # check if instead of a list, onsets is an array (for a single instrument)
    if isinstance(onsets, np.ndarray):
        # create a list with a single array
        onsets = [onsets]
    # number of ensemble parts (for recordings with multiple instruments)
    num_parts = len(onsets)

    # onsets at beats (initialize with nan)
    onsets_at_beats = np.zeros((num_parts, num_beats))
    onsets_at_beats.fill(np.nan)

    # for each instrument of the ensemble
    for ind_ins in range(num_parts):

        # onset for current instrument
        onsets_ins = onsets[ind_ins]

        # for each annotated beat
        for ind_beat in range(num_beats):

            # current beat
            beat = beat_annotations[ind_beat]
            # find the index of the closest onsets to current annotated beat
            ind_onset = util.find_nearest(onsets_ins, beat)

            # compute beat duration in seconds
            if ind_beat < num_beats-1:
                beat_dur = beat_annotations[ind_beat+1] - beat_annotations[ind_beat]
            else:
                # last downbeat is treated differently
                beat_dur = beat_annotations[ind_beat] - beat_annotations[ind_beat-1]

            # check if onset is close enough
            if (abs(beat-onsets_ins[ind_onset]) / beat_dur) < tolerance:
                # save onset at beat
                onsets_at_beats[ind_ins, ind_beat] = onsets_ins[ind_onset]

    # estimate location of beats combining the onsets closest to annotated beat
    if method == 'mean':
        # mean of the onsets, i.e. 'democratic' vote
        if onsets_at_beats.shape[0] == 1: # a single drum
            beats = onsets_at_beats[~(np.isnan(onsets_at_beats))]
        else: # several drums
            beats = np.nanmean(onsets_at_beats, axis=0)

    elif method == 'median':
        # median of the onsets, i.e. one of the onsets
        if onsets_at_beats.shape[0] == 1: # a single drum
            beats = onsets_at_beats[~(np.isnan(onsets_at_beats))]
        else: # several drums
            beats = np.nanmedian(onsets_at_beats, axis=0) 

    else:
        raise AttributeError("Method not implemented.")

    # check if there are nan values (this raises a RuntimeWarning)

    return beats


def normalize_onsets(beats, onsets, tolerance=0.125):
    """Normalize de location of onsets to the relative position between two adjacent beats.


    Parameters
    ----------
    beats : np.ndarray
        location of beats (in seconds)
    onsets : list of np.ndarray
        the elements of the list are arrays of onsets (in seconds), one for each instrument
    tolerance : float
        tolerance value for an onset to lay in a beat interval (relative to the beat duration)

    Returns
    -------
    onsets_normalized : list of np.ndarray
        normalized onsets (in seconds) as a list of arrays, one element for each instrument
    """

    # number of beats (last beat is the end)
    num_beats = beats.shape[0] - 1

    # check if instead of a list, onsets is an array (for a single instrument)
    if isinstance(onsets, np.ndarray):
        # create a list with a single array
        onsets = [onsets]
    # number of ensemble parts (for recordings with multiple instruments)
    num_parts = len(onsets)

    # normalized onsets
    onsets_normalized = [None]*num_parts

    # for each instrument of the ensemble
    for ind_ins in range(num_parts):

        # onset for current instrument
        onsets_ins = onsets[ind_ins]

        # list to save normalized onset
        onsets_ins_norm = [None]*num_beats

        # for each beat
        for ind_beat in range(num_beats):

            # compute beat duration in seconds
            beat_dur = beats[ind_beat+1] - beats[ind_beat]

            # initial and ending time of beat interval
            ini_time = beats[ind_beat] - beat_dur * tolerance
            end_time = beats[ind_beat+1] - beat_dur * tolerance

            # onsets within the beat interval
            onsets_beat = onsets_ins[(onsets_ins >= ini_time) & (onsets_ins < end_time)]

            # normalized onsets within the beat interval
            onsets_norm = (onsets_beat - beats[ind_beat]) / beat_dur

            # save onsets
            onsets_ins_norm[ind_beat] = onsets_norm

        # save all onsets of current instrument
        onsets_normalized[ind_ins] = onsets_ins_norm


    return onsets_normalized


def define_metrical_grid(num_subdivs=4, subdivs_type='isochronous'):
                         #, num_beats=4, beats_type='isochronous',
                         # beats_profile=None, subdivs_profile=None, onsets=None):
    """Definition of the metrical grid to be used for rhythm analysis.


    THIS IS JUST PRELIMINAR

    Parameters
    ----------
    num_subdivs : int
        number of subdivisions of the beat
    subdivs_type : str
        type of subdivisions of the beat

    Returns
    -------
    metrical_grid : np.ndarray
        metrical grid represented as values for beats and subdivisions
    """

    metrical_grid = []

    if subdivs_type == 'isochronous':
        # compute grid for isochronous subdivisions of the beat
        metrical_grid = grid_isochronous_beat(num_subdivs)

    return metrical_grid


def grid_isochronous_beat(num_subdivs=4):
    """Returns an isochronous grid for beat subdivisions

    Parameters
    ----------
    num_subdivs : int
        number of subdivisions of the beat

    Returns
    -------
    grid_vals : np.ndarray
        grid values for the subdivisions of the beat

    """

    # grid values
    grid_vals = np.linspace(0, 1, num=num_subdivs+1)
    grid_vals = grid_vals[:-1]


    return grid_vals


def onsets_to_metrical_grid(onsets, metrical_grid, tolerance=0.125):
    """Assign each onset to a position in the metrical grid.

    parameters
    ----------
    onsets : list of np.ndarray
        the elements of the list are arrays of onsets (normalized), one for each instrument
    metrical_grid : np.ndarray
        metrical grid represented as values for beats and subdivisions
    tolerance : float
        tolerance value for an onset to lay in a beat interval (relative to the beat duration)

    returns
    -------
    onsets_in_grid : list of np.ndarray
        onsets assigned to metrical grid as a list of arrays, one element for each instrument

    """


    # check if instead of a list, onsets is an array (for a single instrument)
    if isinstance(onsets, np.ndarray):
        # create a list with a single array
        onsets = [onsets]
    # number of ensemble parts (for recordings with multiple instruments)
    num_parts = len(onsets)

    # number of beats
    num_beats = len(onsets[0])

    # normalized onsets
    onsets_in_grid = [None]*num_parts

    # for each instrument of the ensemble
    for ind_ins in range(num_parts):

        # onset for current instrument
        onsets_ins = onsets[ind_ins]

        # list to save onset assigned to grid
        ons_ins_in_grid = [None]*num_beats

        # for each beat
        for ind_beat in range(num_beats):

            # onsets within the beat interval
            onsets_beat = onsets_ins[ind_beat]

            # onsets assigned to metrical grid (initialize with nan)
            onsets_grid = np.zeros(metrical_grid.shape)
            onsets_grid.fill(np.nan)

            # for each onset in the beat
            for onset in onsets_beat:
                # closest grid position
                ind_grid = util.find_nearest(metrical_grid, onset)
                # distance from onset to metrical position
                dist = abs(onset - metrical_grid[ind_grid])
                # check if onset is close enough
                if dist < tolerance:
                    # check if there is already an onset in the grid position
                    if np.isnan(onsets_grid[ind_grid]):
                        onsets_grid[ind_grid] = onset
                    else:
                        # if there is already an onset assign the closest to the grid position
                        if dist < abs(onsets_grid[ind_grid] - metrical_grid[ind_grid]):
                            onsets_grid[ind_grid] = onset

            # save onsets in grid
            ons_ins_in_grid[ind_beat] = onsets_grid

        # save all onsets of current instrument
        onsets_in_grid[ind_ins] = ons_ins_in_grid


    return onsets_in_grid



def onsets_to_normal_dist(onsets_in_grid):
    """Fit a normal distribution to each subdivision

    parameters
    ----------
    onsets_in_grid : list of np.ndarray
        onsets assigned to metrical grid as a list of arrays, one element for each instrument

    returns
    -------
    mus : list of floats
        list of mean values for the onsets in each subdivision
    stds : list of floats
        list of standard deviation values for the onsets in each subdivision

    """

    # number of subdivisions
    num_subdivs = onsets_in_grid[0].shape[0]

    # mean value for each subdivision
    mus = [None]*num_subdivs
    # std value for each subdivision
    stds = [None]*num_subdivs

    for ind in range(num_subdivs):
        # onsets in current subdivision
        onsets = []

        # get the onsets at current subdivision
        for ons in onsets_in_grid:
            # check if onset is not nan
            if not np.isnan(ons[ind]):
                # save onset in list
                onsets.append(ons[ind])

        # fit a normal distribution
        mu, std = sp.norm.fit(onsets)
        # save mu and std
        mus[ind] = mu
        stds[ind] = std


    return mus, stds
