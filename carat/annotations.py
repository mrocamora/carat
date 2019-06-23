# encoding: utf-8
# pylint: disable=C0103
"""Utility functions to deal with annotations."""

import csv
import numpy as np

__all__ = ['load_beats', 'load_downbeats']

def load_beats(labels_file, delimiter=',', times_col=0, labels_col=1):
    """Load annotated beats from text (csv) file.

    Parameters
    ----------
    labels_file (str) : name (including path) of the input file
    times_col (int) : column index of the time data
    labels_col (int) : column index of the label data

    Returns
    -------
    beat_times  (np.ndarray) : time instants of the beats
    beat_labels (list)       : labels at the beats (e.g. 1.1, 1.2, etc)

    Examples
    --------
    >>> # load an example file from the candombe dataset
    >>> # (http://www.eumus.edu.uy/candombe/datasets/ISMIR2015/)
    >>> beats, beat_labs = annotations.load_beats('candombe_beats.csv')
    >>> beats[0]
    0.548571428
    >>> beat_labs[0]
    '1.1'

    >>> # load an example file from the samba dataset
    >>> beats, beat_labs = annotations.load_beats('./material/data/samba/[0216]
                                                  S2-TB2-03-SE.beats.txt', delimiter=' ')
    >>> beats
    array([ 2.088,  2.559,  3.012,  3.48 ,  3.933,  4.41 ,  4.867,  5.32 ,
            5.771,  6.229,  6.69 ,  7.167,  7.633,  8.092,  8.545,  9.01 ,
            9.48 ,  9.943, 10.404, 10.865, 11.322, 11.79 , 12.251, 12.714,
           13.167, 13.624, 14.094, 14.559, 15.014, 15.473, 15.931, 16.4  ,
           16.865, 17.331, 17.788, 18.249, 18.706, 19.167, 19.643, 20.096,
           20.557, 21.018, 21.494, 21.945, 22.408, 22.869, 23.31 , 23.773,
           24.235, 24.692, 25.151, 25.608, 26.063, 26.52 ])
    >>> beat_labs
    ['1', '2', '1', '2', '1', '2', '1', '2', '1', '2', '1', '2', '1', '2',
     '1', '2', '1', '2', '1', '2', '1', '2', '1', '2', '1', '2', '1', '2',
     '1', '2', '1', '2', '1', '2', '1', '2', '1', '2', '1', '2', '1', '2',
     '1', '2', '1', '2', '1', '2', '1', '2', '1', '2', '1', '2']


    Notes
   -----
    It is assumed that the beat annotations are provided as a text file (csv).
    Apart from the time data (mandatory) a label can be given for each beat (optional).
    The time data is assumed to be given in seconds.
    The labels may indicate the beat number within the rhythm cycle (e.g. 1.1, 1.2, or 1, 2).
    """

    # read beat time instants
    beat_times = np.genfromtxt(labels_file, delimiter=delimiter, usecols=(times_col))

    # read beat labels
    with open(labels_file, 'r') as fi:
        reader = csv.reader(fi, delimiter=delimiter)
        # number of columns
        ncol = len(next(reader))
        # check if there are no labels
        if ncol == 1:
            beat_labels = []
        else:
            fi.seek(0)
            beat_labels = [row[labels_col] for row in reader]

    return beat_times, beat_labels


def load_downbeats(labels_file, delimiter=',', times_col=0, labels_col=1, downbeat_label='.1'):
    """Load annotated downbeats from text (csv) file.


    Parameters
    ----------
    labels_file (str) : name (including path) of the input file
    times_col (int) : column index of the time data
    labels_col (int) : column index of the label data
    downbeat_label (str) : string to look for in the label data to select downbeats

    Returns
    -------
    downbeat_times  (np.ndarray) : time instants of the downbeats
    downbeat_labels (list)       : labels at the downbeats

    Examples
    --------
    >>> # Load an example labels file
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> beats, beat_labs = annotations.load_beats('candombe_beats.csv')
    >>> librosa.get_duration(y=y, sr=sr)
    61.45886621315193


    >>> # load an example file from the samba dataset
    >>> beats, beat_labs = annotations.load_beats('./material/data/samba/[0216]
                                                  S2-TB2-03-SE.beats.txt', delimiter=' ')
    >>> beats
    array([ 2.088,  2.559,  3.012,  3.48 ,  3.933,  4.41 ,  4.867,  5.32 ,
            5.771,  6.229,  6.69 ,  7.167,  7.633,  8.092,  8.545,  9.01 ,
            9.48 ,  9.943, 10.404, 10.865, 11.322, 11.79 , 12.251, 12.714,
           13.167, 13.624, 14.094, 14.559, 15.014, 15.473, 15.931, 16.4  ,
           16.865, 17.331, 17.788, 18.249, 18.706, 19.167, 19.643, 20.096,
           20.557, 21.018, 21.494, 21.945, 22.408, 22.869, 23.31 , 23.773,
           24.235, 24.692, 25.151, 25.608, 26.063, 26.52 ])
    >>> beat_labs
    ['1', '2', '1', '2', '1', '2', '1', '2', '1', '2', '1', '2', '1', '2',
     '1', '2', '1', '2', '1', '2', '1', '2', '1', '2', '1', '2', '1', '2',
     '1', '2', '1', '2', '1', '2', '1', '2', '1', '2', '1', '2', '1', '2',
     '1', '2', '1', '2', '1', '2', '1', '2', '1', '2', '1', '2']

    >>> downbeats, downbeat_labs = annotations.load_downbeats('./material/data/samba/[0216]
                                                              S2-TB2-03-SE.beats.txt', delimiter='
                                                              ', downbeat_label='1')
    >>> downbeats
    array([ 2.088,  3.012,  3.933,  4.867,  5.771,  6.69 ,  7.633,  8.545,
            9.48 , 10.404, 11.322, 12.251, 13.167, 14.094, 15.014, 15.931,
           16.865, 17.788, 18.706, 19.643, 20.557, 21.494, 22.408, 23.31 ,
           24.235, 25.151, 26.063])
    >>> downbeat_labs
    ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1',
     '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1']


    Notes
    -----
    It is assumed that the annotations are provided as a text file (csv).
    Apart from the time data (mandatory) a label can be given for each downbeat (optional).
    The time data is assumed to be given in seconds.

    If a single file contains both beats and downbeats then the downbeat_label is used to select
    downbeats. The downbeats are those beats whose label has the given downbeat_label string. For
    instance the beat labels can be numbers, e.g. '1', '2'. Then, the downbeat_label is just '1'.
    This is the case for the BRID samba dataset. In the case of the candombe dataset, the beat
    labels indicate bar number and beat number. For instance, '1.1', '1.2', '1.3' and '1.4' are the
    four beats of the first bar. Hence, the string needed to indetify the downbeats is '.1'.
    """

    # read file as beats
    beat_times, beat_labs = load_beats(labels_file, delimiter=delimiter, times_col=times_col,
                                       labels_col=labels_col)

    # if there are no labels in the file or downbeat_label is None, then all entries are downbeats
    if not beat_labs or downbeat_label is None:
        downbeat_times, downbeat_labs = beat_times, beat_labs
    else:
        # get downbeat instants and labels by finding the string downbeat_label in the beat labels
        ind_downbeats = [ind_beat for ind_beat in range(len(beat_labs)) if downbeat_label in
                         beat_labs[ind_beat]]
        downbeat_times = beat_times[ind_downbeats]
        downbeat_labs = [beat_labs[ind_downbeat] for ind_downbeat in ind_downbeats]

    return downbeat_times, downbeat_labs
