# encoding: utf-8
# pylint: disable=C0103
# pylint: disable=too-many-arguments
"""
Util
====

Signal segmentation
-------------------
.. autosummary::
    :toctree: generated/

    segmentSignal
    beat2signal
    get_time_segment

Time-frequency
------------------
.. autosummary::
    :toctree: generated/

    STFT
    fft2mel
    hz2mel
    mel2hz

Miscellaneous
-------------
.. autosummary::
    :toctree: generated/

    example_audio_file
    example_beats_file
    find_nearest
    deltas
"""


import warnings
import numpy as np
import scipy as sp
import scipy.signal
import scipy.fftpack as fft
import pkg_resources
#import scipy.signal
#import exceptions
from .exceptions import ParameterError

EXAMPLE_AUDIO1 = 'example_data/candombe/csic.1995_ansina1_01.wav'
EXAMPLE_AUDIO2 = 'example_data/samba/[0216] S2-TB2-03-SE.wav'
EXAMPLE_BEATS1 = 'example_data/candombe/csic.1995_ansina1_01.csv'
EXAMPLE_BEATS2 = 'example_data/samba/[0216] S2-TB2-03-SE.beats.txt'


__all__ = ['find_nearest', 'STFT', 'hz2mel', 'mel2hz', 'deltas']


def find_nearest(array, value):
    """Find index of the nearest value of an array to a given value

    Parameters
    ----------
    array (numpy.ndarray)  : array
    value (float)          : value

    Returns
    -------
    idx (int)              : index of nearest value in the array
    """

    idx = (np.abs(array-value)).argmin()

    return idx


def STFT(x, window_length, hop, windowing_function=sp.hanning, dft_length=None,
         zp_flag=False):
    """ Calculates the Short-Time Fourier Transform a signal.

    Given an input signal, it calculates the DFT of frames of the signal and stores them
    in bi-dimensional Scipy array.

    **Args**:
        window_len (float): length of the window in seconds (must be positive).
        window (callable): a callable object that receives the window length in samples and
        returns a numpy array containing the windowing function samples.
        hop (float): frame hop between adjacent frames in seconds.
        final_time (positive integer): time (in seconds) up to which the spectrogram is calculated.
        zp_flag (bool): a flag indicating if the *Zero-Phase Windowing* should be performed.

    **Returns**:
	spec: numpy array
	time: numpy array
	frequency: numpy array

    **Raises**:

    """
    # Checking input:
    if x.ndim != 1:
        raise AttributeError("Data must be one-dimensional.")
    # Window length must be odd:
    if window_length%2 == 0:
        window_length = window_length + 1
    # DFT length is equal the window_len+1 (always even)
    if dft_length is None:
        dft_length = window_length + 1
    # If dft_length was set by the user, it should always be larger than the window length.
    if dft_length < window_length:
        warnings.warn("DFT length is smaller than window length.", RuntimeWarning)
    # Partitioning the input signal:
    part_sig = segmentSignal(x, window_length, hop)
    no_cols = part_sig.shape[1]
    # Applying the window:
    window = windowing_function(window_length)
    win_sig = part_sig * sp.transpose(sp.tile(window, (no_cols, 1)))
    # Zero-phase windowing:
    if zp_flag:
        win_sig = fft.fftshift(win_sig, axes=0)
    # Taking the FFT of the partitioned signal
    spec = fft.fftshift(fft.fft(win_sig, n=dft_length, axis=0), axes=0)
    # Normalizing energy
    spec /= sp.sum(window)
    # Calculating time and frequency indices for the data
    frequency = fft.fftshift(fft.fftfreq(dft_length))
    time = sp.arange(no_cols)*float(hop) + ((window_length-1)/2)
    # Creating output spectrogram
    return spec, time, frequency


def segmentSignal(signal, window_len, hop):
    """ Segmentation of an array-like input:

    Given an array-like, this function calculates the DFT of frames of the signal and stores them
    in bi-dimensional Scipy array.


    **Args**:
        signal (array-like): object to be windowed. Must be a one-dimensional array-like object.
        window_len (int): window size in samples.
        hop (int): frame hop between adjacent frames in seconds.

    **Returns**:
        A 2-D numpy array containing the windowed signal. Each element of this array X
        can be defined as:

        X[m,n] = x[n+Hm]

        where, H is the HOP in samples, 0<=n<=N, N = window_len, and 0<m<floor(((len(x)-N)/H)+1).

    **Raises**:
        AttributeError if signal is not one-dimensional.
        ValueError if window_len or hop  are not strictly positives.
    """
    if(window_len <= 0 or hop <= 0):
        raise ValueError("window_len and hop values must be strictly positive numbers.")
    if signal.ndim != 1:
        raise AttributeError("Input signal must be one dimensional.")
    # Calculating the number of columns:
    no_cols = int(sp.floor((sp.size(signal)-window_len)/float(hop))+1)
    # Windowing indices (which element goes to which position in the windowed matrix).
    ind_col = sp.tile(sp.arange(window_len, dtype=np.uint64), (no_cols, 1))
    ind_line = sp.tile(sp.arange(no_cols, dtype=np.uint64)*hop, (window_len, 1))
    ind = sp.transpose(ind_col) + ind_line
    # Partitioned signal:
    part_sig = signal[ind].copy()
    # Windowing partitioned signal
    return part_sig


def __get_segment(y, idx_ini, idx_end):
    """ Get a segment of an array, given by initial and ending indexes.

    **Args**:
        y (numpy array): Must be a one-dimensional array.
        idx_ini (int): initial index.
        idx_end (int): ending index.

    **Returns**:
        segment (numpy array): segment of the signal.

    **Raises**:
        AttributeError if y is not a one-dimensional numpy array.
        ValueError if idx_ini or idx_end fall outside the signal bounds.
        ValueError if idx_ini >= idx_end.
    """

    if not isinstance(y, np.ndarray):
        raise ParameterError('Input must be of type numpy.ndarray, '
                             'given type(y)={}'.format(type(y)))
    if y.ndim != 1:
        raise ParameterError('Input must be one-dimensional, '
                             'given y.ndim={}'.format(y.ndim))
    if idx_ini >= idx_end:
        raise ValueError('Ending index is smaller than initial index.'
                         ' idx_ini={:d}, idx_end={:d}'.format(idx_ini, idx_end))
    if idx_ini < 0 or idx_end >= y.size:
        raise ValueError('Index out of signal bounds.'
                         ' y.size={:d}, idx_ini={:d},'
                         ' idx_end={:d}'.format(y.size, idx_ini, idx_end))

    y_segment = y[idx_ini:idx_end]

    return y_segment


def get_time_segment(y, time, time_ini, time_end):
    """ Get a segment of an array, given by initial and ending indexes.

    **Args**:
        y (numpy array): signal array. Must be a one-dimensional array.
        time (numpy array): corresponding time. Must be a one-dimensional array.
        time_ini (int): initial time value.
        time_end (int): ending time value.

    **Returns**:
        segment (numpy array): segment of the signal.

    **Raises**:
        AttributeError if y or time is not a one-dimensional numpy array.
        ValueError if idx_ini or idx_end fall outside the signal bounds.
        ValueError if idx_ini >= idx_end.
    """

    if not isinstance(y, np.ndarray):
        raise ParameterError('Input must be of type numpy.ndarray, '
                             'given type(y)={}'.format(type(y)))
    if not isinstance(time, np.ndarray):
        raise ParameterError('Input must be of type numpy.ndarray, '
                             'given type(time)={}'.format(type(time)))
    if y.ndim != 1:
        raise ParameterError('Input must be one-dimensional, '
                             'given y.ndim={}'.format(y.ndim))
    if time.ndim != 1:
        raise ParameterError('Input must be one-dimensional, '
                             'given time.ndim={}'.format(time.ndim))
    if time.size != y.size:
        raise ParameterError('Input y and time must be of the same size, '
                             'time.size={:d}, y.size={:d}'.format(time.size, y.size))
    if time_ini >= time_end:
        raise ValueError('Ending time is smaller than initial time.'
                         ' idx_ini={:.3f}, idx_end={:.3f}'.format(time_ini, time_end))

    if time_ini < time[0] or time_end > time[-1]:
        raise ValueError('Time valures are out of signal bounds.'
                         ' time[0]={:.3f}, time[-1]={:.3f}, time_ini={:.3f},\
                         time_end={:.3f},'.format(time[0], time[-1], time_ini, time_end))

    y_segment = y[(time >= time_ini) & (time <= time_end)]

    return y_segment


def beat2signal(y, time, beats, ind_beat):
    """ Get the signal fragment corresponding to a beat given by index ind_beat.
        If instead of beats, downbeats are used, then a bar is returned.

    **Args**:
        y (numpy array): signal array. Must be a one-dimensional array.
        time (numpy array): corresponding time. Must be a one-dimensional array.
        beats (numpy array): time instants of the beats.
        ind_beat (int): index of the desired beat.

    **Returns**:
        beat_segment (numpy array): segment of the signal corresponding to the beat.

    **Raises**:
        AttributeError if y or time is not a one-dimensional numpy array.
        ValueError if ind_beat fall outside the beats bounds.
    """

    if not isinstance(y, np.ndarray):
        raise ParameterError('Input must be of type numpy.ndarray, '
                             'given type(y)={}'.format(type(y)))
    if not isinstance(time, np.ndarray):
        raise ParameterError('Input must be of type numpy.ndarray, '
                             'given type(time)={}'.format(type(time)))
    if not isinstance(beats, np.ndarray):
        raise ParameterError('Input must be of type numpy.ndarray, '
                             'given type(beats)={}'.format(type(beats)))
    if y.ndim != 1:
        raise ParameterError('Input must be one-dimensional, '
                             'given y.ndim={}'.format(y.ndim))
    if time.ndim != 1:
        raise ParameterError('Input must be one-dimensional, '
                             'given time.ndim={}'.format(time.ndim))
    if beats.ndim != 1:
        raise ParameterError('Input must be one-dimensional, '
                             'given beats.ndim={}'.format(beats.ndim))
    if time.size != y.size:
        raise ParameterError('Input y and time must be of the same size, '
                             'time.size={:d}, y.size={:d}'.format(time.size, y.size))
    if ind_beat >= beats.size or ind_beat < 0:
        raise ValueError('Index out of bounds.'
                         ' ind_beat={:d}, beats.size={:d}'.format(ind_beat, beats.size))

    time_ini = beats[ind_beat]
    time_end = beats[ind_beat+1]

    beat_segment = get_time_segment(y, time, time_ini, time_end)

    return beat_segment


def fft2mel(freq, nfilts, minfreq, maxfreq):
    """ This method returns a 2-D Numpy array of weights that map a linearly spaced spectrogram
    to the Mel scale.

    **Args**:
        freq (1-D Numpy array): frequency of the components of the DFT.
        nfilts (): number of output bands.
        minfreq (): frequency of the first MEL coefficient.
        maxfreq (): frequency of the last MEL coefficient.

    **Returns**:
        The center frequencies in Hz of the Mel bands.

        """
    minmel = hz2mel(minfreq)
    maxmel = hz2mel(maxfreq)
    binfrqs = mel2hz(minmel+sp.arange(nfilts+2)/(float(nfilts)+1)*(maxmel-minmel))
    wts = sp.zeros((nfilts, (freq.size)))
    for i in range(nfilts):
        slp = binfrqs[i + sp.arange(3)]
        loslope = (freq - slp[0])/(slp[1] - slp[0])
        hislope = (slp[2] - freq)/(slp[2] - slp[1])
        wts[i, :] = sp.maximum(0.0, sp.minimum(loslope, hislope))
    wts[:, freq < 0] = 0
    wts = sp.dot(sp.diag(2./(binfrqs[2+sp.arange(nfilts)]-binfrqs[sp.arange(nfilts)])), wts)
    binfrqs = binfrqs[1:nfilts+1]

    return wts, binfrqs


def hz2mel(f_hz):
    """ Converts a given frequency in Hz to the Mel scale.

    **Args**:
        f_hz (Numpy array): Array containing the frequencies in HZ that should be converted.

    **Returns**:
        A Numpy array (of same shape as f_zh) containing the converted frequencies.

    """
    f_0 = 0
    f_sp = 200.0/3.0 # Log step
    brkfrq = 1000.0 # Frequency above which the distribution stays linear.
    brkpt = (brkfrq - f_0)/f_sp # First Mel value for linear region.
    logstep = sp.exp(sp.log(6.4)/27) # Step in the log region
    z_mel = sp.where(f_hz < brkfrq, (f_hz - f_0)/f_sp, brkpt +
                     (sp.log(f_hz/brkfrq))/sp.log(logstep))
    return z_mel


def mel2hz(z_mel):
    """ Converts a given frequency in the Mel scale to Hz scale.

    **Args**:
        z_mel (Numpy array): Array of frequencies in the Mel scale that should be converted.

    **Returns**:
        A Numpy array (of same shape as z_mel) containing the converted frequencies.
    """
    f_0 = 0
    f_sp = 200.0/3.0
    brkfrq = 1000.0
    brkpt = (brkfrq - f_0)/f_sp
    logstep = sp.exp(sp.log(6.4)/27)
    f_hz = sp.where(z_mel < brkpt, f_0 + f_sp*z_mel, brkfrq*sp.exp(sp.log(logstep)*(z_mel-brkpt)))
    return f_hz


def deltas(x, w=3):
    """ this function estimates the derivative of x
    """
    if x.ndim == 1:
        y = x.reshape((-1, 1)).T
    else:
        y = x
    if not w%2:
        w -= 1
    hlen = int(sp.floor(w/2))
    if w == 1: # first-order difference:
        win = sp.r_[1, -1]
    else:
        win = sp.r_[hlen:-hlen-1:-1]
    # Extending the input data (avoid border problems)
    extended_x = sp.c_[sp.repeat(y[:, 0].reshape((-1, 1)), hlen, axis=1), y,
                       sp.repeat(y[:, -1].reshape((-1, 1)), hlen, axis=1)]
    d = scipy.signal.lfilter(win, 1, extended_x, axis=1)
    d = d[:, 2*hlen:]
    if x.ndim == 1:
        d = d[0, :]
    return d


def getValidKeywords(kw, func):
    """ This function returns a dictionary containing the keywords arguments in initial_kw
        that are valid for function func.
    """
    import inspect
    valid_kw = {}
    invalid_kw = kw.copy()
    # args, varargs, varkw, defaults = inspect.getargspec(func)
    # NOTE: inspect.getargspec was deprecated, and inspect.getfullargspec is suggested as the
    # standard interface for single-source Python 2/3 code migrating away from legacy getargspec()
    args = inspect.getfullargspec(func)
    #for k, v in kw.iteritems():
    for k, v in kw.items():
        if k in args.args:
            valid_kw[k] = v
            del invalid_kw[k]
    return valid_kw, invalid_kw


def example_audio_file(num_file=None):
    '''Get the path to an included audio example file.

    Examples
    --------
    >>> # Load the waveform from the default example track
    >>> y, sr = carat.audio.load(carat.util.example_audio_file())

    >>> # Load 10 seconds of the waveform from the example track number 1
    >>> y, sr = carat.audio.load(carat.util.example_audio_file(num_file=1), duration=10.0))

    >>> # Load the waveform from the example track number 2
    >>> y, sr = carat.audio.load(carat.util.example_audio_file(num_file=2))

    Parameters
    ----------
    num_file : int
        Number to select among the example files available.

    Returns
    -------
    filename : str
        Path to the audio example file included with `carat`.
    '''

    if num_file == 1:
        EXAMPLE_AUDIO = EXAMPLE_AUDIO1
    elif num_file == 2:
        EXAMPLE_AUDIO = EXAMPLE_AUDIO2
    else:
        EXAMPLE_AUDIO = EXAMPLE_AUDIO2

    return pkg_resources.resource_filename(__name__, EXAMPLE_AUDIO)


def example_beats_file(num_file=None):
    '''Get the path to an included example file of beats annotations.

    Examples
    --------
    >>> # Load beats and downbeats from the example audio file number 1
    >>> beats, b_labs = carat.load_beats(carat.util.example_beats_file(num_file=1))
    >>> downbeats, d_labs = carat.load_downbeats(carat.util.example_beats_file(num_file=1))

    Parameters
    ----------
    num_file : int
        Number to select among the example files available.

    Returns
    -------
    filename : str
        Path to the beats annotations example file included with `carat`.
    '''

    if num_file == 1:
        EXAMPLE_BEATS = EXAMPLE_BEATS1
    elif num_file == 2:
        EXAMPLE_BEATS = EXAMPLE_BEATS2
    else:
        EXAMPLE_BEATS = EXAMPLE_BEATS2

    return pkg_resources.resource_filename(__name__, EXAMPLE_BEATS)
