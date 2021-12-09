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
import scipy.signal
import scipy.fftpack as fft
from scipy.stats import pearsonr
import pkg_resources
from .exceptions import ParameterError

import os
import json
from pathlib import Path
from pkg_resources import resource_filename
import pooch
 
from . import version


# Instantiate the pooch
__data_path = os.environ.get("CARAT_DATA_DIR", pooch.os_cache("carat"))
__GOODBOY = pooch.create(
    __data_path, base_url=f"https://github.com/mrocamora/carat/raw/{version.version}/examples/data/", registry=None
)

__GOODBOY.load_registry(
    pkg_resources.resource_stream(__name__, str(Path("example_data") / "registry.txt"))
)

with open(
    resource_filename(__name__, str(Path("example_data") / "index.json")), "r"
) as fdesc:
    __TRACKMAP = json.load(fdesc)


__all__ = ['find_nearest', 'STFT', 'hz2mel', 'mel2hz', 'deltas']


def find_nearest(array, value):
    """Find index of the nearest value of an array to a given value

    Parameters
    ----------
    array : np.ndarray
    	input array
    value : float
    	value

    Returns
    -------
    idx : int
    	index of nearest value in the array
    """

    idx = (np.abs(array-value)).argmin()

    return idx


def STFT(x, window_length, hop, windowing_function=np.hanning, dft_length=None,
         zp_flag=False):
    """ Calculates the Short-Time Fourier Transform a signal.

    Given an input signal, it calculates the DFT of frames of the signal and stores them
    in bi-dimensional Scipy array.

    Parameters
    ----------
    window_len : float
	length of the window in seconds (must be positive).
    window : callable
	a callable object that receives the window length in samples and
	returns a numpy array containing the windowing function samples.
    hop : float
	frame hop between adjacent frames in seconds.
    final_time : int
	time (in seconds) up to which the spectrogram is calculated (must be positive).
    zp_flag : bool
	a flag indicating if the *Zero-Phase Windowing* should be performed.

    Returns
    -------
    spec : np.array
	(missing)
    time : np.array
	(missing)
    frequency : np.array
	(missing)
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
    win_sig = part_sig * np.transpose(np.tile(window, (no_cols, 1)))
    # Zero-phase windowing:
    if zp_flag:
        win_sig = fft.fftshift(win_sig, axes=0)
    # Taking the FFT of the partitioned signal
    spec = fft.fftshift(fft.fft(win_sig, n=dft_length, axis=0), axes=0)
    # Normalizing energy
    spec /= np.sum(window)
    # Calculating time and frequency indices for the data
    frequency = fft.fftshift(fft.fftfreq(dft_length))
    time = np.arange(no_cols)*float(hop) + ((window_length-1)/2)
    # Creating output spectrogram
    return spec, time, frequency


def segmentSignal(signal, window_len, hop):
    """ Segmentation of an array-like input:

    Given an array-like, this function calculates the DFT of frames of the signal and stores them
    in bi-dimensional Scipy array.


    Parameters
    ----------
    signal : array-like
	object to be windowed. Must be a one-dimensional array-like object.
    window_len : int
	window size in samples.
    hop : int
	frame hop between adjacent frames in seconds.

    Returns
    -------
    part_sig : np.array
	2-D array containing the windowed signal. 
		
    Notes
    -----
    Each element of the output array X can be defined as:

        X[m,n] = x[n+Hm]

    where, H is the HOP in samples, 0<=n<=N, N = window_len, and 0<m<floor(((len(x)-N)/H)+1).

    Raises
    ------
    AttributeError if signal is not one-dimensional.
    ValueError if window_len or hop  are not strictly positives.

    """

    if(window_len <= 0 or hop <= 0):
        raise ValueError("window_len and hop values must be strictly positive numbers.")
    if signal.ndim != 1:
        raise AttributeError("Input signal must be one dimensional.")
    # Calculating the number of columns:
    no_cols = int(np.floor((np.size(signal)-window_len)/float(hop))+1)
    # Windowing indices (which element goes to which position in the windowed matrix).
    ind_col = np.tile(np.arange(window_len, dtype=np.uint64), (no_cols, 1))
    ind_line = np.tile(np.arange(no_cols, dtype=np.uint64)*hop, (window_len, 1))
    ind = np.transpose(ind_col) + ind_line
    # Partitioned signal:
    part_sig = signal[ind].copy()
    # Windowing partitioned signal
    return part_sig


def __get_segment(y, idx_ini, idx_end):
    """ Get a segment of an array, given by initial and ending indexes.

    Parameters
    ----------
    y : np.array
		one-dimensional array.
    idx_ini : int
		initial index.
    idx_end : int
		ending index.

    Returns
    -------
    segment : np.array
	segment of the signal.

    Raises
    ------
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

    Parameters
    ----------
    y : np.array
    	signal array (must be one-dimensional).
    time : np.array
    	corresponding time (must be one-dimensional).
    time_ini : int 
    	initial time value.
    time_end : int  
    	ending time value.

    Returns
    -------
    segment : np.array
    	segment of the signal.

    Raises
    ------
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
        If instead of beats, downbeats are used, then a measure is returned.

    Parameters
    ----------
    y : np.array
    	signal array (must be one-dimensional).
    time : np.array
    	corresponding time (must be one-dimensional).
    beats : np.array
    	time instants of the beats.
    ind_beat : int
    	index of the desired beat.

    Returns
    -------
    beat_segment : np.array
    	segment of the signal corresponding to the beat (or measure).

    Raises
    ------
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
    """ Returns a 2-D Numpy array of weights that maps a linearly spaced spectrogram
    to the Mel scale.

    Parameters
    ----------
    freq : np.array
    	frequency of the components of the DFT (must be one-dimensional).
    nfilts : 
    	number of output bands.
    minfreq : 
    	frequency of the first MEL coefficient.
    maxfreq : 
    	frequency of the last MEL coefficient.

    Returns
    -------
    wts : (check)
    	(check)
    binfrqs : (check)
        center frequencies in Hz of the Mel bands.

        """
    minmel = hz2mel(minfreq)
    maxmel = hz2mel(maxfreq)
    binfrqs = mel2hz(minmel+np.arange(nfilts+2)/(float(nfilts)+1)*(maxmel-minmel))
    wts = np.zeros((nfilts, (freq.size)))
    for i in range(nfilts):
        slp = binfrqs[i + np.arange(3)]
        loslope = (freq - slp[0])/(slp[1] - slp[0])
        hislope = (slp[2] - freq)/(slp[2] - slp[1])
        wts[i, :] = np.maximum(0.0, np.minimum(loslope, hislope))
    wts[:, freq < 0] = 0
    wts = np.dot(np.diag(2./(binfrqs[2+np.arange(nfilts)]-binfrqs[np.arange(nfilts)])), wts)
    binfrqs = binfrqs[1:nfilts+1]

    return wts, binfrqs


def hz2mel(f_hz):
    """ Converts a given frequency in Hz to the Mel scale.

    Parameters
    ----------
    f_hz : np.array
	array of frequencies in HZ that should be converted.

    Returns
    -------
    z_mel : np.array
        array (of same shape as f_zh) containing the converted frequencies.

    """
    f_0 = 0
    f_sp = 200.0/3.0 # Log step
    brkfrq = 1000.0 # Frequency above which the distribution stays linear.
    brkpt = (brkfrq - f_0)/f_sp # First Mel value for linear region.
    logstep = np.exp(np.lib.scimath.log(6.4)/27) # Step in the log region
    z_mel = np.where(f_hz < brkfrq, (f_hz - f_0)/f_sp, brkpt +
                     (np.lib.scimath.log(f_hz/brkfrq))/np.lib.scimath.log(logstep))
    return z_mel


def mel2hz(z_mel):
    """ Converts a given frequency in the Mel scale to Hz.

    Parameters
    ----------
    z_mel : np.array
	array of frequencies in the Mel scale that should be converted.

    Returns
    -------
    f_zh : np.array
        array (of same shape as z_mel) containing the converted frequencies.

    """
    f_0 = 0
    f_sp = 200.0/3.0
    brkfrq = 1000.0
    brkpt = (brkfrq - f_0)/f_sp
    logstep = np.exp(np.lib.scimath.log(6.4)/27)
    f_hz = np.where(z_mel < brkpt, f_0 + f_sp*z_mel, brkfrq*np.exp(np.lib.scimath.log(logstep)*(z_mel-brkpt)))
    return f_hz


def deltas(x, w=3):
    """ estimates the derivative of x
    
    Parameters
    ----------
    x : (check)
	(check)
    w : (check)
	(check)

    Returns
    -------
    d : (check)
        (check)

    """
    if x.ndim == 1:
        y = x.reshape((-1, 1)).T
    else:
        y = x
    if not w%2:
        w -= 1
    hlen = int(np.floor(w/2))
    if w == 1: # first-order difference:
        win = np.r_[1, -1]
    else:
        win = np.r_[hlen:-hlen-1:-1]
    # Extending the input data (avoid border problems)
    extended_x = np.c_[np.repeat(y[:, 0].reshape((-1, 1)), hlen, axis=1), y,
                       np.repeat(y[:, -1].reshape((-1, 1)), hlen, axis=1)]
    d = scipy.signal.lfilter(win, 1, extended_x, axis=1)
    d = d[:, 2*hlen:]
    if x.ndim == 1:
        d = d[0, :]
    return d


def getValidKeywords(kw, func):
    """ returns a dictionary containing the keywords arguments (in a list?) valid for a function.

    Parameters
    ----------
    kw : (check) 
    	(check)
    func : (check)
    	(check)
    
    Returns
    -------
    filename : str
        Path to the audio example file included with `carat`.
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


def example(key):

    """Retrieve the example file identified by 'key'.

    The first time an example is requested, it will be downloaded from
    the remote repository over HTTPS.
    All subsequent requests will use a locally cached copy of the recording.

    For a list of examples (and their keys), see `carat.util.list_examples`.

    By default, local files will be cached in the directory given by
    `pooch.os_cache('carat')`.  You can override this by setting
    an environment variable ``CARAT_DATA_DIR`` prior to importing carat:

    >>> import os
    >>> os.environ['CARAT_DATA_DIR'] = '/path/to/store/data'
    >>> import carat


    Parameters
    ----------
    key : str
        The identifier for the file to load

    Returns
    -------
    path : str
        The path to the requested example file

    Examples
    --------
    >>> # Load 10 seconds of the waveform from an example track of a chico drum
    >>> y, sr = carat.audio.load(carat.util.example("chico_audio"), duration=10.0))

    >>> # Load the waveform from the example track of the Ansina candombe recording
    >>> y, sr = carat.audio.load(carat.util.example("ansina_audio"))

    >>> # Load beats and downbeats from the example file of a chico drum
    >>> beats, b_labs = carat.annotations.load_beats(carat.util.example("chico_beats"))
    >>> downbeats, d_labs = carat.annotations.load_downbeats(carat.util.example("chico_beats"))

    >>> # Load onsets from the example file of a chico drum
    >>> onsets, onset_labs = carat.annotations.load_onsets(carat.util.example("chico_onsets"))
    """

    if key not in __TRACKMAP:
        raise ParameterError("Unknown example key: {}".format(key))

    return __GOODBOY.fetch(__TRACKMAP[key]["path"])


ex = example
"""Alias for example"""


def list_examples():
    """List the available example files included with Carat.

    Each file (audio file, beat annotations, onset annotations) is given 
    a unique identifier (e.g., "chico_audio" or "chico_onsets"),
    listed in the first column of the output.

    A brief description is provided in the second column.

    """
    print("AVAILABLE EXAMPLE FILES")
    print("-" * 68)
    for key in sorted(__TRACKMAP.keys()):
        print("{:10}\t{}".format(key, __TRACKMAP[key]["desc"]))


def compute_correlation_matrix(data1, data2, n=4):
    """Compute correlation matrix for two data sequences.
       The sequences' values are grouped in chunks of size n.

    Parameters
    ----------
    data1 : np.ndarray
        data sequence 1
    data2 : np.ndarray
        data sequence 2
    n : int
        grouping factor

    Returns
    -------
    CM : np.ndarray
        correlation matrix
    """

    # total data length
    N = data1.shape[0]
    # matrix elements
    M = int(np.floor(N / n))

    # correlation matrix
    CM = np.zeros((M,M))

    for k in range(M):
        # segment length
        sl = (k+1) * n
        # segment hop
        sh = n
        for s in range(M-k):
            # segment indexes
            ind_ini = s * sh 
            ind_end = ind_ini + sl
            # calculate Pearson's correlation
            corr, _ = pearsonr(data1[ind_ini:ind_end], data2[ind_ini:ind_end])
            # save correlation value
            CM[k, s] = corr

    return CM
