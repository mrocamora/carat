import numpy as np
import librosa
from scipy.ndimage import uniform_filter1d, gaussian_filter1d

from .tempogram import compute_autocorrelation_local, compute_tempogram_fourier
from .scale import scale_transform_matrix, FMT

def compute_scale_transform_magnitudes(y=None, o=None, sr=44100, target_sr=8000,
                                       n_fft=256, hop_length=160, detrend=True, n_mels=40, sigma=1,
                                       T_w=8, h=.5, norm_sum=False, norm='max',
                                       valid=True, scale_transform='librosa', t_min=.5, kind='cubic', M=None,
                                       fourier=True, theta=np.arange(30, 601), window='hamming'):
    """Compute rhythmic similarity representation

    Args:
        y (np.ndarray): Input signal
        o (np.ndarray): Onset strength
        sr (scalar): Original sampling rate
        target_sr (scalar): Target sampling rate (resampling)
        n_fft (int): Number of FFT points
        hop_length (int): Hop size
        detrend (bool): Makes the spectral flux locally zero-meaned (Default value = True)
        n_mels (int): Number of channels in mel-scale mapping
        sigma (scalar): Gaussian envelope parameter
        T_w (scalar): Window length (seconds)
        h (scalar): Hop size (seconds)
        norm_sum (bool): Normalizes by the number of summands in local autocorrelation (Default value = True)
        norm (str): If 'max', normalizes by the zeroth-lag value. If 'min-max' subtracts the minimum and
                    normalizes by the total range (Default value = 'max')
        valid (bool): Crops autocorrelation to valid (non-zero-padded) region (Default value = True)
        scale_transform (str): Scale transform implementation ('librosa', 'direct', else 'fast mellin')
        t_min (scalar): Minimum spacing for the Fast Scale Transform (< 1)
        kind (str): Interpolation used for Librosa FMT
        M (np.ndarray): Direct Scale Transform matrix
        fourier (bool): Returns Fourier tempogram
        Theta (np.ndarray): Set of tempi (given in BPM) (Default value = np.arange(30, 601))
        window (str): Name of the window function

    Returns:
        Rmean (np.ndarray): Mean feature representation
        rmean (np.ndarray): Mean tempogram
        Smean (np.ndarray): Mean Fourier tempogram
        R (np.ndarray): Feature representation
        r (np.ndarray): Tempogram
        S (np.ndarray): Fourier tempogram (cut according to Theta)
    """
    # This code implements the Scale Transform Magnitudes rhythm representation, as presented in
    #  [1] Holzapfel, Andre, and Yannis Stylianou. "A scale transform based method for rhythmic similarity
    #      of music." 2009 IEEE International Conference on Acoustics, Speech and Signal Processing. IEEE, 2009.
    #  [2] Holzapfel, André, and Yannis Stylianou. "Scale transform in rhythmic similarity of music."
    #      IEEE transactions on audio, speech, and language processing 19.1 (2010): 176-185.

    if y is not None:
        # Resampling the input signal
        x = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

        # Computing and smoothing the onset strength fuction
        # Similar to: http://labrosa.ee.columbia.edu/projects/coversongs/
        o = librosa.onset.onset_strength(y=x, sr=target_sr, n_fft=n_fft, hop_length=hop_length, detrend=detrend, n_mels=n_mels)
        if sigma > 0:
            o_n = gaussian_filter1d(o, sigma)
        else:
            o_n = o
        fs = target_sr / hop_length
    elif o is not None:
        o_n = o
        fs = sr
    else:
        return
    
    # Computing local autocorrelation
    N = int(np.ceil(T_w*fs))
    H = int(np.ceil(h*fs))
    r, _, _ = compute_autocorrelation_local(o_n, fs, N, H, norm_sum=norm_sum, norm=norm, valid=valid)
    
    # Computing scale transform of local autocorrelation
    if scale_transform == 'librosa':
        # Librosa implementation
        R = np.abs(librosa.fmt(r, t_min=t_min, kind=kind, beta=0.5, over_sample=1, axis=0))
    elif scale_transform == 'direct':
        # Direct scale transform implementation
        if M is None:
            M = scale_transform_matrix(N=N)
        R = np.abs(np.dot(M, (r[:-1] - r[1:])))
    else:
        # Fast Mellin transform (De Sena implementation)
        R = np.vstack([np.abs(FMT(r[:, i])) for i in range(r.shape[1])]).T
    
    # Computing periodicity spectra as well
    if fourier:
        S, _, _ = compute_tempogram_fourier(o_n, fs, N, H, Theta=theta, window=window, valid=valid)
        S = np.abs(S)
        #if valid:
        #    S = S[:,p1:-p2]
        for n in range(S.shape[1]):
            S[:, n] /= np.max(S[:, n])
        return np.mean(R, axis=1), np.mean(r, axis=1), np.mean(S, axis=1), R, r, S
    
    return np.mean(R, axis=1), np.mean(r, axis=1), _, R, r, _

def compute_multiband_scale_transform_magnitudes(y=None, sr=44100, target_sr=8000, channels=[0, 5, 40],
                                                 n_fft=256, hop_length=160, detrend=True, n_mels=40, sigma=1,
                                                 T_w=8, h=.5, norm_sum=False, norm='max',
                                                 valid=True, scale_transform='librosa', t_min=.5, kind='cubic', M=None,
                                                 fourier=True, theta=np.arange(30, 601), window='hamming'):
    """Compute rhythmic similarity representation

    Args:
        y (np.ndarray): Input signal
        sr (scalar): Original sampling rate
        target_sr (scalar): Target sampling rate (resampling)
        channels (list): List of channel numbers (boundaries)
        n_fft (int): Number of FFT points
        hop_length (int): Hop size
        detrend (bool): Makes the spectral flux locally zero-meaned (Default value = True)
        n_mels (int): Number of channels in mel-scale mapping
        sigma (scalar): Gaussian envelope parameter
        T_w (scalar): Window length (seconds)
        h (scalar): Hop size (seconds)
        norm_sum (bool): Normalizes by the number of summands in local autocorrelation (Default value = True)
        norm (str): If 'max', normalizes by the zeroth-lag value. If 'min-max' subtracts the minimum and normalizes by the total range (Default value = 'max')
        valid (bool): Crops autocorrelation to valid (non-zero-padded) region (Default value = True)
        scale_transform (str): Scale transform implementation ('librosa', 'direct', else 'fast mellin')
        t_min (scalar): Minimum spacing for the Fast Scale Transform (< 1)
        kind (str): Interpolation used for Librosa FMT
        M (np.ndarray): Direct Scale Transform matrix
        fourier (bool): Returns Fourier tempogram
        Theta (np.ndarray): Set of tempi (given in BPM) (Default value = np.arange(30, 601))
        window (str): Name of the window function

    Returns:
        Rmean (np.ndarray): Mean feature representation in different channels
    """
    # Resampling the input signal
    x = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    
    # Computing and smoothing the onset strength fuction
    # Similar to: http://labrosa.ee.columbia.edu/projects/coversongs/
    o = librosa.onset.onset_strength_multi(y=x, sr=target_sr, n_fft=n_fft, hop_length=hop_length,
                                           detrend=detrend, n_mels=n_mels, channels=channels)
    if sigma > 0:
        o_n = gaussian_filter1d(o, sigma, axis=-1)
    else:
        o_n = o
    fs = target_sr / hop_length
    
    R = []
    for o_i in o_n:
        Ri, _, _, _, _, _ = compute_scale_transform_magnitudes(o=o_i, sr=fs, target_sr=target_sr,
                                                               T_w=T_w, h=h, norm_sum=norm_sum, norm=norm,
                                                               valid=valid, scale_transform=scale_transform, t_min=t_min,
                                                               kind=kind, M=M, fourier=fourier, theta=theta, window=window)
        R.append(Ri)
    R = np.vstack(R)
    return R

def compute_onset_patterns(y=None, sr=44100, target_sr=8000, channels=[0, 40],
                           n_fft=256, hop_length1=160, n_mels=40,
                           hop_length2=25, fmin=.5, n_bins=25, bins_per_octave=5, filter_scale=1,
                           log_before=True, diff=False, log_after=False, masking=True, moving_average=.25,
                           detrend=True, pad=True, min_duration=30):
    """Compute rhythmic similarity representation

    Args:
        y (np.ndarray): Input signal
        sr (scalar): Original sampling rate
        target_sr (scalar): Target sampling rate (resampling)
        channels (list): List of channel numbers (boundaries)
        n_fft (int): Number of FFT points
        hop_length1 (int): Hop size (first transform)
        n_mels (int): Number of channels in mel-scale mapping
        hop_length2 (int): Hop size (second transform)
        fmin (scalar): Minimum modulation frequency
        n_bins (int): Number of modulation frequency bins
        bins_per_octave (int): Number of modulation bins per octave
        filter_scale (scalar): CQT filter scale
        log_before (bool): Applies log-compression before the unsharp mask
        diff (bool): Applies first derivative (time) before the unsharp mask
        log_after (bool): Applies log-compression after the unsharp mask
        masking (bool): Unsharp masking
        moving_average (scalar): Parameter for unsharp masking (seconds)
        detrend (bool): Makes the first stage locally zero-meaned (Default value = True)
        pad (bool): Pad the signal to a minimum duration
        min_duration (scalar): Minimum duration (in seconds)

    Returns:
        Rmean (np.ndarray): Mean feature representation in different channels
    """

    if pad and (len(y) < min_duration * sr):
        y = np.pad(y, (0, min_duration * sr - len(y)))
        
    x = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    fs = target_sr
    
    # First stage
    X = librosa.feature.melspectrogram(y=x, sr=fs, n_fft=n_fft, hop_length=hop_length1, n_mels=n_mels)
    X = np.abs(X)
    
    # Logarithm
    if log_before:
        X = 20*np.log10(1e-16+X)
    
    if diff:
        X = np.pad(np.diff(X, axis=1), ((0,0), (1,0)))
    
    # Unsharp mask
    if masking:
        mask = np.zeros(X.shape)
        N = int(np.ceil(moving_average * fs/hop_length1))
        if N % 2:
            N += 1
        for i, x in enumerate(X):
            mask[i,:] = uniform_filter1d(x, size=N)
        X = np.maximum(0, X - mask)
    
    if log_after:
        X = 20*np.log10(1e-16+X)
    
    # Second stage
    M = []
    for x in X.copy():
        if detrend:
            x -= np.mean(x)
        Y = librosa.cqt(x, sr=fs/hop_length1, hop_length=hop_length2, fmin=fmin, n_bins=n_bins,
                        bins_per_octave=bins_per_octave, filter_scale=filter_scale)
        M.append(np.abs(Y))
    M = np.stack(M)
    
    n_bands = len(channels) - 1
    m_mean = []
    m_median = []
    for i in range(n_bands):
        ni = channels[i]
        nf = channels[i+1]
        m_mean.append(np.mean(np.mean(M[ni:nf], axis=0), axis=1))
        m_median.append(np.mean(np.median(M[ni:nf], axis=0), axis=1)) 
    m_mean = np.concatenate(m_mean)
    m_median = np.concatenate(m_median)
                 
    return m_mean, m_median