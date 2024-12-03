import numpy as np
import librosa

def compute_tempogram_fourier(x, Fs, N, H, Theta=np.arange(30, 601, 1), window='hann', fftbins=True, valid=False):
    """Compute Fourier-based tempogram [FMP, Section 6.2.2]

    Notebook: C6/C6S2_TempogramFourier.ipynb

    Args:
        x (np.ndarray): Input signal
        Fs (scalar): Sampling rate
        N (int): Window length
        H (int): Hop size
        Theta (np.ndarray): Set of tempi (given in BPM) (Default value = np.arange(30, 601, 1))
        window (str): Name of the window function
        fftbins (bool): Whether window is periodic (True) or symmetric (False)
        valid (bool): Computes autocorrelation without padding (Default value = False)

    Returns:
        X (np.ndarray): Tempogram
        T_coef (np.ndarray): Time axis (seconds)
        F_coef_BPM (np.ndarray): Tempo axis (BPM)
    """
    # This code was adapted from Meinard Müller's FMP notebooks, which are licensed under a 
    # MIT License: https://opensource.org/licenses/MIT
    # https://www.audiolabs-erlangen.de/resources/MIR/FMP/C6/C6S2_TempogramFourier.html

    # win = np.hanning(N)
    win = librosa.filters.get_window(window, N, fftbins=fftbins)
    if valid == False:
        N_left = N // 2
        L = x.shape[0]
        L_left = N_left
        L_right = N_left
        L_pad = L + L_left + L_right
        # x_pad = np.pad(x, (L_left, L_right), 'constant')  # doesn't work with jit
        x_pad = np.concatenate((np.zeros(L_left), x, np.zeros(L_right)))
    else:
        L_pad = x.shape[0]
        x_pad = x
    t_pad = np.arange(L_pad)
    M = int(np.floor(L_pad - N) / H) + 1
    K = len(Theta)
    X = np.zeros((K, M), dtype=np.complex_)

    for k in range(K):
        omega = (Theta[k] / 60) / Fs
        exponential = np.exp(-2 * np.pi * 1j * omega * t_pad)
        x_exp = x_pad * exponential
        for n in range(M):
            t_0 = n * H
            t_1 = t_0 + N
            X[k, n] = np.sum(win * x_exp[t_0:t_1])
    if valid == False:
        T_coef = np.arange(M) * H / Fs
    else:
        T_coef = np.arange(M) * H / Fs + (N // 2) / Fs
    F_coef_BPM = Theta
    return X, T_coef, F_coef_BPM

def compute_autocorrelation_local(x, Fs, N, H, norm_sum=True, norm='max', valid=False):
    """Compute local autocorrelation [FMP, Section 6.2.3]

    Notebook: C6/C6S2_TempogramAutocorrelation.ipynb

    Args:
        x (np.ndarray): Input signal
        Fs (scalar): Sampling rate
        N (int): Window length
        H (int): Hop size
        norm_sum (bool): Normalizes by the number of summands in local autocorrelation (Default value = True)
        norm (str): If 'max', normalizes by the zeroth-lag value. If 'min-max' subtracts the minimum and normalizes by the total range (Default value = 'max')
        valid (bool): Computes autocorrelation without padding (Default value = False)

    Returns:
        A (np.ndarray): Time-lag representation
        T_coef (np.ndarray): Time axis (seconds)
        F_coef_lag (np.ndarray): Lag axis
    """
    # This code was adapted from Meinard Müller's FMP notebooks, which are licensed under a 
    # MIT License: https://opensource.org/licenses/MIT
    # https://www.audiolabs-erlangen.de/resources/MIR/FMP/C6/C6S2_TempogramAutocorrelation.html

    # L = len(x)
    if valid == False:
        L_left = round(N / 2)
        L_right = L_left
        x_pad = np.concatenate((np.zeros(L_left), x, np.zeros(L_right)))
    else:
        x_pad = x
    L_pad = len(x_pad)
    M = int(np.floor(L_pad - N) / H) + 1
    A = np.zeros((N, M))
    win = np.ones(N)
    if norm_sum is True:
        lag_summand_num = np.arange(N, 0, -1)
    for n in range(M):
        t_0 = n * H
        t_1 = t_0 + N
        x_local = win * x_pad[t_0:t_1]
        r_xx = np.correlate(x_local, x_local, mode='full')
        r_xx = r_xx[N-1:]
        if norm_sum is True:
            r_xx = r_xx / lag_summand_num
        if norm == 'max':
            r_xx = r_xx / r_xx[0]
        elif norm == 'min-max':
            r_xx = (r_xx - r_xx.min()) / (r_xx.max() - r_xx.min())
        A[:, n] = r_xx
    Fs_A = Fs / H
    if valid == False:
        T_coef = np.arange(A.shape[1]) / Fs_A
    else:
        T_coef = np.arange(A.shape[1]) / Fs_A + (N // 2) / Fs
    F_coef_lag = np.arange(N) / Fs
    return A, T_coef, F_coef_lag
