import numpy as np
from scipy.interpolate import interp1d

def limit_scales(feature, shape, s):
    """Change the dimension of a feature of flattened np.ndarrays

    Args:
        feature (np.ndarray): Input features
        shape (tuple): Original shape of (unflattened) individual feature
        s (int): Maximum scale index

    Returns:
        feature_lim (np.ndarray): Output features
    """
    if feature.ndim == 1:
        feat = feature[np.newaxis,:]
    else:
        feat = feature
    feature_lim = np.zeros((feat.shape[0], s * shape[0]))
    for i, R in enumerate(feat):
        R_lim = R.reshape(shape)[:, :s]
        feature_lim[i,:] = R_lim.flatten()
    return feature_lim

def scale_transform_matrix(N=400, C=500, fs=None):
    """Compute Direct Scale Transform matrix

    Args:
        N (int): Input dimension
        C (int): Maximum scale coefficient

    Returns:
        M (np.ndarray): Transformation matrix
    """
    # This code was derived from the original DST paper:
    #  [1] Williams, W. J., and E. J. Zalubas. "Helicopter transmission fault detection via 
    #      time-frequency, scale and spectral methods." Mechanical systems and signal processing
    #      14.4 (2000): 545-559.
    if fs is None:
        Ts = 1
    else:
        Ts = 1/fs
    k, c = np.meshgrid(np.arange(1,N), np.arange(0, C, np.pi/np.log(N+1)))
    M = np.power(k*Ts, 1/2 - 1j*c)
    M /= (1/2-1j*c)*np.sqrt(2*np.pi)
    return M

def FMT(x, n=None, fs=None, a=None):
    """Compute Fast Mellin (Scale) Transform

    Args:
        x (np.ndarray): Input signal
        n (int): Number of samples to use
        fs (scalar): Sampling rate
        a (scalar): Temporal starting point (in seconds)

    Returns:
        X (np.ndarray): Mellin transform of x
    """
    # This code was ported to Python from the original (in Matlab), which is distributed under a
    # GNU GPL v.2 License: https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html
    # https://www.researchgate.net/publication/305331192_Fast_MellinScale_Transform_v120

    # Sampling frequency
    if fs is None:
        fs = 1
    
    # Length of the input vector
    if n is None:
        n = len(x)
    
    # Starting point
    if a is None:
        a = 1/fs
    
    # Sampling period
    Ts = 1/fs
    
    # Real part of the Mellin parameter p
    # for the scale transform beta = 1/2
    beta = 1/2
    
    # Temporal endpoint
    b = a + n*Ts
    
    NLN = nlognex(n, Ts, a)
    
    ################
    # Time Warping #
    ################
    ExpAxe = CreateExpAxe(n, NLN, Ts, a)
    inter1x = np.linspace(a, b, n)
    inter1y = x[0:n]
    
    #################
    # Interpolation #
    #################  
    #iType = 'cubic'
    #stlength = 16
    #iWS = interp1ext(inter1x, inter1y, ExpAxe, iType, stlength)
    iWS = interp1d(inter1x, inter1y, kind='cubic')(ExpAxe)
    
    #################################################
    # Point-by-point multiply by exponential factor #
    #################################################
    ExpTimeBeta = np.power(ExpAxe, beta)
    xw = iWS * ExpTimeBeta
    
    ############################
    # Mellin transform via FFT #
    ############################
    X = np.fft.rfft(xw)
    
    ########################
    # Energy normalization #
    ########################
    X /= np.sqrt(fs * NLN * n)
    return X
    
def nlognex(n, Ts, a):
    # This code was ported to Python from the original (in Matlab), which is distributed under a
    # GNU GPL v.2 License: https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html
    # https://www.researchgate.net/publication/305331192_Fast_MellinScale_Transform_v120

    # Get the quality factor
    QF = ERQualityFactor();
    
    # JMP factor
    JMP = 1/QF
    
    # Equation proved in the Thesis
    p = np.log((a + (n * Ts)) / a)
    q = np.log((a + (n * Ts)) /(a + ((n - JMP) *Ts)))
    N = int(np.ceil(p / q))
    
    # Force N to be even
    if N % 2 != 0:
        N += 1
    return N

def CreateExpAxe(src_n, dst_N, src_Ts, a):
    # This code was ported to Python from the original (in Matlab), which is distributed under a
    # GNU GPL v.2 License: https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html
    # https://www.researchgate.net/publication/305331192_Fast_MellinScale_Transform_v120

    # Get the quality factor
    QF = ERQualityFactor();
    
    # Factor for the computation of the distance between last two
    # "virtual" (computed like if QF=1) samples
    JMP = 1/QF
    
    ExpAxe = np.zeros(dst_N)

    # Computes the Exponential Axe
    b = a + (src_n * src_Ts) # Temporal End point
    Alpha = (b - (src_Ts * JMP)) / b # Exponential Base
    m = b * Alpha**dst_N
    for idx in range(1, dst_N + 1):
        ExpAxe[idx-1] = m * Alpha**(-idx)
        
    # Be sure that the axe starts from a and end in a+(n*src_Ts)
    ExpAxe = (ExpAxe - ExpAxe[0])/(ExpAxe[-1] - ExpAxe[0])
    ExpAxe = (ExpAxe * (b-a)) + a
    ExpAxe[0] = a
    ExpAxe[-1] = b
    return ExpAxe

def ERQualityFactor():
    # This code was ported to Python from the original (in Matlab), which is distributed under a
    # GNU GPL v.2 License: https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html
    # https://www.researchgate.net/publication/305331192_Fast_MellinScale_Transform_v120

    ERQF = 1 # Standard quality. The worst error is about 10^-1 (order)
    #ERQF = 2 # High quality, take more time (best quality vs speed)
    #ERQF = 3 # Super high quality
    #ERQF = .5 # Low quality (warning: aliasing)
    return ERQF