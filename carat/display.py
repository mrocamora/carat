# encoding: utf-8
# pylint: disable=C0103
# pylint: disable=too-many-arguments
"""
Display
=======
.. autosummary::
    :toctree: generated/

    wave_plot
    map_show
    feature_plot
    embedding_plot
    centroids_plot
    plot_centroid
"""

import copy
import numpy as np
from pylab import get_cmap
from matplotlib import colors
import matplotlib.cm as cm
from matplotlib.axes import Axes
import matplotlib.ticker as mticker
from matplotlib.ticker import NullFormatter
from librosa.display import TimeFormatter
from . import util
from . import microtiming
from .exceptions import ParameterError

__all__ = ['wave_plot', 'map_show', 'feature_plot', 'embedding_plot',
           'centroids_plot', 'plot_centroid', 'onsets_in_grid_plot']


def wave_plot(y, sr=22050, x_axis='time', beats=None, beat_labs=None,
              ax=None, **kwargs):
    '''Plot an audio waveform and beat labels (optinal).

    Parameters
    ----------
    y : np.ndarray
        audio time series
    sr : number > 0 [scalar]
        sampling rate of `y`
    x_axis : str {'time', 'off', 'none'} or None
        If 'time', the x-axis is given time tick-marks.
    ax : matplotlib.axes.Axes or None
        Axes to plot on instead of the default `plt.gca()`.
    kwargs
        Additional keyword arguments to `matplotlib.`

    Returns
    -------

    '''

    kwargs.setdefault('color', 'royalblue')
    kwargs.setdefault('linestyle', '-')
    kwargs.setdefault('alpha', 0.6)

    if y.ndim > 1:
        raise ValueError("`y` must be a one dimensional array. "
                         "Found y.ndim={}".format(y.ndim))

    # time array in seconds
    time = np.arange(y.size)/sr
    # its maximum value
    max_time = np.max(time)

    # check axes and create it if needed
    axes = __check_axes(ax)

    # plot waveform
    out = axes.plot(time, y, **kwargs)

    if beats is not None:
        __plot_beats(beats, max_time, axes, beat_labs=beat_labs, **kwargs)

    # format x axis
    if x_axis == 'time':
        axes.xaxis.set_major_formatter(TimeFormatter(lag=False))
        axes.xaxis.set_label_text('Time (s)')
    elif x_axis is None or x_axis in ['off', 'none']:
        axes.set_xticks([])
    else:
        raise ParameterError('Unknown x_axis value: {}'.format(x_axis))

    return out


def feature_plot(feature, time, x_axis='time', beats=None, beat_labs=None,
                 ax=None, **kwargs):
    '''Plot an audio waveform and beat labels (optinal).


    Parameters
    ----------
    feature : np.ndarray
        feature time series
    time : np.ndarray
        time instant of the feature values
    x_axis : str {'time', 'off', 'none'} or None
        If 'time', the x-axis is given time tick-marks.
    ax : matplotlib.axes.Axes or None
        Axes to plot on instead of the default `plt.gca()`.
    kwargs
        Additional keyword arguments to `matplotlib.`

    Returns
    -------

    '''

    kwargs.setdefault('color', 'seagreen')
    kwargs.setdefault('linestyle', '-')
    kwargs.setdefault('alpha', 0.8)

    if feature.ndim > 1:
        raise ValueError("`feature` must be a one dimensional array. "
                         "Found feature.ndim={}".format(feature.ndim))

    # maximum time value
    max_time = np.max(time)

    # check axes and create it if needed
    axes = __check_axes(ax)

    # plot waveform
    out = axes.plot(time, feature, **kwargs)

    if beats is not None:
        __plot_beats(beats, max_time, axes, beat_labs=beat_labs, **kwargs)

    # format x axis
    if x_axis == 'time':
        axes.xaxis.set_major_formatter(TimeFormatter(lag=False))
        axes.xaxis.set_label_text('Time (s)')
    elif x_axis is None or x_axis in ['off', 'none']:
        axes.set_xticks([])
    else:
        raise ParameterError('Unknown x_axis value: {}'.format(x_axis))

    return out


def centroids_plot(centroids, n_tatums=4, ax_list=None, **kwargs):
    '''Plot centroids of rhythmic patterns clusters.


    Parameters
    ----------
    centroids: np.ndarray
        centroids of the rhythmic patterns clusters
    n_tatums : int
        Number of tatums (subdivisions) per tactus beat
    ax_list : list of matplotlib.axes.Axes or None, one element per centroid
        Axes to plot on instead of the default `plt.gca()`.
    kwargs
        Additional keyword arguments to `matplotlib.`

    Returns
    -------
    ax : list of matplotlib.axes.Axes

    '''
    # number of centroids
    n_centroids = len(centroids)

    # check list of axes
    ax = __check_axes_list(n_centroids, ax_list=ax_list)

    # get colormap
    cmap, _ = __get_colormap_map(n_centroids)

    # plot each cluster
    for ind, centroid in enumerate(centroids):
        plot_centroid(centroid, n_tatums=n_tatums, ax=ax[ind],
                      color=cmap(ind/n_centroids), **kwargs)

    return ax


def plot_centroid(centroid, n_tatums=4, ax=None, **kwargs):
    '''Plot centroid of a rhythmic patterns cluster.


    Parameters
    ----------
    centroid : np.ndarray
        centroid feature values
    n_tatums : int
        Number of tatums (subdivisions) per tactus beat
    ax : matplotlib.axes.Axes or None
        Axes to plot on instead of the default `plt.gca()`.
    kwargs
        Additional keyword arguments to `matplotlib.`

    Returns
    -------

    '''

    kwargs.setdefault('color', 'seagreen')
    kwargs.setdefault('alpha', 0.8)

    if centroid.ndim > 1:
        raise ValueError("`centroid` must be a one dimensional array. "
                         "Found centroid.ndim={}".format(centroid.ndim))

    # number of tatums in centroid
    c_tatums = centroid.size

    # check axes and create it if needed
    axes = __check_axes(ax)

    # plot centroid
    out = axes.bar(np.arange(c_tatums)+1, centroid, **kwargs)

    # configure tickers and labels
    __decorate_axis_centroid(axes, c_tatums, n_tatums)

    return out


def __plot_beats(beats, max_time, ax, beat_labs=None, **kwargs):
    '''Plot beat labels.

    Parameters
    ----------
    beats : np.ndarray
        audio time series
    beat_labs : list
        beat labels
    x_axis : str {'time', 'off', 'none'} or None
        If 'time', the x-axis is given time tick-marks.
    ax : matplotlib.axes.Axes or None
        Axes to plot on instead of the default `plt.gca()`.
    kwargs
        Additional keyword arguments to `matplotlib.`

    Returns
    -------

    '''

    kwargs['color'] = 'black'
    kwargs.setdefault('linestyle', '-')
    kwargs['alpha'] = 0.3
    kwargs.setdefault('linewidth', 2)

    # replace nan values to 0
    beats = np.nan_to_num(beats)

    # consider beats (and labels) bellow max_time
    ind_beat = util.find_nearest(beats, max_time)
    new_beats = beats[:ind_beat]
    if beat_labs is not None:
        new_labs = beat_labs[:ind_beat]

    # plot beat annotations
    for beat in new_beats:
        ax.axvline(x=beat, **kwargs)

    # set ticks and labels
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(new_beats)
    if beat_labs is not None:
        ax2.set_xticklabels(new_labs)
    else:
        ax2.set_xticklabels([])

    return ax2


def map_show(data, x_coords=None, y_coords=None, ax=None,
             n_tatums=4, clusters=None, **kwargs):
    '''Display a feature map.

    Parameters
    ----------
    data : np.ndarray
        Feature map to display
    x_coords : np.ndarray [shape=data.shape[1]+1]
    y_coords : np.ndarray [shape=data.shape[0]+1]
        Optional positioning coordinates of the input data.
    ax : matplotlib.axes.Axes or None
        Axes to plot on instead of the default `plt.gca()`.
    n_tatums : int
        Number of tatums (subdivisions) per tactus beat
    clusters : np.ndarray
        Array indicating cluster number for each pattern of the input data.
        If provided (not None) the clusters area displayed with colors.
    kwargs : additional keyword arguments
        Arguments passed through to `matplotlib.pyplot.pcolormesh`.
        By default, the following options are set:
            - `cmap=gray_r`
            - `rasterized=True`
            - `edgecolors='None'`
            - `shading='flat'`

    Returns
    -------
    axes
        The axis handle for the figure.

    See Also
    --------
    matplotlib.pyplot.pcolormesh

    '''

    kwargs.setdefault('cmap', cm.get_cmap('gray_r'))
    kwargs.setdefault('rasterized', True)
    kwargs.setdefault('edgecolors', 'None')
    kwargs.setdefault('shading', 'flat')

    # number of bars
    bars = data.shape[0]
    # number of tatums in a bar
    tatums = data.shape[1]

    # set the x and y coordinates
    y_coords = np.array(range(tatums+1))+0.5
    x_coords = np.array(range(bars+1))+1

    # check axes and create it if needed
    axes = __check_axes(ax)
    # plot rhythmic patterns map (grayscale)
    out = axes.pcolormesh(x_coords, y_coords, data.T, **kwargs)
    __set_current_image(ax, out)

    # if clusters are given then show them in colors
    if clusters is not None:
        # check clusters and return number of clusters
        n_clusters = __check_clusters(clusters, bars)
        # matrix to plot clusters' map
        mapc = __get_cluster_matrix(clusters, y_coords.size-1)
        # get colormap used to plot clusters
        cmap, norm = __get_colormap_map(n_clusters)
        # plot clusters in colors
        axes.pcolormesh(x_coords, y_coords, mapc, cmap=cmap, norm=norm, alpha=0.6)

    # set axes limits
    axes.set_xlim(x_coords.min()-0.5, x_coords.max()+0.5)
    axes.set_ylim(y_coords.min(), y_coords.max())


    # configure tickers and labels
    __decorate_axis_map(axes, tatums=n_tatums)

    return axes


def embedding_plot(data, clusters=None, ax=None, **kwargs):
    '''Display an 2D or 3D embedding of the rhythmic patterns data.

    Parameters
    ----------
    data : np.ndarray
        Low-embedding data points
    ax : matplotlib.axes.Axes or None
        Axes to plot on instead of the default `plt.gca()`.
    clusters : np.ndarray
        Array indicating cluster number for each point of the input data.
        If provided (not None) the clusters area displayed with colors.
    kwargs : additional keyword arguments
        Arguments passed through to `matplotlib.pyplot.pcolormesh`.

    Returns
    -------
    axes
        The axis handle for the figure.

    See Also
    --------
    matplotlib.pyplot.pcolormesh

    '''

    # number of points
    points = data.shape[0]

    # check if clusters are provided
    if clusters is not None:
        # check clusters and return number of clusters
        n_clusters = __check_clusters(clusters, points)

        # get colormap used to plot clusters
        cmap, norm = __get_colormap_map(n_clusters)

    # check axes and create it if needed
    axes = __check_axes(ax)

    # data dimension to check it is 2D or 3D
    dim = data.shape[1]

    if dim == 3:
        if clusters is None:
            axes.scatter(data[:, 0], data[:, 1], data[:, 2], **kwargs)
        else:
            axes.scatter(data[:, 0], data[:, 1], data[:, 2], c=clusters,
                         cmap=cmap, norm=norm, picker=2, **kwargs)
        __decorate_axis_embedding(axes, dim)

    elif dim == 2:
        if clusters is None:
            axes.scatter(data[:, 0], data[:, 1], **kwargs)
        else:
            axes.scatter(data[:, 0], data[:, 1], c=clusters,
                         cmap=cmap, norm=norm, picker=2, **kwargs)
        __decorate_axis_embedding(axes, dim)

    else:
        raise ValueError("`data` points can have two or three dimension to be plotted. "
                         "Found data.shape[1]={}".format(data.shape[1]))

    return axes


def __check_axes(axes):
    '''Check if "axes" is an instance of an axis object.'''
    if axes is None:
        import matplotlib.pyplot as plt
        axes = plt.gca()
    elif not isinstance(axes, Axes):
        raise ValueError("`axes` must be an instance of matplotlib.axes.Axes. "
                         "Found type(axes)={}".format(type(axes)))
    return axes


def __check_axes_list(n_axes, ax_list=None):
    '''Check if a list has a given length and each element is an instance of an axis object.

    Parameters
    ----------
    n_axes : int
    
    ax_list : 
    
    '''
    if ax_list is None:
        import matplotlib.pyplot as plt
        fig = plt.gcf()
        ax_list = []
        for ind in range(n_axes):
            ax = fig.add_subplot(n_axes, 1, ind+1)
            ax_list.append(ax)
    elif n_axes != len(ax_list):
        raise ValueError("`ax_list` must be of correct size to match number of axes `n_axes`.")
    else:
        for ind in range(n_axes):
            if not isinstance(ax_list[ind], Axes):
                raise ValueError("`axes` must be an instance of matplotlib.axes.Axes. "
                                 "Found type(axes)={}".format(type(ax_list[ind])))
    return ax_list


def __check_clusters(clusters, bars):
    '''Check if "clusters" is an instance of an axis object.
       Check if "clusters" is a one dimensional array of the correct length.
       '''
    if isinstance(clusters, np.ndarray):
        if clusters.ndim == 1:
            if clusters.size == bars:
                # count number of clusters
                n_clusters = np.unique(clusters).size
        else:
            raise ValueError("`clusters` must be a one dimensional array. "
                             "Found clusters.ndim={}".format(clusters.ndim))
    else:
        raise ValueError("`clusters` must be an instance of numpy.ndarray. "
                         "Found type(axes)={}".format(type(clusters)))

    return n_clusters


def __get_cluster_matrix(clusters, n_tatums):
    '''Get clusters' matrix to plot clusters in map.'''
    mapc = np.tile(clusters+0.5, (n_tatums, 1))

    return mapc


def __get_colormap_map(n_clusters):
    '''Get colormap for clusters' matrix.'''

    # make a color map of fixed colors for colormesh
    # cmap = get_cmap('RdBu', n_clusters)
    cmap = get_cmap('tab10', n_clusters)

    bounds = range(n_clusters+1)
    norm = colors.BoundaryNorm(bounds, cmap.N)

    return cmap, norm


def __set_current_image(ax, img):
    '''Helper to set the current image in pyplot mode.

    If the provided `ax` is not `None`, then we assume that the user is using the object API.
    In this case, the pyplot current image is not set.
    '''

    if ax is None:
        import matplotlib.pyplot as plt
        plt.sci(img)


def __decorate_axis_map(axis, tatums=4):
    '''Configure axis ticks and labels for feature map plot'''

    # ticks at beats
    ylims = axis.get_ylim()
    all_tatums = int(ylims[1])
    ticks_beats = [x+0.5 for x in range(0, all_tatums, tatums)]
    num_beats = int(all_tatums / tatums)
    labels_beats = [x+1 for x in range(num_beats)]

    axis.yaxis.set_ticks(ticks_beats)
    axis.set_yticklabels(labels_beats)
    axis.tick_params(labelsize=10)
    # axis.yaxis.set_major_formatter(NullFormatter())
    axis.yaxis.grid()
    gridlines = axis.get_ygridlines()
    for line in gridlines:
        line.set_linestyle('-')
        line.set_linewidth(2)
        line.set_color('black')
    axis.set_ylabel('beats')


def __decorate_axis_centroid(axis, c_tatums=16, n_tatums=4, beat_ticks=True):
    '''Configure axis ticks and labels for centroid plot.

    Parameters
    ----------
    axis : matplotlib.axes.Axes or None
    c_tatums : int
        Number of tatums (subdivisions) in centroid
    n_tatums : int
        Number of tatums (subdivisions) per tactus beat
    beat_ticks : bool
        If `True`, then labels are shown only at ticks corresponding to beats

    '''
    tatums = np.arange(c_tatums)
    axis.xaxis.set_ticks(tatums + 1)
    axis.xaxis.set_ticks_position('top')
    axis.yaxis.set_major_formatter(NullFormatter())
    axis.set_ylim(0, 1)

    if beat_ticks:
        beat_labs = [int(x / n_tatums) + 1 if (x % n_tatums) == 0 else ' ' for x in tatums]
        axis.set_xticklabels(beat_labs)


def __decorate_axis_embedding(axes, dim):
    '''Configure axis ticks and labels for embedding plot'''

    if dim == 3:
        axes.zaxis.set_major_formatter(NullFormatter())
    axes.xaxis.set_major_formatter(NullFormatter())
    axes.yaxis.set_major_formatter(NullFormatter())


def onsets_in_grid_plot(onsets_in_grid, ax=None, mean_std=True, hist_ons=False,
                        n_bins=100, fs=14, **kwargs):
    '''Plot onsets in grid

    Parameters
    ----------
    onsets_in_grid : list of np.ndarray
        onsets assigned to metrical grid as a list of arrays, one element for each instrument
    ax : matplotlib.axes.Axes or None
        Axes to plot on instead of the default `plt.gca()`.
    mean_std : bool
        If `True`, then mean and std are plotted for each subdivision
    hist_ons : bool
        If `True`, then an histogram of all onsets is plotted.
    n_bins : int
        Number of bins to use for the histogram.
    fs : int
        Font size.
    kwargs
        Additional keyword arguments to `matplotlib.`

    Returns
    -------

    '''

    kwargs.setdefault('color', 'seagreen')
    kwargs.setdefault('alpha', 0.6)
    kwargs.setdefault('marker', 'o')
    kwargs.setdefault('markersize', 3)

    # number of beats
    num_beats = len(onsets_in_grid)

    # top and bottom margins of the plot
    margin_top = int(num_beats/10)
    if hist_ons:
        margin_bottom = 4 * margin_top
    elif mean_std:
        margin_bottom = 3 * margin_top
    else:
        margin_bottom = 2 * margin_top

    # total height of the plot
    total_height = margin_bottom + num_beats + margin_top

    # number of subdivisions
    num_subdivs = onsets_in_grid[0].shape[0]

    # check axes and create it if needed
    axes = __check_axes(ax)

    # plot onsets at each beat
    for ind in range(num_beats):
        # onsets in current beat
        onsets = onsets_in_grid[ind]

        # plot the onsets at current subdivisio
        out = axes.plot(onsets, ind * np.ones(num_subdivs) + margin_bottom,
                        linestyle='None', **kwargs)

    # configure tickers and labels
    __decorate_axis_subdivisions(axes, num_subdivs, total_height)

    if mean_std:
        # fit a normal distribution to each subdivision
        mus, stds = microtiming.onsets_to_normal_dist(onsets_in_grid)

        # plot mean and stdev for each subdivision
        for ind in range(num_subdivs):
            axes.errorbar(mus[ind], (margin_bottom*2/3), xerr=stds[ind], fmt='.',
                          capsize=1, color='royalblue')
            axes.axvline(x=mus[ind], ymin=(margin_bottom*2/3)/total_height,
                         ymax=(margin_bottom+num_beats+60)/total_height,
                         linestyle='--', color='royalblue')
            axes.text(mus[ind], (margin_bottom/10), "{:3.0f}".format(mus[ind]*100)+"%",
                      color='royalblue', horizontalalignment='center',
                      verticalalignment='bottom', fontsize=fs)

    if hist_ons:
        # flatten onsets to plot histogram
        onsets_flattened = [val for sublist in onsets_in_grid for val in sublist]
        axes.hist(onsets_flattened, bins=n_bins, density=False, alpha=0.2, facecolor='black')

    return out


def __decorate_axis_subdivisions(axes, num_subdivs, total_height, fs=14):
    '''Configure axis ticks and labels for subdivisions plot
    
    Parameters
    ----------
    axes : 
    num_subdivs : 
    total_height : 
    fs : int
        Font size
    
    '''

    # x-ticks at every subdivision
    x_ticks = np.linspace(0, 1, num_subdivs+1)
    x_labels = ['.'+str(num+1) for num in range(num_subdivs)]
    axes.set_xticks(x_ticks[:-1])
    axes.set_xticklabels(x_labels, fontsize=fs)
    axes.set_yticks([])
    axes.tick_params(length=10, width=1)
    axes.axis([-0.10, 1, 0, total_height])
    axes.spines['right'].set_visible(False)
    axes.spines['left'].set_visible(False)
    axes.set_ylabel(r'beats $\longrightarrow$', fontsize=fs)

def tempo_curve_plot(bpms, beat_labels, ax=None, xlabel='Time (bar)', ylabel='Tempo (BPM)', **kwargs):  
    """Display tempo curve plot

    Parameters
    ----------
    bpms : np.ndarray
        tempo values as beats per minute (bpm)
    beat_labels : list
        labels at the beats (e.g. 1.1, 1.2, etc)
    ax : matplotlib.axes.Axes or None
        Axes to plot on instead of the default `plt.gca()`.
    kwargs
        Additional keyword arguments to `matplotlib.`

    Returns
    -------
    axes
        The axis handle for the figure.
    """    

    # check axes and create it if needed
    axes = __check_axes(ax)

    # plot tempo values
    axes.plot(bpms, **kwargs)

    xlabs = [x.replace('.1', '') if '.1' in x else ' ' for x in beat_labels]
    axes.set_xticks(range(bpms.shape[0]+1))
    axes.set_xticklabels(xlabs)

    axes.legend(loc='upper right')
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)

    return axes


def scape_plot(SP, Fs=1, ax=None, thr=0.9, mkr='X', xlabel='', ylabel='', **kwargs): 
    """Display scape plot

    Parameters
    ----------
    SP : np.ndarray
        scape plot data
    Fs : int
        sampling rate
    thr : float
        threshold used for text annotations
    mkr : string
        marker used for text annotations
    ax : matplotlib.axes.Axes or None
        Axes to plot on instead of the default `plt.gca()`.
    kwargs
        Additional keyword arguments to `matplotlib.`

    Returns
    -------
    axes
        The axis handle for the figure.
    axim
        The axis image for the figure.

    Notes
    -----
    This code is based on the Fundamentals of Music Processing notebooks [1].

    References
    ----------
    .. [1] Meinard Müller and Frank Zalkow
           libfmp: A Python Package for Fundamentals of Music Processing. 
           Journal of Open Source Software (JOSS), 6(63), 2021.
    """    

    # check axes and create it if needed
    axes = __check_axes(ax)

    N = SP.shape[0]
    SP_vis = np.zeros((N,N))
    
    # create scape plot data for display
    for length_minus_one in range(N):
        for start in range(N-length_minus_one):
            center = start + length_minus_one//2
            SP_vis[length_minus_one,center] = SP[length_minus_one,start]
            
    extent = np.array([-0.5, (N-1)+0.5, -0.5, (N-1)+0.5])/Fs  
    cmap_custom = copy.copy(cm.get_cmap("Spectral_r"))
    cmap_custom.set_bad('white')
    axim = axes.imshow(np.ma.masked_values(SP_vis, 0), cmap=cmap_custom,
                       aspect='auto', origin='lower', extent=extent, vmin=-1, vmax=1, **kwargs) 
    
    # loop over data dimensions and create text annotations
    for i in range(N):
        for j in range(N):
            if SP_vis[i, j] >= thr:
                 text = axes.text(j, i, mkr,
                                  ha="center", va="center", color="w")
    # plot borders
    x = np.asarray(range(N))
    x_half_lower = x/2
    x_half_upper = x/2 + N/2 - 1/2 
    axes.plot(x_half_lower/Fs, x/Fs+3/4, '-', linewidth=3, color='black')
    axes.plot(x_half_upper/Fs, np.flip(x, axis=0)/Fs, '-', linewidth=3, color='black')    
    axes.plot(x/Fs, np.zeros(N)/Fs, '-', linewidth=3, color='black')
    axes.set_xlim([0,(N-1)/Fs])
    axes.set_ylim([0,(N-1)/Fs])

    ticks = axes.get_xticks().tolist()
    axes.xaxis.set_major_locator(mticker.FixedLocator(ticks))
    new_ticks = [int(tick)+1 for tick in ticks]
    axes.set_xticklabels(new_ticks)
    ticks = axes.get_yticks().tolist()
    axes.yaxis.set_major_locator(mticker.FixedLocator(ticks))
    new_ticks = [int(tick)+1 for tick in ticks]
    axes.set_yticklabels(new_ticks)

    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)

    return axes, axim
