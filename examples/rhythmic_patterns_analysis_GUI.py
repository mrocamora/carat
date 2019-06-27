#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=C0103
'''
 _  _  __ _ _|_
(_ (_| | (_| |_   computer-aided rhythm analysis toolbox

Rhythmic patterns analysis example with GUI

'''

from __future__ import print_function

import os
import sys
import argparse
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.ticker import NullFormatter
from mpl_toolkits.mplot3d import Axes3D
import sounddevice as sd
import carat


class AudioSignal:
    """
    Audio signal to play rhythmic patterns
    """
    def __init__(self, y, sr, downbeats):

        # set sample rate
        sd.default.samplerate = sr

        # set samples
        self.samples = y

        # set samples
        self.time = np.arange(y.size)/sr

        # set downbeats
        self.downbeats = downbeats


    def play(self, ind):
        """ Play audio corresponding to bar given index ind
        """

        # get the audio segment
        bar_segment = carat.util.beat2signal(self.samples,
                                             self.time,
                                             self.downbeats,
                                             ind)

        # play audio sound
        sd.play(bar_segment)


class PointBrowser:
    """
    Click on a point to select and highlight it. The location of the
    pattern that generated the point will be shown in the lower map.
    Use 'n' and 'b' keys to browse through the next and previous pattern.
    """
    def __init__(self, audio, data, emb, clusters, centroids, num_bars,
                 n_tatums, fig, ax1, ax2, ax3s):

        # initialize last selected index
        self.lastind = -1
        # save audio reference
        self.audio = audio
        # save data reference
        self.data = data
        # save embeddin reference
        self.emb = emb
        # save clusters reference
        self.clusters = clusters
        # save centroids reference
        self.centroids = centroids
        # save number of bars
        self.num_bars = num_bars
        # save number of tatums
        self.n_tatums = n_tatums

        # save fig and axes
        self.fig = fig
        self.ax1 = ax1
        self.ax2 = ax2
        self.ax3s = ax3s

        # set current event
        self.event = None

        # initialize text box
        props = dict(boxstyle='round', facecolor='0.95')  #, alpha=0.5)
        self.text = plt.figtext(0.55, 0.36, 'bar : none', bbox=props, ha='left')

    def playback(self, event):
        """ playback button
        """
        self.event = event
        self.audio.play(self.lastind)

    def onpress(self, event):
        """ Process key press ('n' = next, 'b' = back)
        """
        self.event = event
        # test if a bar was previously selected
        if self.lastind is None:
            return
        # if space is pressed play current bar
        if event.key == ' ':
            self.audio.play(self.lastind)
        else:
            # test if 'n' or 'b' is the key pressed
            if event.key not in ('n', 'b'):
                return
    	    # set increment according to key pressed
            if event.key == 'n':
                inc = 1
            else:
                inc = -1

    	    # add increment to last index
            self.lastind += inc
    	    # limit the index scope between 0 and numb_bars-1
            self.lastind = np.clip(self.lastind, 0, self.num_bars-1)

    	    # update figure
            self.update()

    def onpick(self, event):
        """ process click on a point
        """
        self.event = event
        # get index of selected pattern
        self.lastind = event.ind[0]
        # update figure
        self.update()


    def onclick(self, event):
        """ process click on map
        """
        self.event = event
        # see if the mouse is over the map
        if event.inaxes != self.ax2.axes:
            return

        # get index of selected pattern
        self.lastind = int(event.xdata-1)
        # update figure
        self.update()


    def update(self):
        """ update figures
        """
        if self.lastind is None:
            return

        # get current selected pattern index
        dataind = self.lastind

        # clear and redraw scatter plot, highliting selected pattern
        self.ax1.cla()
        carat.display.embedding_plot(self.emb, ax=self.ax1, clusters=self.clusters)
        self.ax1.scatter(self.emb[dataind, 0], self.emb[dataind, 1], self.emb[dataind, 2],
                         c='black', s=100, alpha=0.3, picker=2)
        self.ax1.xaxis.set_major_formatter(NullFormatter())
        self.ax1.yaxis.set_major_formatter(NullFormatter())
        self.ax1.zaxis.set_major_formatter(NullFormatter())

        # update xaxis ticks in map to indicate selected pattern
        self.ax2.xaxis.set_ticks([dataind+1])

        # change text message to show selected pattern
        self.text.set_text('bar : %4d'%(dataind+1))

        # update centroid plot adding current selected pattern
        for ax3 in self.ax3s:
            ax3.cla()
        carat.display.centroids_plot(self.centroids, n_tatums=self.n_tatums,
                                     ax_list=self.ax3s)
        for ind, ax3 in enumerate(self.ax3s):
            if ind == self.clusters[dataind]:
                carat.display.plot_centroid(self.data[dataind, :], n_tatums=self.n_tatums,
                                            ax=ax3, color='gray', alpha=0.4, hatch='\\')
        # draw the canvas
        self.fig.canvas.draw()


def rhythmic_patterns_analysis(input_file, annotations_file, delimiter,
                               downbeat_label, n_tatums, n_clusters):
    '''Rhythmic patterns analysis

    :parameters:
      - input_file : str
          path to input audio file (wav, mp3, m4a, flac, etc.)
      - annotations_file : str
          path to the annotations file (txt, csv, etc)
      - delimiter: str
          delimiter string to process the annotations file
      - downbeat_label: str
          string to look for in the label data to select downbeats
      - n_tatums: int
          number of tatums (subdivisions) per tactus beat
      - n_clusters: int
          number of clusters for rhythmic patterns clustering
    '''

    # =================== MAIN PROCESSING ===================

    # 1. load the wav file
    print('Loading audio file ...', input_file)
    y, sr = carat.audio.load(input_file, sr=None)

    # 2. load beat and downbeat annotations
    print('Loading beat and downbeat annotations ...', annotations_file)
    beats, _ = carat.annotations.load_beats(annotations_file, delimiter=delimiter)
    downbeats, _ = carat.annotations.load_downbeats(annotations_file, delimiter=delimiter,
                                                    downbeat_label=downbeat_label)
    # number of beats per bar
    n_beats = int(round(beats.size/downbeats.size))

    # 3. compute accentuation feature
    print('Computing accentuation feature ...')
    acce, times, _ = carat.features.accentuation_feature(y, sr, minfreq=20, maxfreq=200)

    # 4. compute feature map
    print('Computing feature map ...')
    map_acce, _, _, _ = carat.features.feature_map(acce, times, beats, downbeats,
                                                   n_beats=n_beats, n_tatums=n_tatums)

    # 5. cluster rhythmic patterns
    print('Clustering rhythmic patterns ...')
    cluster_labs, centroids, _ = carat.clustering.rhythmic_patterns(map_acce, n_clusters=n_clusters)

    # 6. manifold learning
    print('Dimensionality reduction ...')
    map_emb = carat.clustering.manifold_learning(map_acce)


    # =================== PLOTS AND GUI ===================

    mp.rcParams['toolbar'] = 'None'
    fig = plt.figure(figsize=(12, 8), facecolor='white')
    fig.canvas.set_window_title('carat - rhythmic patterns analysis')
    plt.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.03)

    # plot low-dimensional embedding of feature data
    ax1 = plt.subplot2grid((6, 3), (0, 0), colspan=2, rowspan=4, projection='3d')
    carat.display.embedding_plot(map_emb, ax=ax1, clusters=cluster_labs)

    # plot feature map with clusters in colors
    ax2 = plt.subplot2grid((6, 3), (4, 0), colspan=3, rowspan=2)
    carat.display.map_show(map_acce, ax=ax2, n_tatums=n_tatums, clusters=cluster_labs)

    # plot cluster centroids
    ax3s = []
    for ind in range(n_clusters):
        ax3 = plt.subplot2grid((6, 3), (ind, 2))
        ax3s.append(ax3)
    carat.display.centroids_plot(centroids, n_tatums=n_tatums, ax_list=ax3s)

    # additional configuration for GUI
    plt.figtext(0.03, 0.95, 'filename: ' + os.path.basename(input_file))
    plt.figtext(0.55, 0.42, '   n : next')
    plt.figtext(0.55, 0.40, '   b : back')

    # load audiofile for playback
    audio = AudioSignal(y, sr, downbeats)

    axplay = plt.axes([0.03, 0.35, 0.05, 0.04])
    bplay = Button(axplay, 'play', color='0.95', hovercolor='grey')

    browser = PointBrowser(audio, map_acce, map_emb, cluster_labs, centroids,
                           downbeats.size, n_tatums, fig, ax1, ax2, ax3s)
    fig.canvas.mpl_connect('pick_event', browser.onpick)
    fig.canvas.mpl_connect('key_press_event', browser.onpress)
    fig.canvas.mpl_connect('button_press_event', browser.onclick)

    bplay.on_clicked(browser.playback)

    plt.show()


def process_arguments(args):
    '''Argparse function to get the program parameters'''

    parser = argparse.ArgumentParser(
        description=print(__doc__))

    parser.add_argument('input_file',
                        action='store',
                        help='path to the input file (wav, mp3, etc)')
    parser.add_argument('annotations_file',
                        action='store',
                        help='path to the annotations file (txt, csv, etc)')
    parser.add_argument('-d', '--delimiter',
                        help='delimiter string to process the annotations file',
                        metavar="delimiter_str", default=',', type=str, action='store')
    parser.add_argument('-b', '--downbeat',
                        help='string to look for in the label data to select downbeats',
                        metavar="downbeat_str", default='.1', type=str, action='store')
    parser.add_argument('-t', '--n_tatums',
                        help='number of tatums (subdivisions) per tactus beat',
                        metavar="n_tatums_int", default='4', type=int, action='store')
    parser.add_argument('-c', '--n_clusters',
                        help='number of clusters for rhythmic patterns clustering',
                        metavar="n_clusters_int", default='4', type=int, action='store')

    return vars(parser.parse_args(args))


if __name__ == '__main__':
    # get the parameters
    parameters = process_arguments(sys.argv[1:])

    # run the rhythmic patterns analysis
    rhythmic_patterns_analysis(parameters['input_file'], parameters['annotations_file'],
                               parameters['delimiter'], parameters['downbeat'],
                               parameters['n_tatums'], parameters['n_clusters'])
