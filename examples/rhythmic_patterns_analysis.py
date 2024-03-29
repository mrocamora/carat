#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=C0103
'''
 _  _  __ _ _|_
(_ (_| | (_| |_   computer-aided rhythm analysis toolbox

Rhythmic patterns analysis example

'''

import sys
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from carat import audio, annotations, features, clustering, display

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

    # 1. load the wav file
    print('Loading audio file ...', input_file)
    y, sr = audio.load(input_file, sr=None)

    # 2. load beat and downbeat annotations
    print('Loading beat and downbeat annotations ...', annotations_file)
    beats, _ = annotations.load_beats(annotations_file, delimiter=delimiter)
    downbeats, _ = annotations.load_downbeats(annotations_file, delimiter=delimiter,
                                                    downbeat_label=downbeat_label)
    # number of beats per bar
    n_beats = int(round(beats.size/downbeats.size))

    # 3. compute accentuation feature
    print('Computing accentuation feature ...')
    acce, times, _ = features.accentuation_feature(y, sr, minfreq=20, maxfreq=200)

    # 4. compute feature map
    print('Computing feature map ...')
    map_acce, _, _, _ = features.feature_map(acce, times, beats, downbeats,
                                                   n_beats=n_beats, n_tatums=n_tatums)

    # 5. cluster rhythmic patterns
    print('Clustering rhythmic patterns ...')
    cluster_labs, centroids, _ = clustering.rhythmic_patterns(map_acce, n_clusters=n_clusters)

    # 6. manifold learning
    print('Dimensionality reduction ...')
    map_emb = clustering.manifold_learning(map_acce)

    # 7. plot everything
    plt.figure()

    ax1 = plt.subplot(211)
    # plot feature map
    display.map_show(map_acce, ax=ax1, n_tatums=n_tatums)

    ax2 = plt.subplot(212)
    # plot feature map with clusters in colors
    display.map_show(map_acce, ax=ax2, n_tatums=n_tatums, clusters=cluster_labs)

    fig = plt.figure()
    ax3 = fig.add_subplot(111, projection='3d')
    # plot low-dimensional embedding of feature data
    display.embedding_plot(map_emb, ax=ax3, clusters=cluster_labs)

    fig = plt.figure()
    display.centroids_plot(centroids, n_tatums=n_tatums)

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
