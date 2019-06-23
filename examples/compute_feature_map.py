#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=C0103
'''
 _  _  __ _ _|_
(_ (_| | (_| |_   computer-aided rhythm analysis toolbox

Compute accentuation feature map example

'''

from __future__ import print_function

import sys
import argparse
import matplotlib.pyplot as plt
import carat

def compute_feature_map(input_file, annotations_file, delimiter, downbeat_label,
                        n_tatums, n_clusters):
    '''Accentuation feature computation

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

    # 1. load the wav file and resample to 22.050 KHz
    print('Loading audio file ...', input_file)
    y, sr = carat.audio.load(input_file)#, duration=20.0)

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
                                                         # maxfreq=None,
                                                         # log_flag=True)

    # 4. compute feature map
    print('Computing feature map ...')
    map_acce, _, _, _ = carat.features.feature_map(acce, times, beats, downbeats,
                                                   n_beats=n_beats, n_tatums=n_tatums)

    # 5. cluster rhythmic patterns
    print('Clustering rhythmic patterns ...')
    clusters = carat.clustering.rhythmic_patterns(map_acce, n_clusters=n_clusters)

    # 6. plot everything
    plt.figure()
    ax1 = plt.subplot(211)
    carat.display.mapshow(map_acce, ax=ax1, n_tatums=n_tatums)

    ax2 = plt.subplot(212)
    carat.display.mapshow(map_acce, ax=ax2, n_tatums=n_tatums, clusters=clusters)

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

    # run the feature map computation
    compute_feature_map(parameters['input_file'], parameters['annotations_file'],
                        parameters['delimiter'], parameters['downbeat'],
                        parameters['n_tatums'], parameters['n_clusters'])
