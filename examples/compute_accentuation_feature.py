#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=C0103
'''
 _  _  __ _ _|_
(_ (_| | (_| |_   computer-aided rhythm analysis toolbox

Compute accentuation feature example

'''

import sys
import argparse
import matplotlib.pyplot as plt
from carat import audio, features, annotations, display

def compute_features(input_file, annotations_file, delimiter):
    '''Accentuation feature computation

    :parameters:
      - input_file : str
          path to input audio file (wav, mp3, m4a, flac, etc.)
      - annotations_file : str
          path to the annotations file (txt, csv, etc)
      - delimiter: str
          delimiter string to process the annotations file
    '''

    # 1. load the wav file
    print('Loading audio file ...', input_file)
    y, sr = audio.load(input_file, sr=None, duration=10.0)

    # 2. compute accentuation feature
    print('Computing accentuation feature ...')
    acce, times, _ = features.accentuation_feature(y, sr, minfreq=20, maxfreq=200)

    # 3. load beat annotations
    print('Loading beat annotations ...', annotations_file)
    beats, beat_labs = annotations.load_beats(annotations_file, delimiter=delimiter)

    # 4. plot everything
    # plot waveform
    ax1 = plt.subplot(2, 1, 1)
    display.wave_plot(y, sr, ax=ax1, beats=beats, beat_labs=beat_labs)
    # plot accentuation feature
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    display.feature_plot(acce, times, ax=ax2, beats=beats, beat_labs=beat_labs)

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

    return vars(parser.parse_args(args))


if __name__ == '__main__':
    # get the parameters
    parameters = process_arguments(sys.argv[1:])

    # run the accentuation feature computation
    compute_features(parameters['input_file'], parameters['annotations_file'],
                     parameters['delimiter'])
