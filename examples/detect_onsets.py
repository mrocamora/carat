#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=C0103
'''
 _  _  __ _ _|_
(_ (_| | (_| |_   computer-aided rhythm analysis toolbox

Onset detection example

'''

import sys
import argparse
import matplotlib.pyplot as plt
from carat import annotations, audio, display, onsets


def onset_detection(input_file, output_file, threshold, freqs, log_flag):
    '''Onset detection

    :parameters:
      - input_file : str
          path to input audio file (wav, mp3, m4a, flac, etc.)
      - annotations_file : str
          path to the annotations file (txt, csv, etc)
      - delimiter: str
          delimiter string to process the annotations file
    '''

    # 1. load the wav file
    # use an example audio file provided
    print('Loading audio file ...', input_file)
    y, sr = audio.load(input_file, sr=None, duration=30.0)

    # 2. values for compute accentuation feature
    hop = 5e-3            # hop size
    nfilts = 80           # Number of MEL filters
    alpha = 10e4          # compression parameter for dB conversion - log10(alpha*abs(S)+1)

    # 3. values for peak detection on the accentuation feature function
    pre_avg = 14      # number of past frames for moving average
    pos_avg = 10      # number of future frames for moving average
    pre_max = 14      # number of past frames for moving maximum
    pos_max = 10      # number of future frames for moving maximum

    # 4. onsets detection
    print('Detecting onsets')
    onset_times, feature_values = onsets.detection(y, fs=sr, hop=hop, nfilts=nfilts,
                                                   log_flag=log_flag, alpha=alpha, minfreq=freqs[0],
                                                   maxfreq=freqs[1], threshold=threshold,
                                                   pre_avg=pre_avg, pos_avg=pos_avg, 
                                                   pre_max=pre_max, pos_max=pos_max)

    # 5. plot everything
    plt.plot()
    display.wave_plot(y, sr, onsets=onset_times)
    plt.title('detected onsets')
    plt.show()

    # 6. save onsets to csv
    print('Saving onset annotations ...', output_file)
    annotations.save_onsets(output_file, onset_times)


def process_arguments(args):
    '''Argparse function to get the program parameters'''

    parser = argparse.ArgumentParser(
        description=print(__doc__))

    parser.add_argument('input_file',
                        action='store',
                        help='path to the input file (wav, mp3, etc)')
    parser.add_argument('output_file',
                        action='store',
                        help='path to the output csv file')
    parser.add_argument('--threshold', '-t',
                        required=True,
                        type=float,
                        help='threshold for peak picking')
    parser.add_argument('--frequencies', '-f',
                        required=True,
                        type=int,
                        nargs=2,
                        help='bound frequencies for summing frequency band')
    parser.add_argument('--log', '-l', action='store_true',
                        help='If LOG should be taken before taking differentiation')

    return vars(parser.parse_args(args))


if __name__ == '__main__':
    # get the parameters
    parameters = process_arguments(sys.argv[1:])

    # run the detection of onsets
    onset_detection(parameters['input_file'], parameters['output_file'],
                    parameters['threshold'], parameters['frequencies'],
                    parameters['log'])
