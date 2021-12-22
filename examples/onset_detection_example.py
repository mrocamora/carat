#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=C0103
'''
 _  _  __ _ _|_
(_ (_| | (_| |_   computer-aided rhythm analysis toolbox

Onset detection example

'''

import os, sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from carat import audio, features, annotations, display, onsets, util

def onset_detection():
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
    input_file = util.example("chico_audio")
    print('Loading audio file ...', input_file)
    y, sr = audio.load(input_file, sr=None, duration=30.0)

    # 2. compute accentuation feature
    print('Computing accentuation feature ...')
    hop = 5e-3            # hop size
    nfilts = 80           # Number of MEL filters
    log_flag = True       # If LOG should be taken before taking differentiation
    alpha = 10e4          # compression parameter for dB conversion - log10(alpha*abs(S)+1)
    freqs = [500, 3000]   # chico bound frequencies for summing frequency band

    acce, times, _ = features.accentuation_feature(y, sr, hop=hop, nfilts=nfilts, log_flag=log_flag, alpha=alpha,
                                                   minfreq=freqs[0], maxfreq=freqs[1])

    # 3. peak detection on the accentuation feature function
    print('Peak picking on the accentuation feature ...')
    threshd = 0.180   # threshold for peak-picking (chico)
    pre_avg = 14      # number of past frames for moving average
    pos_avg = 10      # number of future frames for moving average
    pre_max = 14      # number of past frames for moving maximum
    pos_max = 10      # number of future frames for moving maximum

    peak_indxs, mov_avg, mov_max = features.peak_detection(acce, threshold=threshd, 
                                                           pre_avg=pre_avg, pos_avg=pos_avg, 
                                                           pre_max=pre_max, pos_max=pos_max)

    # time instants of the onsets
    onset_times = times[peak_indxs]

    # 4. load onset annotations
    # use onset annotations provided for the example audio file
    onset_annotations_file = util.example("chico_onsets")
    print('Loading onset annotations ...', onset_annotations_file)
    # load onset annotations
    onsets_annot, _ = annotations.load_onsets(onset_annotations_file)

    # 5. onsets detection with only one function
    onsets_all, _ = onsets.detection(y, fs=sr, hop=hop, nfilts=nfilts, log_flag=log_flag, alpha=alpha,
                                     minfreq=freqs[0], maxfreq=freqs[1], threshold=threshd,
                                     pre_avg=pre_avg, pos_avg=pos_avg, pre_max=pre_max, pos_max=pos_max)

    # check if the onsets obtained are the same
    np.testing.assert_allclose(onset_times, onsets_all)

    # 6. plot everything
    # plot waveform
    ax1 = plt.subplot(2, 1, 1)
    display.wave_plot(y, sr, ax=ax1, onsets=onsets_annot)
    plt.title('annotated onsets')
    # plot accentuation feature
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    display.feature_plot(acce, times, ax=ax2, onsets=onset_times)
    plt.title('detected onsets')

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
    # parameters = process_arguments(sys.argv[1:])

    # run the detection of onsets
    onset_detection()
