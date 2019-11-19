#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=C0103
'''
 _  _  __ _ _|_
(_ (_| | (_| |_   computer-aided rhythm analysis toolbox

Align the location of beats to the location of onsets closest to the beats

'''

from __future__ import print_function

import sys
import argparse
import matplotlib.pyplot as plt
import carat

def align_beats_to_onsets(audio_file, beat_annotations_file, onset_annotations_file, delimiter):
    '''Align the location of beats to the location of onsets closest to the beats

    :parameters:
      - audio_file : str
          path to input audio file (wav, mp3, m4a, flac, etc.)
      - beat_annotations_file : str
          path to the beat annotations file (txt, csv, etc)
      - onset_annotations_file : str
          path to the annotations file (txt, csv, etc)
      - delimiter: str
          delimiter string to process the annotations file
    '''

    # 1. load the wav file
    print('Loading audio file ...', audio_file)
    y, sr = carat.audio.load(audio_file, sr=None, duration=30.0)

    # 2. load beat annotations
    print('Loading beat annotations ...', beat_annotations_file)
    beats, _ = carat.annotations.load_beats(beat_annotations_file, delimiter=delimiter)

    # 3. load onset annotations
    print('Loading onset annotations ...', onset_annotations_file)
    onsets, _ = carat.annotations.load_onsets(onset_annotations_file, delimiter=delimiter)

    # 4. compute beats from onsets
    print('Computing beats from onsets ...')
    beat_ons = carat.microtiming.beats_from_onsets(beats, onsets)

    # 5. plot everything
    ax1 = plt.subplot(2, 1, 1)
    carat.display.wave_plot(y, sr, ax=ax1, beats=beats)
    # plot aligned beats
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    carat.display.wave_plot(y, sr, ax=ax2, beats=beat_ons)


    plt.show()


def process_arguments(args):
    '''Argparse function to get the program parameters'''

    parser = argparse.ArgumentParser(
        description=print(__doc__))

    parser.add_argument('audio_file',
                        action='store',
                        help='path to the audio file (wav, mp3, etc)')
    parser.add_argument('beat_annotations_file',
                        action='store',
                        help='path to the beat annotations file (txt, csv, etc)')
    parser.add_argument('onset_annotations_file',
                        action='store',
                        help='path to the onset annotations file (txt, csv, etc)')
    parser.add_argument('-d', '--delimiter',
                        help='delimiter string to process the annotations file',
                        metavar="delimiter_str", default=',', type=str, action='store')

    return vars(parser.parse_args(args))


if __name__ == '__main__':
    # get the parameters
    parameters = process_arguments(sys.argv[1:])

    # run the alignment of beats to onsets
    align_beats_to_onsets(parameters['audio_file'],
                          parameters['beat_annotations_file'],
                          parameters['onset_annotations_file'],
                          parameters['delimiter'])
