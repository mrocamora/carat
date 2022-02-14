#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=C0103
'''
 _  _  __ _ _|_
(_ (_| | (_| |_   computer-aided rhythm analysis toolbox

Onset detection example using a preset for parameter values

'''

import sys
import argparse
import os
import matplotlib.pyplot as plt
from carat import annotations, audio, display, onsets, util


def onset_detection(input_file, output_file, preset_name,
                    json_file, verbose):
    '''Onset detection

    :parameters:
      - input_file : str
          path to input audio file (wav, mp3, m4a, flac, etc.)
      - output_file : str
          path to the output csv file
      - preset_name : str
            name of the preset to load (e.g. "chico")
      - json_file : str
           path to the json file of presets
    '''

    # 1. load the wav file
    # use an example audio file provided
    print('Loading audio file ...', input_file)
    y, sr = audio.load(input_file, sr=None)

    # 2. load preset values
    print('Loading preset parameter values ...', json_file)
    preset = util.load_preset(json_file, preset_name)
    if verbose:
        print('Values are:', preset)

    # 3. onsets detection
    print('Detecting onsets')
    onset_times, feature_values = onsets.detection(y, fs=sr, **preset)

    # 4. plot everything
    plt.plot()
    display.wave_plot(y, sr, onsets=onset_times)
    plt.title('detected onsets')
    plt.show()

    # 5. save onsets to csv
    print('Saving onset annotations ...', output_file)
    annotations.save_onsets(output_file, onset_times)


def process_arguments(args, dirname):
    '''Argparse function to get the program parameters'''

    parser = argparse.ArgumentParser(
        description=print(__doc__))

    parser.add_argument('input_file',
                        action='store',
                        help='path to the input file (wav, mp3, etc)')
    parser.add_argument('output_file',
                        action='store',
                        help='path to the output csv file')
    parser.add_argument('preset_name',
                        action='store',
                        help='name of the preset to load (e.g. chico)')
    parser.add_argument('--json_file',
                        action='store',
                        help='path to the json file of presets',
                        default=os.path.join(dirname,'../carat/presets/onsets.json'),
                        required=False)
    parser.add_argument('--verbose',
                        action='store',
                        help='path to the json file of presets',
                        default=True,
                        required=False)

    return vars(parser.parse_args(args))


if __name__ == '__main__':
    # get path relative to script (and not execution)
    dirname = os.path.dirname(__file__)

    # get the parameters
    parameters = process_arguments(sys.argv[1:], dirname)

    # run the detection of onsets
    onset_detection(parameters['input_file'], parameters['output_file'],
                    parameters['preset_name'], parameters['json_file'],
                    parameters['verbose'])
