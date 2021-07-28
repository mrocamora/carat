#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=C0103
'''
 _  _  __ _ _|_
(_ (_| | (_| |_   computer-aided rhythm analysis toolbox

Compute microtiming pattern of onsets within a beat

'''

import sys
import argparse
import matplotlib.pyplot as plt
import carat

def compute_microtiming_pattern_for_beats(beat_annotations_file,
                                          onset_annotations_file,
                                          delimiter):
    '''Computation of the microtiming pattern of onsets within a beat

    :parameters:
      - beat_annotations_file : str
          path to the beat annotations file (txt, csv, etc)
      - onset_annotations_file : str
          path to the annotations file (txt, csv, etc)
      - delimiter: str
          delimiter string to process the annotations file
    '''

    # 1. load beat annotations
    print('Loading beat annotations ...', beat_annotations_file)
    beats, _ = carat.annotations.load_beats(beat_annotations_file, delimiter=delimiter)

    # 2. load onset annotations
    print('Loading onset annotations ...', onset_annotations_file)
    onsets, _ = carat.annotations.load_onsets(onset_annotations_file, delimiter=delimiter)

    # 3. compute beats from onsets
    print('Computing beats from onsets ...')
    beat_ons = carat.microtiming.beats_from_onsets(beats, onsets)

    # 4. normalize onsets
    print('Normalizing onsets ...')
    ons_norm = carat.microtiming.normalize_onsets(beat_ons, onsets)

    # 5. assigning onsets to metrical grid
    print('Assigning onsets to metrical grid ...')
    metrical_grid = carat.microtiming.define_metrical_grid()
    ons_in_grid = carat.microtiming.onsets_to_metrical_grid(ons_norm, metrical_grid)

    # 6. plot everything
    ax1 = plt.subplot(2, 1, 1)
    carat.display.onsets_in_grid_plot(ons_in_grid[0], ax=ax1, hist_ons=True)
    ax1.set_xlabel('Subdivision within the beat', fontsize=14)

    plt.show()


def process_arguments(args):
    '''Argparse function to get the program parameters'''

    parser = argparse.ArgumentParser(
        description=print(__doc__))

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

    # compute the microtiming pattern of onsets within a beat
    compute_microtiming_pattern_for_beats(parameters['beat_annotations_file'],
                                          parameters['onset_annotations_file'],
                                          parameters['delimiter'])
