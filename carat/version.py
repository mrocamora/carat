#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
"""Version info"""

import sys
import importlib

short_version = 'v0.1'
version = 'v0.1.5'


def __get_mod_version(modname):

    try:
        if modname in sys.modules:
            mod = sys.modules[modname]
        else:
            mod = importlib.import_module(modname)
        try:
            return mod.__version__
        except AttributeError:
            return 'installed, no version number available'

    except ImportError:
        return None


def show_versions():
    '''Return the version information for all carat dependencies.'''

    core_deps = ['audioread',
                 'numpy',
                 'scipy',
                 'sklearn']

    extra_deps = ['numpydoc',
                  'sphinx',
                  'matplotlib']

    print('INSTALLED VERSIONS')
    print('------------------')
    print('python: {}\n'.format(sys.version))
    print('carat: {}\n'.format(version))
    for dep in core_deps:
        print('{}: {}'.format(dep, __get_mod_version(dep)))
    print('')
    for dep in extra_deps:
        print('{}: {}'.format(dep, __get_mod_version(dep)))
    # pass
