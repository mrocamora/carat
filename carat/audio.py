# encoding: utf-8
# pylint: disable=C0103
"""Utility functions to deal with audio."""

import librosa

__all__ = ['load']

# simply use librosa.load (this may change in the future)
load = librosa.load
