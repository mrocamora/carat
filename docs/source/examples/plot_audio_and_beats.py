# -*- coding: utf-8 -*-
# pylint: disable=C0103
"""
====================
Plot audio and beats
====================

This example shows how to load/plot an audio file and the corresponding beat annotations file.
"""

# Code source: Martín Rocamora
# License: MIT

##############################################
# Imports
#   - matplotlib for visualization
#
from __future__ import print_function
import matplotlib.pyplot as plt
import carat

##############################################
# First, we'll load one of the audio files included in `carat`.
# We get the path to the audio file example number  1, and load 10 seconds of the file.
#
# **Note 1:** By default, `carat` will resample the signal to 22050Hz, but this can disabled
# by saying `sr=None` (`carat` uses librosa for loading audio files, so it inherits
# all its functionality and behaviour).
audio_path = carat.util.example_audio_file(num_file=1)

y, sr = carat.audio.load(audio_path, duration=10.0)

##############################################
# Next, we'll load the annotations provided for the example audio file.
# We get the path to the annotations file corresponding to example number 1,
# and then we load beats and downbeats, along with their labels.
annotations_path = carat.util.example_beats_file(num_file=1)

beats, beat_labs = carat.annotations.load_beats(annotations_path)
downbeats, downbeat_labs = carat.annotations.load_downbeats(annotations_path)

##############################################
# **Note 2:** It is assumed that the beat annotations are provided as a text file (csv).
# Apart from the time data (mandatory) a label can be given for each beat (optional).
# The time data is assumed to be given in seconds. The labels may indicate the beat number
# within the rhythm cycle (e.g. 1.1, 1.2, or 1, 2).
#
# **Note 3:** The same annotations file is used for both beats and downbeats.
# This is based on annotation labels that provide a particular string to identify the downbeats.
# In this case, this string is .1, and is the one used by default. You can specify the string to
# look for in the labels data to select downbeats by setting the `downbeat_label` parameter value.
# For instance, `downbeat_label='1'` is used for loading annotations of the samba files included.
#
# **Note 4:** By default the columns are assumed to be separated by a comma, but you can specify
# another separating string by setting the `delimiter` parameter value. For instance, a blank space
# `delimiter=' '` is used for loading annotations of the samba files included.
#
# Let's print the first 10 beat and the first 3 downbeats, with their corresponding labels.
print(beats[:10])
print(beat_labs[:10])

print(downbeats[:3])
print(downbeat_labs[:3])

##############################################
# Finally we plot the audio waveform and the beat annotations

plt.figure(figsize=(12, 6))
ax1 = plt.subplot(2, 1, 1)
carat.display.wave_plot(y, sr, ax=ax1)
ax2 = plt.subplot(2, 1, 2, sharex=ax1)
carat.display.wave_plot(y, sr, ax=ax2, beats=downbeats, beat_labs=downbeat_labs)
plt.tight_layout()

plt.show()
