# -*- coding: utf-8 -*-
# pylint: disable=C0103
"""
=========================
Plot accentuation feature
=========================

This example shows how to compute an accentuation feature from de audio waveform.
"""

# Code source: Martín Rocamora
# License: MIT

##############################################
# Imports
#   - matplotlib for visualization
#
import matplotlib.pyplot as plt
import carat

##############################################
# The accentuation feature is based on the Spectral flux,
# that consists in seizing the changes in the spectral magnitude
# of the audio signal along different frequency bands.
# In principle, the feature value is high when a note has been
# articulated and close to zero otherwise.
#
# First, we'll load one of the audio files included in `carat`.
# We get the path to the audio file example number  1, and load 10 seconds of the file.
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
# Then, we'll compute the accentuation feature.
#
# **Note:** This example is tailored towards the rhythmic patterns of the lowest
# sounding of the three drum types taking part in the recording, so the analysis
# focuses on the low frequencies (20 to 200 Hz).
acce, times, _ = carat.features.accentuation_feature(y, sr, minfreq=20, maxfreq=200)

##############################################
# Finally we plot the audio waveform, the beat annotations and the accentuation feature values.

# plot waveform and accentuation feature
plt.figure(figsize=(12, 6))
# plot waveform
ax1 = plt.subplot(2, 1, 1)
carat.display.wave_plot(y, sr, ax=ax1, beats=beats, beat_labs=beat_labs)
# plot accentuation feature
ax2 = plt.subplot(2, 1, 2, sharex=ax1)
carat.display.feature_plot(acce, times, ax=ax2, beats=beats, beat_labs=beat_labs)
plt.tight_layout()

plt.show()
