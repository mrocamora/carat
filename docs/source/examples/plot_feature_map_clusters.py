# -*- coding: utf-8 -*-
# pylint: disable=C0103
"""
=========================
Plot feature map clusters
=========================

This example shows how to cluster rhythmic patterns from a feature map.

This is based on the rhythmic patterns analysis proposed in [CIM2014]_.
"""

# Code source: Martín Rocamora
# License: MIT

##############################################
# Imports
#   - matplotlib for visualization
#
import matplotlib.pyplot as plt
from carat import util, audio, display, annotations, features, clustering

##############################################
# We group rhythmic patterns into clusters to aid the analysis
# of their differences and similarities.
#
# First, we'll load one of the audio files included in `carat`.
audio_path = util.example_audio_file(num_file=1)

y, sr = audio.load(audio_path)

##############################################
# Next, we'll load the annotations provided for the example audio file.
annotations_path = util.example_beats_file(num_file=1)

beats, beat_labs = annotations.load_beats(annotations_path)
downbeats, downbeat_labs = annotations.load_downbeats(annotations_path)

##############################################
# Then, we'll compute the accentuation feature.
#
# **Note:** This example is tailored towards the rhythmic patterns of the lowest
# sounding of the three drum types taking part in the recording, so the analysis
# focuses on the low frequencies (20 to 200 Hz).
acce, times, _ = features.accentuation_feature(y, sr, minfreq=20, maxfreq=200)

##############################################
# Next, we'll compute the feature map.
n_beats = int(round(beats.size/downbeats.size))
n_tatums = 4

map_acce, _, _, _ = features.feature_map(acce, times, beats, downbeats, n_beats=n_beats,
                                         n_tatums=n_tatums)

##############################################
# Then, we'll group rhythmic patterns into clusters. This is done using the classical
# K-means method with Euclidean distance (but other clustering methods and distance
# measures can be used too).
#
# **Note:** The number of clusters n_clusters has to be specified as an input parameter.
n_clusters = 4

cluster_labs, centroids, _ = clustering.rhythmic_patterns(map_acce, n_clusters=n_clusters)

##############################################
# Finally we plot the feature map and the obtained clusters.

plt.figure(figsize=(12, 6))
# plot feature map
ax1 = plt.subplot(211)
display.map_show(map_acce, ax=ax1, n_tatums=n_tatums)
# plot feature map with clusters in colors
ax2 = plt.subplot(212)
display.map_show(map_acce, ax=ax2, n_tatums=n_tatums, clusters=cluster_labs)

plt.show()
