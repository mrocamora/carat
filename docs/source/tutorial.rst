Tutorial
^^^^^^^^

This section covers some fundamentals of using *carat*, including a package overview and basic usage. 
We assume basic familiarity with Python and NumPy/SciPy.


Overview
~~~~~~~~

The *carat* package is structured as a collection of submodules:

  - carat

    - :ref:`carat.annotations <annotations>`
        Functions for loading annotations files, such as beat annotations.
        
    - :ref:`carat.display <display>`
        Visualization and display routines using `matplotlib`.  

    - :ref:`carat.clustering <clustering>`
        Functions for clustering and low-dimensional embedding.

    - :ref:`carat.features <features>`
        Feature extraction and manipulation.

    - :ref:`carat.util <util>`
        Helper utilities.


.. _quickstart:

Quickstart
~~~~~~~~~~
The following is a brief example program for rhythmic patterns analysis using carat. 

It is based on the rhythmic patterns analysis proposed in [CIM2014]_

.. code-block:: python
    :linenos:

    '''
     _  _  __ _ _|_
    (_ (_| | (_| |_   computer-aided rhythm analysis toolbox

    Rhythmic patterns analysis example

    '''
    import carat

    # 1. Get the file path to an included audio example
    #    This is a recording of an ensemble of candombe drums
    audio_path = carat.util.example_audio_file(num_file=1)

    # 2. Get the file path the annotations for the example audio file
    annotations_path = carat.util.example_beats_file(num_file=1)

    # 3. Load the audio waveform `y` and its sampling rate as `sr`
    y, sr = carat.audio.load(audio_path, sr=None)

    # 4. Load beats/downbeats time instants and beat/downbeats labels
    beats, beat_labs = carat.annotations.load_beats(annotations_path)
    downbeats, downbeat_labs = carat.annotations.load_downbeats(annotations_path)

    # 5. Compute an accentuation feature indicating when a note has been articulated 
    #    We focus on the low frequency band (20 to 200 Hz) to get low sounding drum events
    acce, times, _ = carat.features.accentuation_feature(y, sr, minfreq=20, maxfreq=200)
     
    # 6. Compute a feature map of the rhythmic patterns
    # number of beats per bar
    n_beats = int(round(beats.size/downbeats.size))
    # you have to provide the number of tatums (subdivisions) per beat
    n_tatums = 4
    # compute the feature map from the feature signal and the beat/dowbeat annotations
    map_acce, _, _, _ = carat.features.feature_map(acce, times, beats, downbeats, 
                                                   n_beats=n_beats, n_tatums=n_tatums)


    # 7. Group rhythmic patterns into clusters
    # set the number of clusters to look for
    n_clusters = 4
    # clustering of rhythmic patterns
    cluster_labs, centroids, _ = carat.clustering.rhythmic_patterns(map_acce, n_clusters=n_clusters)


The first step of the program::

    audio_path = carat.util.example_audio_file(num_file=1)

gets the path to an audio example file included with *carat*.  After this step,
``audio_path`` will be a string variable containing the path to the example audio file.

Similarly, the following line::

    annotations_path = carat.util.example_beats_file(num_file=1)

gets the path to the annotations file for the same example.

The second step::

    y, sr = carat.audio.load(audio_path)
    
loads and decodes the audio as a ``y``, represented as a one-dimensional
NumPy floating point array.  The variable ``sr`` contains the sampling rate of
``y``, that is, the number of samples per second of audio.  By default, all audio is
mixed to mono and resampled to 22050 Hz at load time.  This behavior can be overridden
by supplying additional arguments to ``carat.audio.load()``.

Next, we load the annotations::

    beats, beat_labs = carat.annotations.load_beats(annotations_path)
    downbeats, downbeat_labs = carat.annotations.load_downbeats(annotations_path)

The ``beats`` are a one-dimensional Numpy array representing the time location of beats, and
``beat_labs`` is a list of ``string`` elements that correspond to the labels given for each beat.
This is the same for ``downbeats`` and ``downbeat_labs``, except that they correspond to downbeats.

Then, we compute an accentuation feature from the audio waveform:: 

    acce, times, _ = carat.features.accentuation_feature(y, sr, minfreq=20, maxfreq=200)

This is based on the Spectral flux, that consists in seizing the changes in the spectral magnitude
of the audio signal along different frequency bands. In principle, the feature value is high when
a note has been articulated and close to zero otherwise. Note that this example is tailored towards
the rhythmic patterns of the lowest sounding of the three drum types taking part in the recording,
so the analysis focuses on the low frequencies (20 to 200 Hz).

The feature values are stored in the one-dimensional Numpy array ``acce``, and the time instants
corresponding to each feature value are given in ``times``, which is also a one-dimensional Numpy array.

Next, we compute the feature map from the feature signal and the beat/downbeat annotations::

    n_beats = int(round(beats.size/downbeats.size))
    n_tatums = 4
    map_acce, _, _, _ = carat.features.feature_map(acce, times, beats, downbeats, 
                                                   n_beats=n_beats, n_tatums=n_tatums)

Note that we have to provide the beats and the downbeats, which were
loaded from the annotations. Besides, the number of beats per bar and the number of of tatums
(subdivisions) per beat has to be provided.

In this step the accentuation feature is organized into a feature map. First, the feature signal is
time-quantized to the rhythm metric structure by considering a grid of tatum pulses equally distributed
within the annotated beats. The corresponding feature value is taken as the maximum within window
centered at the frame closest to each tatum instant. This yields feature vectors whose coordinates
correspond to the tatum pulses of the rhythm cycle (or bar). Finally, a feature map of the
cycle-length rhythmic patterns of the audio file is obtained by building a matrix whose columns are
consecutive feature vectors, and stored in ``map_acce`` as a Numpy array matrix.

Finally, the rhythmic patterns of the feature map are grouped into clusters::

    n_clusters = 4
    cluster_labs, centroids, _ = carat.clustering.rhythmic_patterns(map_acce, n_clusters=n_clusters)

Note that the number of clusters ``n_clusters`` has to be specified as an input parameter.
The clustering is done using the classical K-means method with Euclidean distance (but other 
clustering methods and distance measures can be used too).

The result of the clustering is a set of cluster numbers given in ``cluster_labs``, that indicate to
which cluster belongs each rhythmic pattern. Besides, the centroid of each cluster is given in
``centroids`` as a representative rhythmic pattern of the group. In this way, they represent the
different types of rhythmic patterns found in the recording. 

    ...


More examples
~~~~~~~~~~~~~

More example scripts are provided in the :ref:`Examples <moreexamples>` section.
