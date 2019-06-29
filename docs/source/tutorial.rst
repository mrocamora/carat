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

    # 2. Load the audio waveform `y` and its sampling rate as `sr`
    y, sr = carat.audio.load(audio_path, sr=None)

    # 3. Load beats/downbeats time instants and beat/downbeats labels
    beats, beat_labs = carat.annotations.load_beats(annotations_path)
    downbeats, downbeat_labs = carat.annotations.load_downbeats(annotations_path)

    # 4. Compute an accentuation feature indicating when a note has been articulated 
    #    We focus on the low frequency band (20 to 200 Hz) to get low sounding drum events
    acce, times, _ = carat.features.accentuation_feature(y, sr, minfreq=20, maxfreq=200)
     
    # 5. Compute a feature map of the rhythmic patterns
    # number of beats per bar
    n_beats = int(round(beats.size/downbeats.size))
    # you have to provide the number of tatums (subdivisions) per beat
    n_tatums = 4
    # compute the feature map from the feature signal and the beat/dowbeat annotations
    map_acce, _, _, _ = carat.features.feature_map(acce, times, beats, downbeats, 
                                                   n_beats=n_beats, n_tatums=n_tatums)

The first step of the program::

    audio_path = carat.util.example_audio_file(num_file=1)

gets the path to an audio example file included with *carat*.  After this step,
``audio_path`` will be a string variable containing the path to the example audio file.

The second step::

    y, sr = carat.audio.load(audio_path)
    
loads and decodes the audio as a ``y``, represented as a one-dimensional
NumPy floating point array.  The variable ``sr`` contains the sampling rate of
``y``, that is, the number of samples per second of audio.  By default, all audio is
mixed to mono and resampled to 22050 Hz at load time.  This behavior can be overridden
by supplying additional arguments to ``carat.audio.load()``.

Next, we get the path to the annotations::

    annotations_path = carat.util.example_beats_file(num_file=1)

The output of the beat tracker is an estimate of the tempo (in beats per minute), 
and an array of frame numbers corresponding to detected beat events.

frame here correspond to short windows of the signal (``y``), each 
separated by ``hop_length = 512`` samples.  Since v0.3, *librosa* uses centered frames, so 
that the *k*\ th frame is centered around sample ``k * hop_length``.

The next operation converts the frame numbers ``beat_frames`` into timings::

    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

Now, ``beat_times`` will be an array of timestamps (in seconds) corresponding to
detected beat events.

Finally, ::

    carat.output.features_csv('feature_map.csv', map_acce)

    ...


More examples
~~~~~~~~~~~~~

More example scripts are provided in the xxx section.
