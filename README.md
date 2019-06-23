carat
=====

Computer-Aided Rhythm Analysis Toolbox


[![PyPI](https://img.shields.io/pypi/v/carat.svg)](https://pypi.python.org/pypi/carat)
[![License](https://img.shields.io/pypi/l/carat.svg)](https://github.com/mrocamora/carat/blob/master/LICENSE.md)


Documentation
-------------
See http://carat.github.io/carat/ for a complete reference manual and introductory tutorials.


Demonstration notebooks
-----------------------
What does carat do?  Here are some quick demonstrations:

* [Introduction notebook](http://nbviewer.ipython.org/github/librosa/librosa/blob/master/examples/carat%20demo.ipynb): a brief introduction to some commonly used features.


Installation
------------

The latest stable release is available on PyPI, and you can install it by saying
```
pip install carat
```


To build librosa from source, say `python setup.py build`.
Then, to install librosa, say `python setup.py install`.
If all went well, you should be able to execute the demo scripts under `examples/`
(OS X users should follow the installation guide given below).

Alternatively, you can download or clone the repository and use `pip` to handle dependencies:

```
unzip carat.zip
pip install -e carat
```
or
```
git clone https://github.com/carat/carat.git
pip install -e carat
```

By calling `pip list` you should see `carat` now as an installed pacakge:
```
carat (0.x.x, /path/to/carat)
```

### Hints for the Installation

`carat` uses `librosa` to load audio files.

`librosa` uses `soundfile` and `audioread` to load audio files.
Note that `soundfile` does not currently support MP3, which will cause librosa to
fall back on the `audioread` library.

#### soundfile

If you're using `pip` on a Linux environment, you may need to install `libsndfile`
manually.  Please refer to the [SoundFile installation documentation](https://pysoundfile.readthedocs.io/#installation) for details.

#### audioread and MP3 support

To fuel `audioread` with more audio-decoding power (e.g., for reading MP3 files),
you may need to install either *ffmpeg* or *GStreamer*.

*Note that on some platforms, `audioread` needs at least one of the programs to work properly.*

Here are some common commands for different operating systems:

* Linux (apt-get): `apt-get install ffmpeg` or `apt-get install gstreamer1.0-plugins-base gstreamer1.0-plugins-ugly`
* Linux (yum): `yum install ffmpeg` or `yum install gstreamer1.0-plugins-base gstreamer1.0-plugins-ugly`
* Mac: `brew install ffmpeg` or `brew install gstreamer`
* Windows: download binaries from the website

For GStreamer, you also need to install the Python bindings with
```
pip install pygobject
```

Citing
------

If you want to cite carat please cite the paper published at AAWM 2019:

    Rocamora, and Jure. "carat: Computer-Aided Rhythmic Analysis Toolbox." In Proceedings of Analytical Approaches to World Music. 2019.
