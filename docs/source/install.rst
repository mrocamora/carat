Installation instructions
^^^^^^^^^^^^^^^^^^^^^^^^^

pypi
~~~~
The simplest way to install *carat* is through the Python Package Index (PyPI).
This will ensure that all required dependencies are fulfilled.
This can be achieved by executing the following command::

    pip install carat

or::

    sudo pip install carat

to install system-wide, or::

    pip install -u carat

to install just for your own user.


Source
~~~~~~

If you've downloaded the archive manually from the `releases
<https://github.com/mrocamora/carat/releases/>`_ page, you can install using the
`setuptools` script::

    tar xzf carat-VERSION.tar.gz
    cd carat-VERSION/
    python setup.py install

If you intend to develop librosa or make changes to the source code, you can
install with `pip install -e` to link to your actively developed source tree::

    tar xzf carat-VERSION.tar.gz
    cd carat-VERSION/
    pip install -e .

Alternately, the latest development version can be installed via pip::

    pip install git+https://github.com/mrocamora/carat


ffmpeg
~~~~~~

To fuel `audioread` with more audio-decoding power, you can install *ffmpeg* which
ships with many audio decoders.  Note that conda users on Linux and OSX will
have this installed by default; Windows users must install ffmpeg separately.

OSX users can use *homebrew* to install ffmpeg by calling
`brew install ffmpeg` or get a binary version from their website https://www.ffmpeg.org.
