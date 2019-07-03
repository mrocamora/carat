from setuptools import setup, find_packages
from importlib.machinery import SourceFileLoader
import sys


version = SourceFileLoader('carat.version',
                               'carat/version.py').load_module()

with open('README.md', 'r') as fdesc:
    long_description = fdesc.read()

setup(
    name='carat',
    version=version.version,
    description='Computer-Aided Rhythm Analysis Toolbox',
    author='Martín Rocamora',
    author_email='rocamora@fing.edu.uy',
    url='http://github.com/mrocamora/carat',
    download_url='http://github.com/mrocamora/carat/releases',
    packages=find_packages(),
    package_data={'': ['example_data/*']},
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords='audio music sound',
    license='MIT',
    install_requires=[
        'audioread >= 2.0.0',
        'numpy >= 1.8.0',
        'scipy >= 1.0.0',
        'scikit-learn >= 0.14.0',
        'soundfile >= 0.9.0',
        'librosa >= 0.6.3',
        'matplotlib >= 2.0.0',
        'sounddevice >= 0.3.13'
    ],
    extras_require={
        'docs': ['numpydoc', 'sphinx!=1.3.1', 'sphinx_rtd_theme',
                 'matplotlib >= 2.0.0',
                 'sphinxcontrib-versioning >= 2.2.1',
                 'sphinx-gallery'],
        'tests': ['matplotlib >= 2.1'],
        'display': ['matplotlib >= 1.5'],
    }
)
