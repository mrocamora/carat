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
    download_url='http://github.com/mrocamora/librosa/releases',
    packages=find_packages(),
    package_data={'': ['example_data/*']},
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Development Status :: 1 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    keywords='audio music sound',
    license='ISC',
    install_requires=[
        'audioread >= 2.0.0',
        'numpy >= 1.8.0',
        'scipy >= 1.0.0',
        'scikit-learn >= 0.14.0',
        'soundfile >= 0.9.0',
    ],
    extras_require={
        'docs': ['numpydoc',
                 'matplotlib >= 2.0.0'],
        'tests': ['matplotlib >= 2.1'],
        'display': ['matplotlib >= 1.5'],
    }
)
