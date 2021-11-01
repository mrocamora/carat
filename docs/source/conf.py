# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from pathlib import Path
import sphinx

#sys.path.insert(0, os.path.abspath('../../'))

srcpath = os.path.abspath(Path(os.path.dirname(__file__)) / '../../')
sys.path.insert(0, srcpath)


# -- Project information -----------------------------------------------------

project = 'carat'
copyright = '2021, carat development team'
author = 'The carat development team'

from importlib.machinery import SourceFileLoader

carat_version = SourceFileLoader(
    "carat.version", os.path.abspath(Path(srcpath) / 'carat' / 'version.py')
).load_module()

# The short X.Y version.
version = carat_version.version
# The full version, including alpha/beta/rc tags.
release = carat_version.version

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.viewcode',
              'numpydoc',
              'sphinx.ext.autosummary',
              'sphinx_gallery.gen_gallery']

autosummary_generate = True 

# Galley
sphinx_gallery_conf = {
        'examples_dirs': 'examples/',
        'gallery_dirs': 'auto_examples',
        'backreferences_dir': None,
        'reference_url': {
            'sphinx_gallery': None,
            'numpy': 'http://docs.scipy.org/doc/numpy/',
            'np': 'http://docs.scipy.org/doc/numpy/',
            'scipy': 'http://docs.scipy.org/doc/scipy/reference',
            'matplotlib': 'http://matplotlib.org/',
            'sklearn': 'http://scikit-learn.org/stable',
        }
    }

# Generate plots for example sections
numpydoc_use_plots = True


# The master toctree document.
master_doc = 'index'

#--------
# Doctest
#--------

doctest_global_setup = """
import numpy as np
import scipy
import carat
"""

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build']

# The reST default role (used for this markup: `text`) to use for all documents.
default_role = 'autolink'

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
