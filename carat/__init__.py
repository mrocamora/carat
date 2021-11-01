#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Top-level module for carat"""

import warnings
import re
from .version import version as __version__
from .version import show_versions

# And all the carat sub-modules
# from ._cache import cache
from . import util
from . import audio
from . import features
from . import annotations
from . import microtiming
from . import clustering
from . import display

# Exporting exception classes at the top level
#from .util.exceptions import *  # pylint: disable=wildcard-import

# Exporting all core functions is okay here: suppress the import warning
#from .core import *  # pylint: disable=wildcard-import

warnings.filterwarnings('always',
                        category=DeprecationWarning,
                        module='^{0}'.format(re.escape(__name__)))
