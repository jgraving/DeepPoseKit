# -*- coding: utf-8 -*-
"""
This setup script is Copyright (C) 2012-2018 Michael L. Waskom
Modified by Jacob M. Graving <jgraving@gmail.com> 2015-2018

Copyright 2018 Jacob M. Graving <jgraving@gmail.com>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import sys
import warnings

major = sys.version_info.major
minor = sys.version_info.minor

if major >= 3 and minor >= 6:
  ImportError = ModuleNotFoundError


# temporarily redirect config directory to prevent matplotlib importing
# testing for writeable directory which results in sandbox error
os.environ["MPLCONFIGDIR"] = "."

DESCRIPTION = "An API for pose estimation"
LONG_DESCRIPTION = """\
An API for pose estimation
"""

DISTNAME = 'deepposekit'
MAINTAINER = 'Jacob Graving <jgraving@gmail.com>'
MAINTAINER_EMAIL = 'jgraving@gmail.com'
URL = 'https://github.com/jgraving/deepposekit'
LICENSE = 'Apache 2.0'
DOWNLOAD_URL = 'https://github.com/jgraving/deepposekit.git'
VERSION = '0.1.dev'

from setuptools import setup, find_packages

def check_dependencies():
    install_requires = []
    try:
        import numpy
    except ImportError:
        install_requires.append('numpy')
    try:
        import matplotlib
    except ImportError:
        install_requires.append('matplotlib')
    try:
        import pandas
    except ImportError:
        install_requires.append('pandas')
    try:
        import h5py
    except ImportError:
        install_requires.append('h5py')
    try:
        import imgaug
    except ImportError:
        install_requires.append('imgaug')
    try:
        import keras
    except ImportError:
        install_requires.append('keras')
    try:
        import cv2
    except ImportError:
        install_requires.append('opencv-python')
    
    return install_requires

if __name__ == "__main__":

    install_requires = check_dependencies()

    setup(name=DISTNAME,
          author=MAINTAINER,
          author_email=MAINTAINER_EMAIL,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          long_description=LONG_DESCRIPTION,
          license=LICENSE,
          url=URL,
          version=VERSION,
          download_url=DOWNLOAD_URL,
          install_requires=install_requires,
          packages=find_packages(),
          zip_safe=False,
          classifiers=['Intended Audience :: Science/Research',
                       'Programming Language :: Python :: 2.7',
                       'Programming Language :: Python :: 3',
                       'Topic :: Scientific/Engineering :: Visualization',
                       'Topic :: Scientific/Engineering :: Image Recognition',
                       'Topic :: Scientific/Engineering :: Information Analysis',
                       'Topic :: Multimedia :: Video'
                       'Operating System :: POSIX',
                       'Operating System :: Unix',
                       'Operating System :: MacOS'])