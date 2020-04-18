# -*- coding: utf-8 -*-
"""
Copyright 2018-2019 Jacob M. Graving <jgraving@gmail.com>

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
from setuptools import setup, find_packages

DESCRIPTION = "a toolkit for pose estimation using deep learning"
DISTNAME = "deepposekit"
MAINTAINER = "Jacob Graving <jgraving@gmail.com>"
MAINTAINER_EMAIL = "jgraving@gmail.com"
URL = "https://github.com/jgraving/deepposekit"
LICENSE = "Apache 2.0"
DOWNLOAD_URL = "https://github.com/jgraving/deepposekit.git"
VERSION = "0.3.9"

directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

if __name__ == "__main__":

    setup(
        name=DISTNAME,
        author=MAINTAINER,
        author_email=MAINTAINER_EMAIL,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        long_description=long_description,
        long_description_content_type="text/markdown",
        license=LICENSE,
        url=URL,
        version=VERSION,
        download_url=DOWNLOAD_URL,
        install_requires=[
            "numpy",
            "matplotlib",
            "pandas",
            "h5py",
            "imgaug>=0.2.9",
            "opencv-python",
            "pyyaml",
        ],
        packages=find_packages(),
        zip_safe=False,
        classifiers=[
            "Intended Audience :: Science/Research",
            "Programming Language :: Python :: 3",
            "Topic :: Scientific/Engineering :: Image Recognition",
        ],
    )
