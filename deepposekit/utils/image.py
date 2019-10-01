# -*- coding: utf-8 -*-
# Copyright 2018-2019 Jacob M. Graving <jgraving@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2

__all__ = ["check_grayscale", "largest_factor", "n_downsample"]


def check_grayscale(image, return_color=False):
    if image.ndim == 2:
        grayscale = True
    elif image.shape[-1] == 1:
        grayscale = True
    else:
        grayscale = False
    if return_color and grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image, grayscale


def largest_factor(x):
    n = n_downsample(x)
    return x // 2 ** n


def n_downsample(x):
    n = 0
    while x % 2 == 0 and x > 2:
        n += 1
        x /= 2.0
    return n
