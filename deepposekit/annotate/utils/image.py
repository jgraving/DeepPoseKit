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

__all__ = ["check_image_array"]


def check_image_array(self, X):
    if X.ndim > 2:
        self.rows = X.shape[1]
        self.cols = X.shape[2]
        if X.ndim > 3:
            self.channels = X.shape[3]
        else:
            self.channels = 1
    else:
        raise ValueError(
            """X is an invalid shape.
                         Must be 3-d or 4-d array-like"""
        )
    return X.reshape(X.shape[0], -1)
