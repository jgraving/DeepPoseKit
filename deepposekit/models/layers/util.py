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

from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K

__all__ = ["ImageNormalization"]


class ImageNormalization(Layer):
    """Image normalization layer.
    Normalize the value range of image arrays from [0,255] to [-1,1],
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as input.
    """

    def __init__(self, **kwargs):
        super(ImageNormalization, self).__init__(**kwargs)

    def call(self, inputs):
        return inputs * (2.0 / 255.0) - 1.0

    def compute_output_shape(self, input_shape):
        return input_shape
