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

from tensorflow.keras import layers
from tensorflow.keras import backend as K
import numpy as np


def channel_squeeze_excite_block(input, ratio=0.25):
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]
    cse_shape = (1, 1, filters)

    cse = layers.GlobalAveragePooling2D()(init)
    cse = layers.Reshape(cse_shape)(cse)
    ratio_filters = int(np.round(filters * ratio))
    if ratio_filters < 1:
        ratio_filters += 1
    cse = layers.Conv2D(
        ratio_filters,
        (1, 1),
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        use_bias=False,
    )(cse)
    cse = layers.BatchNormalization()(cse)
    cse = layers.Conv2D(
        filters,
        (1, 1),
        activation="sigmoid",
        kernel_initializer="he_normal",
        use_bias=False,
    )(cse)

    if K.image_data_format() == "channels_first":
        cse = layers.Permute((3, 1, 2))(cse)

    cse = layers.Multiply()([init, cse])
    return cse


def spatial_squeeze_excite_block(input):
    sse = layers.Conv2D(
        1,
        (1, 1),
        activation="sigmoid",
        padding="same",
        kernel_initializer="he_normal",
        use_bias=False,
    )(input)
    sse = layers.Multiply()([input, sse])

    return sse


def squeeze_excite_block(input, ratio=0.25):
    cse = channel_squeeze_excite_block(input, ratio)
    sse = spatial_squeeze_excite_block(input)
    output = layers.Maximum()([cse, sse])
    return output
