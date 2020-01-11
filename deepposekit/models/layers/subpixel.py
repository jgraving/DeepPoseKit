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

from deepposekit.models.backend.backend import find_subpixel_maxima

from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.python.keras.utils.conv_utils import normalize_data_format


class SubpixelMaxima2D(Layer):
    """Subpixel maxima layer for 2D inputs.
    Convolves a 2D Gaussian kernel to find
    the subpixel maxima and 2D indices
    for the channels in the input.
    The output is ordered as [row, col, maximum].
    # Arguments
        index: Integer,
            The index to slice the channels to.
            Default is None, which does not slice the channels.
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
    # Input shape
        4D tensor with shape:
        - If `data_format` is `"channels_last"`:
            `(batch, rows, cols, channels)`
        - If `data_format` is `"channels_first"`:
            `(batch, channels, rows, cols)`
    # Output shape
        3D tensor with shape:
        - If `data_format` is `"channels_last"`:
            `(batch, 3, index)`
        - If `data_format` is `"channels_first"`:
            `(batch, index, 3)`
    """

    def __init__(
        self,
        kernel_size,
        sigma,
        upsample_factor,
        index=None,
        coordinate_scale=1.0,
        confidence_scale=1.0,
        data_format=None,
        **kwargs
    ):
        super(SubpixelMaxima2D, self).__init__(**kwargs)
        self.data_format = normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=4)
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.upsample_factor = upsample_factor
        self.index = index
        self.coordinate_scale = coordinate_scale
        self.confidence_scale = confidence_scale

    def compute_output_shape(self, input_shape):
        if self.data_format == "channels_first":
            n_channels = self.index if self.index is not None else input_shape[1]
        elif self.data_format == "channels_last":
            n_channels = self.index if self.index is not None else input_shape[3]
        return (input_shape[0], n_channels, 3)

    def call(self, inputs):
        if self.data_format == "channels_first":
            inputs = inputs[:, : self.index]
        elif self.data_format == "channels_last":
            inputs = inputs[..., : self.index]
        outputs = find_subpixel_maxima(
            inputs,
            self.kernel_size,
            self.sigma,
            self.upsample_factor,
            self.coordinate_scale,
            self.confidence_scale,
            self.data_format,
        )
        return outputs

    def get_config(self):
        config = {
            "data_format": self.data_format,
            "kernel_size": self.kernel_size,
            "sigma": self.sigma,
            "upsample_factor": self.upsample_factor,
            "index": self.index,
            "coordinate_scale": self.coordinate_scale,
            "confidence_scale": self.confidence_scale,
        }
        base_config = super(SubpixelMaxima2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
