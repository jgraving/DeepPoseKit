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

from tensorflow.keras.layers import Layer, InputSpec

from tensorflow.keras.layers import UpSampling2D

from deepposekit.models.backend.backend import (
    resize_images,
    find_maxima,
    depth_to_space,
    space_to_depth,
)

from tensorflow.python.keras.utils.conv_utils import (
    normalize_data_format,
    normalize_tuple,
)

__all__ = ["UpSampling2D", "Maxima2D", "SubPixelUpscaling", "SubPixelDownscaling"]


class Maxima2D(Layer):
    """Maxima layer for 2D inputs.
    Finds the maxima and 2D indices
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
        index=None,
        coordinate_scale=1.0,
        confidence_scale=1.0,
        data_format=None,
        **kwargs
    ):
        super(Maxima2D, self).__init__(**kwargs)
        self.data_format = normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=4)
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
        outputs = find_maxima(
            inputs, self.coordinate_scale, self.confidence_scale, self.data_format
        )
        return outputs

    def get_config(self):
        config = {
            "data_format": self.data_format,
            "index": self.index,
            "coordinate_scale": self.coordinate_scale,
            "confidence_scale": self.confidence_scale,
        }
        base_config = super(Maxima2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SubPixelUpscaling(Layer):
    """ Sub-pixel convolutional upscaling layer based on the paper "Real-Time Single Image
    and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network"
    (https://arxiv.org/abs/1609.05158).
    This layer requires a Convolution2D prior to it, having output filters computed according to
    the formula :
        filters = k * (scale_factor * scale_factor)
        where k = a user defined number of filters (generally larger than 32)
              scale_factor = the upscaling factor (generally 2)
    This layer performs the depth to space operation on the convolution filters, and returns a
    tensor with the size as defined below.
    # Example :
    ```python
        # A standard subpixel upscaling block
        x = Convolution2D(256, 3, 3, padding='same', activation='relu')(...)
        u = SubPixelUpscaling(scale_factor=2)(x)
        [Optional]
        x = Convolution2D(256, 3, 3, padding='same', activation='relu')(u)
    ```
        In practice, it is useful to have a second convolution layer after the
        SubPixelUpscaling layer to speed up the learning process.
        However, if you are stacking multiple SubPixelUpscaling blocks, it may increase
        the number of parameters greatly, so the Convolution layer after SubPixelUpscaling
        layer can be removed.
    # Arguments
        scale_factor: Upscaling factor.
        data_format: Can be None, 'channels_first' or 'channels_last'.
    # Input shape
        4D tensor with shape:
        `(samples, k * (scale_factor * scale_factor) channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, k * (scale_factor * scale_factor) channels)` if data_format='channels_last'.
    # Output shape
        4D tensor with shape:
        `(samples, k channels, rows * scale_factor, cols * scale_factor))` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows * scale_factor, cols * scale_factor, k channels)` if data_format='channels_last'.
    """

    def __init__(self, scale_factor=2, data_format=None, **kwargs):
        super(SubPixelUpscaling, self).__init__(**kwargs)

        self.scale_factor = scale_factor
        self.data_format = normalize_data_format(data_format)

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        y = depth_to_space(x, self.scale_factor, self.data_format)
        return y

    def compute_output_shape(self, input_shape):
        if self.data_format == "channels_first":
            b, k, r, c = input_shape
            return (
                b,
                k // (self.scale_factor ** 2),
                r * self.scale_factor,
                c * self.scale_factor,
            )
        else:
            b, r, c, k = input_shape
            return (
                b,
                r * self.scale_factor,
                c * self.scale_factor,
                k // (self.scale_factor ** 2),
            )

    def get_config(self):
        config = {"scale_factor": self.scale_factor, "data_format": self.data_format}
        base_config = super(SubPixelUpscaling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SubPixelDownscaling(Layer):
    """ Sub-pixel convolutional downscaling layer based on the paper "Real-Time Single Image
    and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network"
    (https://arxiv.org/abs/1609.05158).
    This layer requires a Convolution2D prior to it, having output filters computed according to
    the formula :
        filters = k * (scale_factor * scale_factor)
        where k = a user defined number of filters (generally larger than 32)
              scale_factor = the upscaling factor (generally 2)
    This layer performs the depth to space operation on the convolution filters, and returns a
    tensor with the size as defined below.
    # Example :
    ```python
        # A standard subpixel upscaling block
        x = Convolution2D(256, 3, 3, padding='same', activation='relu')(...)
        u = SubPixelDownscaling(scale_factor=2)(x)
        [Optional]
        x = Convolution2D(256, 3, 3, padding='same', activation='relu')(u)
    ```
        In practice, it is useful to have a second convolution layer after the
        SubPixelUpscaling layer to speed up the learning process.
        However, if you are stacking multiple SubPixelUpscaling blocks, it may increase
        the number of parameters greatly, so the Convolution layer after SubPixelUpscaling
        layer can be removed.
    # Arguments
        scale_factor: Upscaling factor.
        data_format: Can be None, 'channels_first' or 'channels_last'.
    # Input shape
        4D tensor with shape:
        `(samples, k * (scale_factor * scale_factor) channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, k * (scale_factor * scale_factor) channels)` if data_format='channels_last'.
    # Output shape
        4D tensor with shape:
        `(samples, k channels, rows * scale_factor, cols * scale_factor))` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows * scale_factor, cols * scale_factor, k channels)` if data_format='channels_last'.
    """

    def __init__(self, scale_factor=2, data_format=None, **kwargs):
        super(SubPixelDownscaling, self).__init__(**kwargs)

        self.scale_factor = scale_factor
        self.data_format = normalize_data_format(data_format)

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        y = space_to_depth(x, self.scale_factor, self.data_format)
        return y

    def compute_output_shape(self, input_shape):
        if self.data_format == "channels_first":
            b, k, r, c = input_shape
            return (
                b,
                k // (self.scale_factor ** 2),
                r // self.scale_factor,
                c // self.scale_factor,
            )
        else:
            b, r, c, k = input_shape
            return (
                b,
                r // self.scale_factor,
                c // self.scale_factor,
                k * (self.scale_factor ** 2),
            )

    def get_config(self):
        config = {"scale_factor": self.scale_factor, "data_format": self.data_format}
        base_config = super(SubPixelDownscaling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
