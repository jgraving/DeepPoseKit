# -*- coding: utf-8 -*-
"""
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

from tensorflow.keras.legacy import interfaces
from tensorflow.keras.engine import Layer
from tensorflow.keras.engine import InputSpec

from tensorflow.keras.utils import conv_utils

from ..backend import (resize_images, find_maxima,
                       register_translation, register_rotation,
                       rotate_images, translate_images,
                       depth_to_space, space_to_depth)

try:
    from tensorflow.keras.backend import normalize_data_format
except:
    from tensorflow.keras.utils.conv_utils import normalize_data_format


__all__ = ['UpSampling2D', 'Maxima2D', 'RegisterTranslation2D',
           'RegisterRotation2D', 'Rotate2D', 'Translate2D']


class UpSampling2D(Layer):
    """Upsampling layer for 2D inputs.
    Repeats the rows and columns of the data
    by size[0] and size[1] respectively with interpolation.
    # Arguments
        size: int, or tuple of 2 integers.
            The upsampling factors for rows and columns.
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
        interpolation: A string,
            one of 'nearest' (default), 'bilinear', or 'bicubic'
    # Input shape
        4D tensor with shape:
        - If `data_format` is `"channels_last"`:
            `(batch, rows, cols, channels)`
        - If `data_format` is `"channels_first"`:
            `(batch, channels, rows, cols)`
    # Output shape
        4D tensor with shape:
        - If `data_format` is `"channels_last"`:
            `(batch, upsampled_rows, upsampled_cols, channels)`
        - If `data_format` is `"channels_first"`:
            `(batch, channels, upsampled_rows, upsampled_cols)`
    """

    @interfaces.legacy_upsampling2d_support
    def __init__(self, size=(2, 2), data_format=None,
                 interpolation='nearest', **kwargs):
        super(UpSampling2D, self).__init__(**kwargs)
        self.data_format = normalize_data_format(data_format)
        self.interpolation = interpolation
        self.size = conv_utils.normalize_tuple(size, 2, 'size')
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            height = self.size[0] * input_shape[2] if input_shape[2] is not None else None
            width = self.size[1] * input_shape[3] if input_shape[3] is not None else None
            return (input_shape[0],
                    input_shape[1],
                    height,
                    width)
        elif self.data_format == 'channels_last':
            height = self.size[0] * input_shape[1] if input_shape[1] is not None else None
            width = self.size[1] * input_shape[2] if input_shape[2] is not None else None
            return (input_shape[0],
                    height,
                    width,
                    input_shape[3])

    def call(self, inputs):
        return resize_images(inputs, self.size[0], self.size[1],
                             self.interpolation, self.data_format)

    def get_config(self):
        config = {'size': self.size,
                  'data_format': self.data_format,
                  'interpolation': self.interpolation}
        base_config = super(UpSampling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


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

    def __init__(self, index=None, coordinate_scale=1.,
                 confidence_scale=255., data_format=None, **kwargs):
        super(Maxima2D, self).__init__(**kwargs)
        self.data_format = normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=4)
        self.index = index
        self.coordinate_scale = coordinate_scale
        self.confidence_scale = confidence_scale

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            n_channels = self.index if self.index is not None else input_shape[1]
            
        elif self.data_format == 'channels_last':
            n_channels = self.index if self.index is not None else input_shape[3]
        return (input_shape[0],
                n_channels,
                3)

    def call(self, inputs):
        if self.data_format == 'channels_first':
            inputs = inputs[:, :self.index]
        elif self.data_format == 'channels_last':
            inputs = inputs[..., :self.index]
        outputs = find_maxima(inputs, self.coordinate_scale,
                              self.confidence_scale, self.data_format)
        return outputs

    def get_config(self):
        config = {'data_format': self.data_format,
                  'index': self.index,
                  'coordinate_scale': self.coordinate_scale,
                  'confidence_scale': self.confidence_scale}
        base_config = super(Maxima2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RegisterTranslation2D(Layer):
    """Register translation layer for 2D inputs.
    Takes in target and source images and calculates
    the translational shift that maximizes cross-correlation.
    # Arguments
        upsample_factor : integer, optional
            The upsampling factor for subpixel resolution.  Defaults to 1.
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
        2D tensor with shape:
        - If `data_format` is `"channels_last"`:
            `(batch, 2)`
        - If `data_format` is `"channels_first"`:
            `(batch, 2)`
    """

    def __init__(self, upsample_factor=1,
                 data_format=None, **kwargs):
        super(RegisterTranslation2D, self).__init__(**kwargs)
        self.data_format = normalize_data_format(data_format)
        self.upsample_factor = upsample_factor

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 2)

    def call(self, inputs):
        if not isinstance(inputs, list):
            raise ValueError('''inputs must be a list 
                             [target_images, src_images]''')
        target_images = inputs[0]
        src_images = inputs[1]
        return register_translation(inputs[0], inputs[1],
                                    self.upsample_factor, self.data_format)

    def get_config(self):
        config = {'data_format': self.data_format,
                  'upsample_factor': self.upsample_factor}
        base_config = super(RegisterTranslation2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RegisterRotation2D(Layer):
    """Register rotation layer for 2D inputs.
    Takes in target and source images and calculates
    the rotational shift that maximizes cross-correlation.
    # Arguments
        upsample_factor : integer, optional
            The upsampling factor for subpixel resolution.  Defaults to 1.
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
        2D tensor with shape:
        - If `data_format` is `"channels_last"`:
            `(batch, 1)`
        - If `data_format` is `"channels_first"`:
            `(batch, 1)`
    """

    def __init__(self, upsample_factor=1,
                 rotation_resolution=1, rotation_guess=0,
                 data_format=None, **kwargs):
        super(RegisterRotation2D, self).__init__(**kwargs)
        self.data_format = normalize_data_format(data_format)
        self.upsample_factor = upsample_factor
        self.rotation_resolution = rotation_resolution
        self.rotation_guess = rotation_guess

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)

    def call(self, inputs):
        if not isinstance(inputs, list):
            raise ValueError('''inputs must be a list
                             [target_images, src_images]''')
        return register_rotation(inputs[0], inputs[1],
                                 self.rotation_resolution, self.rotation_guess,
                                 self.upsample_factor, self.data_format)

    def get_config(self):
        config = {'data_format': self.data_format,
                  'upsample_factor': self.upsample_factor,
                  'rotation_resolution': self.rotation_resolution,
                  'rotation_guess': self.rotation_guess}
        base_config = super(RegisterRotation2D, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


class Rotate2D(Layer):
    """Rotation layer for 2D inputs.
    Takes in a 4-D array and rotation angles in radians.
    # Arguments
        interpolation: A string,
            one of 'bilinear' (default), 'nearest'
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
        4D tensor with shape:
        - If `data_format` is `"channels_last"`:
            `(batch, rows, cols, channels)`
        - If `data_format` is `"channels_first"`:
            `(batch, channels, rows, cols)`
    """

    def __init__(self, interpolation='bilinear',
                 data_format=None, **kwargs):
        super(Rotate2D, self).__init__(**kwargs)
        self.data_format = normalize_data_format(data_format)
        self.interpolation = interpolation

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        if not isinstance(inputs, list):
            raise ValueError('''inputs must be a list 
                             [images, angles]''')
        return rotate_images(inputs[0], inputs[1],
                             self.interpolation, self.data_format)

    def get_config(self):
        config = {'data_format': self.data_format,
                  'interpolation': self.interpolation}
        base_config = super(Rotate2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Translate2D(Layer):
    """Translation layer for 2D inputs.
    Takes in a 4-D array and shift values in column-major [x,y] format.
    Returns shifted images
    # Arguments
        interpolation: A string,
            one of 'bilinear' (default), 'nearest'
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
        2D tensor with shape:
        - If `data_format` is `"channels_last"`:
            `(batch, rows, cols, channels)`
        - If `data_format` is `"channels_first"`:
            `(batch, rows, cols, channels)`
    """

    def __init__(self, interpolation='bilinear',
                 data_format=None, **kwargs):
        super(Translate2D, self).__init__(**kwargs)
        self.data_format = normalize_data_format(data_format)
        self.interpolation = interpolation

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        if not isinstance(inputs, list):
            raise ValueError('''inputs must be a list 
                             [images, shifts]''')
        return translate_images(inputs[0], inputs[1],
                                self.interpolation, self.data_format)

    def get_config(self):
        config = {'data_format': self.data_format,
                  'interpolation': self.interpolation}
        base_config = super(Translate2D, self).get_config()
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
        if self.data_format == 'channels_first':
            b, k, r, c = input_shape
            return (b, k // (self.scale_factor ** 2), r * self.scale_factor, c * self.scale_factor)
        else:
            b, r, c, k = input_shape
            return (b, r * self.scale_factor, c * self.scale_factor, k // (self.scale_factor ** 2))

    def get_config(self):
        config = {'scale_factor': self.scale_factor,
                  'data_format': self.data_format}
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
        super(SubPixelDownscaling, self).__init__(**kwargs)

        self.scale_factor = scale_factor
        self.data_format = normalize_data_format(data_format)

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        y = space_to_depth(x, self.scale_factor, self.data_format)
        return y

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            b, k, r, c = input_shape
            return (b, k // (self.scale_factor ** 2), r // self.scale_factor, c // self.scale_factor)
        else:
            b, r, c, k = input_shape
            return (b, r // self.scale_factor, c // self.scale_factor, k * (self.scale_factor ** 2))

    def get_config(self):
        config = {'scale_factor': self.scale_factor,
                  'data_format': self.data_format}
        base_config = super(SubPixelDownscaling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))