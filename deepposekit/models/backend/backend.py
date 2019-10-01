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

import tensorflow.keras.backend as K
from tensorflow.keras.backend import int_shape, permute_dimensions, dtype, floatx
import tensorflow as tf
import numpy as np

from deepposekit.models.backend.utils import gaussian_kernel_2d
from deepposekit.models.backend.registration import _upsampled_registration

__all__ = [
    "resize_images",
    "find_maxima",
    "find_subpixel_maxima",
    "depth_to_space",
    "space_to_depth",
]


def resize_images(x, height_factor, width_factor, interpolation, data_format):
    """Resizes the images contained in a 4D tensor.
    # Arguments
        x: Tensor or variable to resize.
        height_factor: Positive integer.
        width_factor: Positive integer.
        interpolation: string, "nearest", "bilinear" or "bicubic"
        data_format: string, `"channels_last"` or `"channels_first"`.
    # Returns
        A tensor.
    # Raises
        ValueError: if `data_format` is neither `"channels_last"` or `"channels_first"`.
    """
    if interpolation == "nearest":
        tf_resize = tf.image.resize_nearest_neighbor
    elif interpolation == "bilinear":
        tf_resize = tf.image.resize_bilinear
    elif interpolation == "bicubic":
        tf_resize = tf.image.resize_bicubic
    else:
        raise ValueError("Invalid interpolation method:", interpolation)
    if data_format == "channels_first":
        original_shape = int_shape(x)
        new_shape = tf.shape(x)[2:]
        new_shape *= tf.constant(
            np.array([height_factor, width_factor]).astype("int32")
        )
        x = permute_dimensions(x, [0, 2, 3, 1])
        x = tf_resize(x, new_shape, align_corners=True)
        x = permute_dimensions(x, [0, 3, 1, 2])
        x.set_shape(
            (
                None,
                None,
                original_shape[2] * height_factor
                if original_shape[2] is not None
                else None,
                original_shape[3] * width_factor
                if original_shape[3] is not None
                else None,
            )
        )
        return x
    elif data_format == "channels_last":
        original_shape = int_shape(x)
        new_shape = tf.shape(x)[1:3]
        new_shape *= tf.constant(
            np.array([height_factor, width_factor]).astype("int32")
        )
        x = tf_resize(x, new_shape, align_corners=True)
        x.set_shape(
            (
                None,
                original_shape[1] * height_factor
                if original_shape[1] is not None
                else None,
                original_shape[2] * width_factor
                if original_shape[2] is not None
                else None,
                None,
            )
        )
        return x
    else:
        raise ValueError("Invalid data_format:", data_format)


def _find_maxima(x, coordinate_scale=1, confidence_scale=255.0):

    x = K.cast(x, K.floatx())

    col_max = K.max(x, axis=1)
    row_max = K.max(x, axis=2)

    maxima = K.max(col_max, 1)
    maxima = K.expand_dims(maxima, -2) / confidence_scale

    cols = K.cast(K.argmax(col_max, -2), K.floatx())
    rows = K.cast(K.argmax(row_max, -2), K.floatx())
    cols = K.expand_dims(cols, -2) * coordinate_scale
    rows = K.expand_dims(rows, -2) * coordinate_scale

    maxima = K.concatenate([cols, rows, maxima], -2)

    return maxima


def find_maxima(x, coordinate_scale=1, confidence_scale=255.0, data_format=None):
    """Finds the 2D maxima contained in a 4D tensor.
    # Arguments
        x: Tensor or variable.
        data_format: string, `"channels_last"` or `"channels_first"`.
    # Returns
        A tensor.
    # Raises
        ValueError: if `data_format` is neither `"channels_last"` or `"channels_first"`.
    """
    if data_format == "channels_first":
        x = permute_dimensions(x, [0, 2, 3, 1])
        x = _find_maxima(x, coordinate_scale, confidence_scale)
        x = permute_dimensions(x, [0, 2, 1])
        return x
    elif data_format == "channels_last":
        x = _find_maxima(x, coordinate_scale, confidence_scale)
        x = permute_dimensions(x, [0, 2, 1])
        return x
    else:
        raise ValueError("Invalid data_format:", data_format)


def _find_subpixel_maxima(
    x, kernel_size, sigma, upsample_factor, coordinate_scale=1.0, confidence_scale=1.0
):

    kernel = gaussian_kernel_2d(kernel_size, sigma)
    kernel = tf.expand_dims(kernel, 0)

    x_shape = tf.shape(x)
    rows = x_shape[1]
    cols = x_shape[2]

    max_vals = tf.reduce_max(tf.reshape(x, [-1, rows * cols]), axis=1)
    max_vals = tf.reshape(max_vals, [-1, 1]) / confidence_scale

    row_pad = rows // 2 - kernel_size // 2
    col_pad = cols // 2 - kernel_size // 2
    padding = [[0, 0], [row_pad, row_pad - 1], [col_pad, col_pad - 1]]
    kernel = tf.pad(kernel, padding)

    row_center = row_pad + (kernel_size // 2)
    col_center = col_pad + (kernel_size // 2)
    center = tf.stack([row_center, col_center])
    center = tf.expand_dims(center, 0)
    center = tf.cast(center, dtype=tf.float32)

    shifts = _upsampled_registration(x, kernel, upsample_factor)
    shifts = center - shifts
    shifts *= coordinate_scale
    maxima = tf.concat([shifts[:, ::-1], max_vals], -1)

    return maxima


def find_subpixel_maxima(
    x,
    kernel_size,
    sigma,
    upsample_factor,
    coordinate_scale=1.0,
    confidence_scale=1.0,
    data_format=None,
):
    """Finds the 2D maxima contained in a 4D tensor.
    # Arguments
        x: Tensor or variable.
        data_format: string, `"channels_last"` or `"channels_first"`.
    # Returns
        A tensor.
    # Raises
        ValueError: if `data_format` is neither `"channels_last"` or `"channels_first"`.
    """
    if data_format == "channels_first":
        x = permute_dimensions(x, [0, 2, 3, 1])
        x_shape = K.shape(x)
        batch = x_shape[0]
        row = x_shape[1]
        col = x_shape[2]
        channels = x_shape[3]
        x = permute_dimensions(x, [0, 3, 1, 2])
        x = K.reshape(x, [batch * channels, row, col])
        x = _find_subpixel_maxima(
            x, kernel_size, sigma, upsample_factor, coordinate_scale, confidence_scale
        )
        x = K.reshape(x, [batch, channels, 3])
        return x
    elif data_format == "channels_last":
        x_shape = K.shape(x)
        batch = x_shape[0]
        row = x_shape[1]
        col = x_shape[2]
        channels = x_shape[3]
        x = permute_dimensions(x, [0, 3, 1, 2])
        x = K.reshape(x, [batch * channels, row, col])
        x = _find_subpixel_maxima(
            x, kernel_size, sigma, upsample_factor, coordinate_scale, confidence_scale
        )
        x = K.reshape(x, [batch, channels, 3])

        return x
    else:
        raise ValueError("Invalid data_format:", data_format)


def _preprocess_conv2d_input(x, data_format):
    """Transpose and cast the input before the conv2d.
    # Arguments
        x: input tensor.
        data_format: string, `"channels_last"` or `"channels_first"`.
    # Returns
        A tensor.
    """
    if dtype(x) == "float64":
        x = tf.cast(x, "float32")
    if data_format == "channels_first":
        # TF uses the last dimension as channel dimension,
        # instead of the 2nd one.
        # TH input shape: (samples, input_depth, rows, cols)
        # TF input shape: (samples, rows, cols, input_depth)
        x = tf.transpose(x, (0, 2, 3, 1))
    return x


def _postprocess_conv2d_output(x, data_format):
    """Transpose and cast the output from conv2d if needed.
    # Arguments
        x: A tensor.
        data_format: string, `"channels_last"` or `"channels_first"`.
    # Returns
        A tensor.
    """

    if data_format == "channels_first":
        x = tf.transpose(x, (0, 3, 1, 2))

    if floatx() == "float64":
        x = tf.cast(x, "float64")
    return x


def depth_to_space(input, scale, data_format=None):
    """ Uses phase shift algorithm to convert
    channels/depth to spatial resolution """
    if data_format is None:
        data_format = K.image_data_format()
    data_format = data_format.lower()
    input = _preprocess_conv2d_input(input, data_format)
    out = tf.nn.depth_to_space(input, scale)
    out = _postprocess_conv2d_output(out, data_format)
    return out


def space_to_depth(input, scale, data_format=None):
    """ Uses phase shift algorithm to convert
    spatial resolution to channels/depth """
    if data_format is None:
        data_format = K.image_data_format()
    data_format = data_format.lower()
    input = _preprocess_conv2d_input(input, data_format)
    out = tf.nn.space_to_depth(input, scale)
    out = _postprocess_conv2d_output(out, data_format)
    return out
