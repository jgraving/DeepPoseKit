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

import tensorflow as tf
import numpy as np
from deepposekit.models.backend.utils import fftshift1d, fft2d, find_maxima, fix

__all__ = ["_upsampled_registration"]


def _col_kernel(upsampled_region_size, upsample_factor, axis_offsets, data_shape):

    data_shape_float = tf.cast(data_shape, tf.float32)
    col_constant = tf.cast(data_shape_float[2] * upsample_factor, tf.complex64)
    col_constant = -1j * 2 * np.pi / col_constant

    col_kernel_a = tf.range(0, data_shape_float[2], dtype=tf.float32)
    col_kernel_a = fftshift1d(col_kernel_a)  # TODO: replace with tf.signal.fftshift
    col_kernel_a = tf.reshape(col_kernel_a, (-1, 1))
    col_kernel_a -= tf.floor(data_shape_float[2] / 2.0)
    col_kernel_a = tf.reshape(col_kernel_a, (1, -1))
    col_kernel_a = tf.tile(col_kernel_a, (data_shape[0], 1))

    col_kernel_b = tf.range(0, upsampled_region_size, dtype=tf.float32)
    col_kernel_b = tf.reshape(col_kernel_b, (1, -1))
    col_kernel_b = tf.tile(col_kernel_b, (data_shape[0], 1))
    col_kernel_b = tf.transpose(col_kernel_b)
    col_kernel_b -= tf.transpose(axis_offsets[:, 1])
    col_kernel_b = tf.transpose(col_kernel_b)

    col_kernel_a = tf.expand_dims(col_kernel_a, 1)
    col_kernel_b = tf.expand_dims(col_kernel_b, -1)

    col_kernel = col_kernel_a * col_kernel_b
    col_kernel = tf.transpose(col_kernel, perm=(0, 2, 1))
    col_kernel = col_constant * tf.cast(col_kernel, tf.complex64)
    col_kernel = tf.exp(col_kernel)
    return col_kernel


def _row_kernel(upsampled_region_size, upsample_factor, axis_offsets, data_shape):

    data_shape_float = tf.cast(data_shape, tf.float32)
    row_constant = tf.cast(data_shape_float[1] * upsample_factor, tf.complex64)
    row_constant = -1j * 2 * np.pi / row_constant

    row_kernel_a = tf.range(0, upsampled_region_size, dtype=tf.float32)
    row_kernel_a = tf.reshape(row_kernel_a, (1, -1))
    row_kernel_a = tf.tile(row_kernel_a, (data_shape[0], 1))
    row_kernel_a = tf.transpose(row_kernel_a)
    row_kernel_a = row_kernel_a - axis_offsets[:, 0]

    row_kernel_b = tf.range(0, data_shape_float[1], dtype=tf.float32)
    row_kernel_b = fftshift1d(row_kernel_b)  # TODO: replace with tf.signal.fftshift
    row_kernel_b = tf.reshape(row_kernel_b, (1, -1))
    row_kernel_b = tf.tile(row_kernel_b, (data_shape[0], 1))
    row_kernel_b = row_kernel_b - tf.floor(data_shape_float[1] / 2.0)

    row_kernel_a = tf.expand_dims(row_kernel_a, 1)
    row_kernel_b = tf.expand_dims(row_kernel_b, -1)

    row_kernel = tf.transpose(row_kernel_a) * row_kernel_b
    row_kernel = tf.transpose(row_kernel, perm=(0, 2, 1))
    row_kernel = row_constant * tf.cast(row_kernel, tf.complex64)

    row_kernel = tf.exp(row_kernel)

    return row_kernel


def _upsampled_dft(data, upsampled_region_size, upsample_factor, axis_offsets):
    """
    Upsampled DFT by matrix multiplication.
    This code is intended to provide the same result as if the following
    operations were performed:
        - Embed the array "data" in an array that is ``upsample_factor`` times
          larger in each dimension.  ifftshift to bring the center of the
          image to (1,1).
        - Take the FFT of the larger array.
        - Extract an ``[upsampled_region_size]`` region of the result, starting
          with the ``[axis_offsets+1]`` element.
    It achieves this result by computing the DFT in the output array without
    the need to zeropad. Much faster and memory efficient than the zero-padded
    FFT approach if ``upsampled_region_size`` is much smaller than
    ``data.size * upsample_factor``.
    Parameters
    ----------
    data : 2D ndarray
        The input data array (DFT of original data) to upsample.
    upsampled_region_size : integer or tuple of integers, optional
        The size of the region to be sampled.  If one integer is provided, it
        is duplicated up to the dimensionality of ``data``.
    upsample_factor : integer, optional
        The upsampling factor.  Defaults to 1.
    axis_offsets : tuple of integers, optional
        The offsets of the region to be sampled.  Defaults to None (uses
        image center)
    Returns
    -------
    output : 2D ndarray
            The upsampled DFT of the specified region.
    """
    data_shape = tf.shape(data)

    col_kernel = _col_kernel(
        upsampled_region_size, upsample_factor, axis_offsets, data_shape
    )
    row_kernel = _row_kernel(
        upsampled_region_size, upsample_factor, axis_offsets, data_shape
    )

    upsampled_dft = tf.matmul(tf.matmul(row_kernel, data), col_kernel)

    return upsampled_dft


def _upsampled_registration(target_image, src_image, upsample_factor):

    upsample_factor = tf.constant(upsample_factor, tf.float32)

    target_shape = tf.shape(target_image)
    target_image = tf.reshape(target_image, target_shape[:3])
    src_shape = tf.shape(src_image)
    src_image = tf.reshape(src_image, src_shape[:3])

    src_freq = fft2d(src_image)
    target_freq = fft2d(target_image)

    shape = tf.reshape(tf.shape(src_freq)[1:3], (1, 2))
    shape = tf.cast(shape, tf.float32)
    shape = tf.tile(shape, (tf.shape(target_freq)[0], 1))
    image_product = src_freq * tf.math.conj(target_freq)
    cross_correlation = tf.signal.ifft2d(image_product)

    maxima = find_maxima(tf.abs(cross_correlation))
    midpoints = fix(tf.cast(shape, tf.float32) / 2.0)

    shifts = maxima
    shifts = tf.where(shifts > midpoints, shifts - shape, shifts)
    shifts = tf.round(shifts * upsample_factor) / upsample_factor

    upsampled_region_size = tf.math.ceil(upsample_factor * 1.5)
    dftshift = fix(upsampled_region_size / 2.0)
    normalization = tf.cast(tf.size(src_freq[0]), tf.float32)
    normalization *= upsample_factor ** 2
    sample_region_offset = dftshift - shifts * upsample_factor

    data = tf.math.conj(image_product)
    upsampled_dft = _upsampled_dft(
        data, upsampled_region_size, upsample_factor, sample_region_offset
    )

    cross_correlation = tf.math.conj(upsampled_dft)
    cross_correlation /= tf.cast(normalization, tf.complex64)
    cross_correlation = tf.abs(cross_correlation)

    maxima = find_maxima(cross_correlation)
    maxima = maxima - dftshift
    shifts = shifts + maxima / upsample_factor

    return shifts
