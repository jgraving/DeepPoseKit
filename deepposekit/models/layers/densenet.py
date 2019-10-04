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

import numpy as np

from tensorflow.keras import layers

from deepposekit.models.layers.convolutional import UpSampling2D
from deepposekit.models.layers.util import ImageNormalization
from deepposekit.models.layers.convolutional import (
    SubPixelDownscaling,
    SubPixelUpscaling,
)
from deepposekit.models.layers.imagenet_densenet import DenseNet121


__all__ = [
    "Concatenate",
    "DenseConv2D",
    "DenseConvBlock",
    "Compression",
    "TransitionDown",
    "TransitionUp",
    "DenseNet",
    "FrontEnd",
    "ImageNetFrontEnd",
    "OutputChannels",
]


class Concatenate:  # (layers.Layer):
    def __init__(self, **kwargs):
        # super(Concatenate, self).__init__(self, **kwargs)
        self.concat = layers.Concatenate()

    def call(self, inputs):
        if isinstance(inputs, list):
            if len(inputs) > 1:
                outputs = self.concat(inputs)
            else:
                outputs = inputs[0]
            return outputs
        else:
            return inputs

    def __call__(self, inputs):
        return self.call(inputs)

    # def get_config(self):
    #    config = {}
    #    base_config = super(Concatenate, self).get_config()
    #    return dict(list(base_config.items()) + list(config.items()))


class DenseConv2D:  # (layers.Layer):
    def __init__(self, growth_rate=64, bottleneck_factor=1, **kwargs):
        # super(DenseConv2D, self).__init__(self, **kwargs)
        self.concat = Concatenate()

        bottleneck_filters = int(np.round(growth_rate * bottleneck_factor))

        self.bottleneck_1x1 = layers.Conv2D(
            bottleneck_filters,
            (1, 1),
            padding="same",
            activation="selu",
            kernel_initializer="lecun_normal",
        )
        self.conv_3x3 = layers.Conv2D(
            growth_rate,
            (3, 3),
            padding="same",
            activation="selu",
            kernel_initializer="lecun_normal",
        )

    def call(self, inputs):
        concat = self.concat(inputs)
        bottleneck_1x1 = self.bottleneck_1x1(concat)
        conv_3x3 = self.conv_3x3(bottleneck_1x1)
        outputs = [concat, conv_3x3]
        return outputs

    def __call__(self, inputs):
        return self.call(inputs)

    # def get_config(self):
    #    config = {}
    #    base_config = super(DenseConv2D, self).get_config()
    #    return dict(list(base_config.items()) + list(config.items()))


class DenseConvBlock:  # (layers.Layer):
    def __init__(self, growth_rate=64, n_layers=1, bottleneck_factor=1, **kwargs):
        # super(DenseConv2D, self).__init__(self, **kwargs)
        n_layers = np.minimum(n_layers, 3)
        n_layers = np.maximum(n_layers, 1)
        self.dense_conv = DenseConv2D(growth_rate * n_layers, bottleneck_factor)

    def call(self, inputs):
        return self.dense_conv(inputs)

    def __call__(self, inputs):
        return self.call(inputs)

    # def get_config(self):
    #    config = {}
    #    base_config = super(DenseConv2D, self).get_config()
    #    return dict(list(base_config.items()) + list(config.items()))


class Compression:  # (layers.Layer):
    def __init__(self, compression_factor=0.5, **kwargs):
        # super(Compression, self).__init__(self, **kwargs)
        self.concat = Concatenate()
        self.compression_factor = compression_factor

    def call(self, inputs):
        concat = self.concat(inputs)

        n_channels = int(concat.shape[-1])
        compression_filters = int(np.round(n_channels * self.compression_factor))
        self.compression_1x1 = layers.Conv2D(
            compression_filters,
            (1, 1),
            padding="same",
            activation="selu",
            kernel_initializer="lecun_normal",
        )

        outputs = self.compression_1x1(concat)
        return outputs

    def __call__(self, inputs):
        return self.call(inputs)

    # def get_config(self):
    #    config = {}
    #    base_config = super(Compression, self).get_config()
    #    return dict(list(base_config.items()) + list(config.items()))


class TransitionDown:  # (layers.Layer):
    def __init__(self, compression_factor=0.5, pool_size=2, **kwargs):
        # super(TransitionDown, self).__init__(self, **kwargs)
        self.concat = Concatenate()
        self.compression_factor = compression_factor
        self.pool = layers.MaxPool2D(pool_size)

    def call(self, inputs):
        concat = self.concat(inputs)
        pooled = self.pool(concat)

        n_channels = int(concat.shape[-1])
        compression_filters = int(np.round(n_channels * self.compression_factor))
        self.compression_1x1 = layers.Conv2D(
            compression_filters,
            (1, 1),
            padding="same",
            activation="selu",
            kernel_initializer="lecun_normal",
        )

        compression_1x1 = self.compression_1x1(pooled)
        outputs = [compression_1x1]
        return outputs

    def __call__(self, inputs):
        return self.call(inputs)

    # def get_config(self):
    #    config = {}
    #    base_config = super(TransitionDown, self).get_config()
    #    return dict(list(base_config.items()) + list(config.items()))


class TransitionUp:  # (layers.Layer):
    def __init__(self, compression_factor=0.5, **kwargs):
        # super(TransitionDown, self).__init__(self, **kwargs)
        self.concat = Concatenate()
        self.compression_factor = compression_factor

        self.upsample = (
            SubPixelUpscaling()
        )  # layers.UpSampling2D(interpolation='bilinear')

    def call(self, inputs):
        concat = self.concat(inputs)

        n_channels = int(concat.shape[-1])
        compression_filters = int(np.round(n_channels * self.compression_factor))
        possible_values = np.arange(0, 10000, 4)
        idx = np.argmin(np.abs(compression_filters - possible_values))
        compression_filters = possible_values[idx]

        self.compression_1x1 = layers.Conv2D(
            compression_filters,
            (1, 1),
            padding="same",
            activation="selu",
            kernel_initializer="lecun_normal",
        )

        compression_1x1 = self.compression_1x1(concat)
        upsampled = self.upsample(compression_1x1)
        outputs = [upsampled]
        return outputs

    def __call__(self, inputs):
        return self.call(inputs)

    # def get_config(self):
    #    config = {}
    #    base_config = super(TransitionDown, self).get_config()
    #    return dict(list(base_config.items()) + list(config.items()))


class FrontEnd:  # (layers.Layer):
    def __init__(
        self,
        growth_rate=64,
        n_downsample=1,
        compression_factor=0.5,
        bottleneck_factor=1,
        **kwargs
    ):
        # super(FrontEnd, self).__init__(self, **kwargs)
        self.growth_rate = growth_rate
        self.compression_factor = compression_factor
        self.bottleneck_factor = bottleneck_factor
        self.conv_7x7 = layers.Conv2D(
            growth_rate,
            (7, 7),
            strides=(2, 2),
            padding="same",
            activation="selu",
            kernel_initializer="lecun_normal",
        )
        self.n_downsample = n_downsample
        self.pool_input = SubPixelDownscaling()
        self.dense_conv = [
            DenseConvBlock(growth_rate, (idx + 1), bottleneck_factor)
            for idx in range(n_downsample)
        ]
        self.transition_down = [
            TransitionDown(compression_factor) for idx in range(n_downsample - 1)
        ]
        self.pooled_outputs = [
            TransitionDown(compression_factor, pool_size=2 ** (n_downsample - 1 - idx))
            for idx in range(n_downsample - 1)
        ]

    def call(self, inputs):
        conv_7x7 = self.conv_7x7(inputs)
        pooled_inputs = self.pool_input(inputs)
        outputs = [pooled_inputs, conv_7x7]
        residual_outputs = []
        for idx in range(self.n_downsample - 1):
            outputs = self.dense_conv[idx](outputs)
            concat_outputs = Concatenate()(outputs)
            outputs = [concat_outputs]

            # Pool each dense layer to match output size
            pooled_outputs = self.pooled_outputs[idx](outputs)
            residual_outputs.append(Concatenate()(pooled_outputs))

            outputs = self.transition_down[idx](outputs)

        outputs = self.dense_conv[-1](outputs)
        outputs = Concatenate()(outputs)
        residual_outputs.append(outputs)
        residual_outputs = [
            Compression(self.compression_factor)(res) for res in residual_outputs
        ]
        outputs = Concatenate()(residual_outputs)
        return [outputs]

    def __call__(self, inputs):
        return self.call(inputs)

    # def get_config(self):
    #    config = {'n_downsample': self.n_downsample}
    #    base_config = super(FrontEnd, self).get_config()
    #    return dict(list(base_config.items()) + list(config.items()))


class ImageNetFrontEnd:  # (layers.Layer):
    def __init__(self, input_shape, n_downsample=1, compression_factor=0.5, **kwargs):
        # super(FrontEnd, self).__init__(self, **kwargs)
        self.input_shape = input_shape
        self.compression_factor = compression_factor
        self.n_downsample = n_downsample
        if n_downsample > 3:
            raise ValueError("ImageNetFrontEnd does not support `n_downsample` > 3")

    def call(self, inputs):
        pretrained_model = DenseNet121(
            input_shape=self.input_shape, residuals=self.n_downsample
        )
        outputs = pretrained_model(inputs)
        outputs = layers.BatchNormalization()(outputs)
        outputs = layers.Activation("relu")(outputs)
        outputs = Compression(self.compression_factor)(outputs)
        return [outputs]

    def __call__(self, inputs):
        return self.call(inputs)


class DenseNet:  # (layers.Layer):
    def __init__(
        self,
        growth_rate=64,
        n_downsample=1,
        n_upsample=None,
        downsample_factor=0,
        compression_factor=0.5,
        bottleneck_factor=1,
        **kwargs
    ):
        # super(DenseNet, self).__init__(self, **kwargs)
        self.n_downsample = n_downsample
        self.growth_rate = growth_rate
        self.compression_factor = compression_factor
        self.bottleneck_factor = bottleneck_factor
        self.n_upsample = n_downsample if n_upsample is None else n_upsample
        self.transition_input = TransitionDown(compression_factor)
        self.dense_conv_down = [
            DenseConvBlock(growth_rate, (idx + downsample_factor), bottleneck_factor)
            for idx in range(1, self.n_downsample)
        ]
        self.transition_down = [
            TransitionDown(compression_factor) for idx in range(self.n_downsample - 1)
        ]
        self.dense_conv_encoded = DenseConvBlock(
            growth_rate, downsample_factor, bottleneck_factor
        )
        self.dense_conv_up = [
            DenseConvBlock(growth_rate, (idx + downsample_factor), bottleneck_factor)
            for idx in range(self.n_upsample)
        ][::-1]
        self.transition_up = [
            TransitionUp(compression_factor) for idx in range(self.n_upsample)
        ]
        self.dense_conv_output = DenseConvBlock(
            growth_rate, downsample_factor, bottleneck_factor
        )

    def call(self, inputs):
        residual_outputs = [Concatenate()(inputs)]
        outputs = self.transition_input(inputs)

        # Encoder
        for idx in range(self.n_downsample - 1):
            outputs = self.dense_conv_down[idx](outputs)
            concat_outputs = Concatenate()(outputs)
            outputs = [concat_outputs]
            residual_outputs.append(concat_outputs)
            outputs = self.transition_down[idx](outputs)
        residual_outputs.append(Concatenate()(outputs))
        outputs = self.dense_conv_encoded(outputs)

        # Compress the feature maps for residual connections
        residual_outputs = residual_outputs[::-1]
        residual_outputs = [
            Compression(self.compression_factor)(res) for res in residual_outputs
        ]

        # Decoder
        for idx in range(self.n_upsample):
            outputs.append(residual_outputs[idx])
            outputs = self.dense_conv_up[idx](outputs)
            outputs = self.transition_up[idx](outputs)
        outputs.append(residual_outputs[-1])
        outputs = self.dense_conv_output(outputs)
        return [Concatenate()(outputs)]

    def __call__(self, inputs):
        return self.call(inputs)


class OutputChannels:
    def __init__(self, n_output_channels, activation="linear", name=None, **kwargs):
        self.output_channels = layers.Conv2D(
            n_output_channels, (1, 1), padding="same", activation=activation, name=name
        )
        self.concat = Concatenate()

    def call(self, inputs):
        outputs = self.concat(inputs)
        return self.output_channels(outputs)

    def __call__(self, inputs):
        return self.call(inputs)
