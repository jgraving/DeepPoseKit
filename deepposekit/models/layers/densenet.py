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

import numpy as np

from tensorflow.keras import layers

from .convolutional import UpSampling2D
from .util import ImageNormalization
from .squeeze_excitation import squeeze_excite_block
from .convolutional import SubPixelDownscaling, SubPixelUpscaling

__all__ = [
    "ConvBatchNorm2D",
    "Concatenate",
    "DenseConv2D",
    "DenseBlock",
    "TransitionDown",
    "TransitionUp",
    "DenseNet",
]


class ConvBatchNorm2D:
    def __init__(
        self,
        filters,
        kernel_size,
        activation,
        initializer="glorot_uniform",
        batchnorm=True,
        use_bias=False,
        name=None,
        strides=1,
        separable=False,
    ):
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.initializer = initializer
        self.use_bias = use_bias
        self.strides = strides
        if activation != "selu" and batchnorm:
            self.batchnorm = True
        else:
            self.batchnorm = False
        if activation.lower() == "selu":
            self.initializer = "lecun_normal"
        elif activation.lower() == "linear":
            self.initializer = "glorot_uniform"
        if separable:
            self.conv2d = layers.SeparableConv2D(
                self.filters,
                self.kernel_size,
                padding="same",
                activation=self.activation,
                use_bias=self.use_bias,
                strides=self.strides,
                depthwise_initializer=self.initializer,
                pointwise_initializer=self.initializer,
                name=name,
            )
        else:
            self.conv2d = layers.Conv2D(
                self.filters,
                self.kernel_size,
                padding="same",
                activation="linear",
                use_bias=self.use_bias,
                strides=self.strides,
                kernel_initializer=self.initializer,
                name=name,
            )
        if self.batchnorm:
            self.batch_normalization = layers.BatchNormalization()
        self.activate = layers.Activation(self.activation)

    def __call__(self, inputs):
        if self.batchnorm:
            inputs = self.batch_normalization(inputs)
            if self.activation is not "linear":
                inputs = self.activate(inputs)
            outputs = self.conv2d(inputs)
        else:
            outputs = self.conv2d(inputs)
            if self.activation is not "linear":
                outputs = self.activate(outputs)

        return outputs


class Concatenate:
    def __init__(self):
        pass

    def __call__(self, inputs):
        if isinstance(inputs, list):
            if len(inputs) > 1:
                outputs = layers.Concatenate()(inputs)
            else:
                outputs = inputs[0]
        else:
            raise TypeError("""inputs must be a list""")
        outputs_list = [outputs]
        return outputs_list, outputs


class DenseConv2D:
    def __init__(
        self,
        growth_rate,
        activation,
        bottleneck_factor,
        initializer,
        batchnorm,
        use_bias,
        separable=False,
    ):
        self.growth_rate = growth_rate
        self.activation = activation
        self.bottleneck_factor = bottleneck_factor
        self.bottleneck_filters = int(np.round(growth_rate * bottleneck_factor))
        self.initializer = initializer
        self.batchnorm = batchnorm
        self.use_bias = use_bias
        self.separable = separable

    def __call__(self, inputs):
        outputs_list, inputs = Concatenate()(inputs)
        n_channels = inputs.shape[-1]
        bottleneck = inputs
        if n_channels > self.bottleneck_filters:
            bottleneck = ConvBatchNorm2D(
                self.bottleneck_filters,
                (1, 1),
                self.activation,
                self.initializer,
                self.batchnorm,
                self.use_bias,
            )(bottleneck)
        outputs = ConvBatchNorm2D(
            self.growth_rate,
            (3, 3),
            self.activation,
            self.initializer,
            self.batchnorm,
            self.use_bias,
            separable=self.separable,
        )(bottleneck)

        outputs_list.append(outputs)
        return outputs_list


class DenseBlock:
    def __init__(
        self,
        n_layers,
        growth_rate,
        activation,
        bottleneck_factor,
        initializer,
        batchnorm,
        use_bias,
        separable=False,
    ):
        self.n_layers = n_layers
        self.dense_conv2d = DenseConv2D(
            growth_rate,
            activation,
            bottleneck_factor,
            initializer,
            batchnorm,
            use_bias,
            separable,
        )

    def __call__(self, inputs):
        outputs_list = self.dense_conv2d(inputs)
        for idx in range(self.n_layers - 1):
            outputs_list = self.dense_conv2d(outputs_list)
        return outputs_list


class TransitionDown:
    def __init__(
        self,
        compression_factor,
        activation,
        pooling,
        initializer,
        batchnorm,
        use_bias,
        pool_size=2,
        squeeze_excite=False,
    ):
        self.compression_factor = compression_factor
        self.activation = activation
        self.pooling = pooling
        self.initializer = initializer
        self.batchnorm = batchnorm
        self.use_bias = use_bias
        self.pool_size = pool_size
        self.squeeze_excite = squeeze_excite

    def __call__(self, inputs):
        outputs_list, inputs = Concatenate()(inputs)

        n_channels = int(inputs.shape[-1])

        if self.pooling:
            if self.pooling.lower().startswith("average"):
                self.pool2d = layers.AveragePooling2D(self.pool_size)
            elif self.pooling.lower().startswith("max"):
                self.pool2d = layers.MaxPooling2D(self.pool_size)
            elif self.pooling.lower().startswith("subpixel"):
                self.pool2d = SubPixelDownscaling(self.pool_size)
            pool = self.pool2d(inputs)
        else:
            pool = inputs
        if n_channels > 3:
            compression_filters = int(np.round(n_channels * self.compression_factor))
        else:
            compression_filters = n_channels

        compression = pool
        if n_channels > compression_filters:
            compression = ConvBatchNorm2D(
                compression_filters,
                (1, 1),
                self.activation,
                self.initializer,
                self.batchnorm,
                self.use_bias,
            )(compression)
            if self.squeeze_excite:
                compression = squeeze_excite_block(compression)

        outputs_list = [compression]
        return outputs_list


class TransitionUp:
    def __init__(
        self,
        compression_factor,
        activation,
        initializer,
        batchnorm,
        use_bias,
        interpolation="nearest",
        squeeze_excite=False,
    ):
        self.compression_factor = compression_factor
        self.activation = activation
        self.initializer = initializer
        self.batchnorm = batchnorm
        self.use_bias = use_bias
        self.interpolation = interpolation
        self.squeeze_excite = squeeze_excite
        if self.interpolation is "subpixel":
            self.compression_factor = 1

    def __call__(self, inputs):
        outputs_list, inputs = Concatenate()(inputs)

        n_channels = int(inputs.shape[-1])
        if n_channels > 3:
            compression_filters = int(np.round(n_channels * self.compression_factor))
        else:
            compression_filters = n_channels
        if self.interpolation is "subpixel":
            # compression_filters *= 4
            possible_values = np.arange(0, 10000, 4)
            idx = np.argmin(np.abs(compression_filters - possible_values))
            compression_filters = possible_values[idx]
        compression = inputs
        if n_channels != compression_filters:
            compression = ConvBatchNorm2D(
                compression_filters,
                (1, 1),
                self.activation,
                self.initializer,
                self.batchnorm,
                self.use_bias,
            )(compression)
            if self.squeeze_excite:
                compression = squeeze_excite_block(compression)

        if self.interpolation is "subpixel":
            upsampled = SubPixelUpscaling()(compression)
        else:
            upsampled = UpSampling2D(interpolation=self.interpolation)(compression)

        outputs_list = [upsampled]
        return outputs_list


class DenseNet:
    def __init__(
        self,
        n_output_channels,
        n_downsample,
        n_upsample,
        n_layers,
        growth_rate,
        bottleneck_factor=4,
        compression_factor=1,
        batchnorm=True,
        use_bias=False,
        activation="relu",
        pooling="average",
        interpolation="nearest",
        initializer="glorot_uniform",
        separable=False,
        squeeze_excite=False,
        stack_idx=0,
        multiplier=1,
    ):
        self.n_output_channels = n_output_channels
        self.n_downsample = n_downsample
        self.n_upsample = n_upsample
        self.n_layers = n_layers
        self.growth_rate = growth_rate
        self.bottleneck_factor = bottleneck_factor
        self.compression_factor = compression_factor
        self.batchnorm = batchnorm
        self.use_bias = use_bias
        self.activation = activation
        self.pooling = pooling
        self.interpolation = interpolation
        self.initializer = initializer
        self.separable = separable
        self.squeeze_excite = squeeze_excite
        self.transition_down = TransitionDown(
            compression_factor,
            activation,
            pooling,
            initializer,
            batchnorm,
            use_bias,
            squeeze_excite=squeeze_excite,
        )
        self.transition_up = TransitionUp(
            compression_factor,
            activation,
            initializer,
            batchnorm,
            use_bias,
            interpolation=interpolation,
            squeeze_excite=squeeze_excite,
        )
        if self.pooling.lower().startswith("max"):
            self.Pooling2D = layers.MaxPooling2D
        elif self.pooling.lower().startswith("average"):
            self.Pooling2D = layers.AveragePooling2D
        self.stack_idx = stack_idx
        self.multiplier = multiplier
        if stack_idx == 0:
            self.n_downsample -= 1

    def __call__(self, inputs):
        outputs_list = inputs
        down_list = []
        if self.stack_idx == 0:
            down_list.append(outputs_list)
            if self.batchnorm:
                init_activation = "linear"
            else:
                init_activation = self.activation
            downsampled = ConvBatchNorm2D(
                filters=int(
                    np.round(self.growth_rate * self.n_layers * self.bottleneck_factor)
                ),
                kernel_size=(7, 7),
                activation=init_activation,
                initializer=self.initializer,
                batchnorm=False,
                use_bias=self.use_bias,
                strides=2,
                separable=False,
            )(outputs_list[0])
            if self.squeeze_excite:
                downsampled = squeeze_excite_block(downsampled)
            outputs_list = TransitionDown(
                1,
                self.activation,
                self.pooling,
                self.initializer,
                self.batchnorm,
                False,
                squeeze_excite=False,
            )(outputs_list)
            outputs_list.append(downsampled)
            down_list.append(outputs_list)

        for idx in range(self.n_downsample):
            outputs_list = DenseBlock(
                self.n_layers * self.multiplier,
                self.growth_rate,
                self.activation,
                self.bottleneck_factor,
                self.initializer,
                self.batchnorm,
                self.use_bias,
                self.separable,
            )(outputs_list)

            if self.stack_idx == 0 and idx == 0:
                [down_list[-1].append(item) for item in outputs_list]
            else:
                down_list.append(outputs_list)
            outputs_list = self.transition_down(outputs_list)
            if self.multiplier < 1:
                self.multiplier += 1
        for idx in range(self.n_upsample):
            outputs_list = DenseBlock(
                self.n_layers * self.multiplier,
                self.growth_rate,
                self.activation,
                self.bottleneck_factor,
                self.initializer,
                self.batchnorm,
                self.use_bias,
                self.separable,
            )(outputs_list)
            outputs_list = self.transition_up(outputs_list)
            if idx + 1 <= len(down_list):
                [
                    outputs_list.append(x)
                    for x in TransitionDown(
                        self.compression_factor,
                        self.activation,
                        None,
                        self.initializer,
                        self.batchnorm,
                        self.use_bias,
                        squeeze_excite=self.squeeze_excite,
                    )(down_list[-1 * (idx + 1)])
                ]
            if self.multiplier > 1:
                self.multiplier -= 1
        transition_diff = len(down_list) + -1 * (idx + 1)
        for idx in range(transition_diff + 1):
            pool_size = int(2 ** (transition_diff - idx))
            if pool_size > 1:
                [
                    outputs_list.append(x)
                    for x in TransitionDown(
                        self.compression_factor,
                        self.activation,
                        self.pooling,
                        self.initializer,
                        self.batchnorm,
                        self.use_bias,
                        pool_size=pool_size,
                        squeeze_excite=self.squeeze_excite,
                    )(down_list[idx])
                ]
            else:
                [
                    outputs_list.append(x)
                    for x in TransitionDown(
                        self.compression_factor,
                        self.activation,
                        None,
                        self.initializer,
                        self.batchnorm,
                        self.use_bias,
                        squeeze_excite=self.squeeze_excite,
                    )(down_list[idx])
                ]

        outputs_list = DenseBlock(
            self.n_layers * self.multiplier,
            self.growth_rate,
            self.activation,
            self.bottleneck_factor,
            self.initializer,
            self.batchnorm,
            self.use_bias,
            self.separable,
        )(outputs_list)

        outputs_list = TransitionDown(
            self.compression_factor,
            self.activation,
            None,
            self.initializer,
            self.batchnorm,
            self.use_bias,
            squeeze_excite=self.squeeze_excite,
        )(outputs_list)
        outputs = outputs_list[-1]

        if self.batchnorm:
            outputs = ConvBatchNorm2D(
                self.n_output_channels,
                (1, 1),
                self.activation,
                self.initializer,
                self.batchnorm,
                True,
                name="output_" + str(self.stack_idx),
            )(outputs)
        else:
            outputs = ConvBatchNorm2D(
                self.n_output_channels,
                (1, 1),
                "linear",
                self.initializer,
                self.batchnorm,
                True,
                name="output_" + str(self.stack_idx),
            )(outputs)
        normalized = ImageNormalization()(outputs)
        outputs_list.append(normalized)

        return outputs_list, outputs
