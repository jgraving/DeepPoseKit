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

__all__ = ["ConvBlock2D", "ConvPool2D"]


class ConvBlock2D:
    def __init__(
        self,
        n_layers,
        filters,
        kernel_size,
        activation,
        initializer="glorot_uniform",
        batchnorm=False,
        use_bias=True,
        name=None,
    ):
        self.n_layers = n_layers
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.initializer = initializer
        self.use_bias = use_bias
        if activation.lower() is not "selu" and batchnorm:
            self.batchnorm = True
        else:
            self.batchnorm = False
        if activation.lower() is "selu":
            self.initializer = "lecun_normal"
        self.name = name

    def __call__(self, inputs):
        outputs = inputs
        for idx in range(self.n_layers):
            outputs = layers.Conv2D(
                self.filters,
                self.kernel_size,
                padding="same",
                activation=self.activation,
                use_bias=self.use_bias,
                kernel_initializer=self.initializer,
                name=self.name,
            )(outputs)
            if self.batchnorm:
                outputs = layers.BatchNormalization()(outputs)
        return outputs


class ConvPool2D:
    def __init__(
        self,
        n_layers,
        filters,
        kernel_size,
        activation,
        pooling="max",
        initializer="glorot_uniform",
        batchnorm=False,
        use_bias=True,
        name=None,
    ):
        self.n_layers = n_layers
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.initializer = initializer
        self.use_bias = use_bias
        self.pooling = pooling
        if activation.lower() is not "selu" and batchnorm:
            self.batchnorm = True
        else:
            self.batchnorm = False
        if activation.lower() is "selu":
            self.initializer = "lecun_normal"
        if pooling is "average":
            self.Pooling2D = layers.AveragePooling2D
        else:
            self.Pooling2D = layers.MaxPooling2D
        self.name = name

    def __call__(self, inputs):
        outputs = ConvBlock2D(
            self.n_layers,
            self.filters,
            self.kernel_size,
            self.activation,
            self.initializer,
            self.batchnorm,
            self.use_bias,
        )(inputs)
        outputs = self.Pooling2D()(outputs)

        return outputs
