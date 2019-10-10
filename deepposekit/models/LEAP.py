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

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization

from deepposekit.models.layers.convolutional import UpSampling2D
from deepposekit.models.layers.util import ImageNormalization
from deepposekit.models.layers.leap import ConvBlock2D, ConvPool2D
from deepposekit.models.engine import BaseModel


class LEAP(BaseModel):
    def __init__(
        self,
        train_generator,
        filters=64,
        upsampling=False,
        activation="relu",
        batchnorm=False,
        use_bias=True,
        pooling="max",
        interpolation="bilinear",
        subpixel=False,
        initializer="glorot_uniform",
        **kwargs
    ):
        """
        Define a LEAP model from Pereira et al., 2018 [1]
        See `References` for details on the model architecture.

        Parameters
        ----------
        train_generator : class deepposekit.io.TrainingGenerator
            A deepposekit.io.TrainingGenerator class for generating
            images and confidence maps.
        filters : int, default = 64
            The base number of channels to output from each
            convolutional layer. Increases by up to a factor
            of 4 with network depth.
        upsampling_layers: bool, default = False
            Whether to use upsampling or transposed convolutions
            for upsampling layers. Default is False, which uses
            transposed convolutions.
        activation: str or callable, default = 'relu'
            The activation function to use for each convolutional layer.
        batchnorm : bool, default = False
            Whether to use batch normalization in each convolutional block.
            If activation is 'selu' then batchnorm is automatically set to
            False, as the network is already self-normalizing.
        pooling: str, default = 'max'
            The type of pooling to use during downsampling.
            Must be either 'max' or 'average'.
        interpolation: str, default = 'nearest'
            The type of interpolation to use when upsampling.
            Must be 'nearest', 'bilinear', or 'bicubic'.
            The default is 'nearest', which is the most efficient.
        subpixel: bool, default = True
            Whether to use subpixel maxima for calculating
            keypoint coordinates in the prediction model.
        initializer: str or callable, default='glorot_uniform'
            The initializer for the convolutional kernels.
            Default is 'glorot_uniform' which is the keras default.
            If activation is 'selu', the initializer is automatically
            changed to 'lecun_normal', which is the recommended initializer
            for that activation function [4].

        Attributes
        -------
        train_model: keras.Model
            A model for training the network to produce confidence maps with
            one input layer for images
        predict_model: keras.Model
            A model for predicting keypoint coordinates using with Maxima2D or
            SubpixelMaxima2D layers at the output of the network.

        Both of these models share the same computational graph,
        so training train_model updates the weights of predict_model

        References
        ----------
        1.  Pereira, T. D., Aldarondo, D. E., Willmore, L., Kislin, M.,
            Wang, S. S. H., Murthy, M., & Shaevitz, J. W. (2019).
            Fast animal pose estimation using deep neural networks.
            Nature methods, 16(1), 117.


        """
        self.filters = filters
        self.upsampling = upsampling
        self.activation = activation
        if activation is "selu":
            batchnorm = False
            use_bias = False
        if batchnorm:
            use_bias = False
        self.batchnorm = batchnorm
        self.use_bias = use_bias
        self.pooling = pooling
        self.interpolation = interpolation
        self.subpixel = subpixel
        self.initializer = initializer
        super(LEAP, self).__init__(train_generator, subpixel, **kwargs)

    def __init_model__(self):
        if self.train_generator.downsample_factor is not 0:
            raise ValueError("LEAP is only compatible with a downsample_factor of 0")
        normalized = ImageNormalization()(self.inputs)

        x1 = ConvPool2D(
            n_layers=3,
            filters=self.filters,
            kernel_size=3,
            pooling=self.pooling,
            activation=self.activation,
            initializer=self.initializer,
            batchnorm=self.batchnorm,
            use_bias=self.use_bias,
        )(normalized)
        x2 = ConvPool2D(
            n_layers=3,
            filters=self.filters * 2,
            kernel_size=3,
            pooling=self.pooling,
            activation=self.activation,
            initializer=self.initializer,
            batchnorm=self.batchnorm,
            use_bias=self.use_bias,
        )(x1)
        x3 = ConvBlock2D(
            n_layers=3,
            filters=self.filters * 4,
            kernel_size=3,
            activation=self.activation,
            initializer=self.initializer,
            batchnorm=self.batchnorm,
            use_bias=self.use_bias,
        )(x2)

        if self.upsampling:
            x4 = UpSampling2D(interpolation=self.interpolation)(x3)
        else:
            x4 = Conv2DTranspose(
                self.filters * 2,
                kernel_size=3,
                strides=2,
                padding="same",
                activation=self.activation,
                kernel_initializer="glorot_normal",
                use_bias=self.use_bias,
            )(x3)
        if self.batchnorm:
            x4 = BatchNormalization()(x4)

        x4 = ConvBlock2D(
            n_layers=2,
            filters=self.filters * 2,
            kernel_size=3,
            activation=self.activation,
            initializer=self.initializer,
            batchnorm=self.batchnorm,
            use_bias=self.use_bias,
        )(x4)

        if self.upsampling:
            x_out = UpSampling2D(interpolation=self.interpolation)(x4)
            x_out = Conv2D(
                self.train_generator.n_output_channels,
                kernel_size=3,
                padding="same",
                activation="linear",
            )(x_out)
        else:
            x_out = Conv2DTranspose(
                self.train_generator.n_output_channels,
                kernel_size=3,
                strides=2,
                padding="same",
                activation="linear",
                kernel_initializer="glorot_normal",
            )(x4)

        self.train_model = Model(self.inputs, x_out, name=self.__class__.__name__)

    def get_config(self):
        config = {
            "name": self.__class__.__name__,
            "filters": self.filters,
            "batchnorm": self.batchnorm,
            "upsampling": self.upsampling,
            "use_bias": self.use_bias,
            "activation": self.activation,
            "pooling": self.pooling,
            "interpolation": self.interpolation,
            "subpixel": self.subpixel,
            "initializer": self.initializer,
        }
        base_config = super(LEAP, self).get_config()
        return dict(list(config.items()) + list(base_config.items()))
