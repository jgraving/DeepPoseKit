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

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Add, MaxPool2D
import tensorflow.keras.backend as K

from functools import partial

import numpy as np

from deepposekit.models.engine import BaseModel
from deepposekit.models.layers.util import ImageNormalization, Float
from deepposekit.models.layers.convolutional import UpSampling2D
from deepposekit.utils import image as image_utils


class ResidualBlock:
    def __init__(self, filters, bottleneck_factor=2):
        self.filters = filters
        self.bottleneck_factor = bottleneck_factor

        conv = partial(Conv2D, activation="relu", padding="same", use_bias=False)

        self.identity_bn = BatchNormalization()
        self.identity_1x1 = conv(filters, kernel_size=(1, 1))

        self.bottleneck_1x1_bn = BatchNormalization()
        self.bottleneck_1x1 = conv(filters // bottleneck_factor, kernel_size=(1, 1))

        self.bottleneck_3x3_bn = BatchNormalization()
        self.bottleneck_3x3 = conv(filters // bottleneck_factor, kernel_size=(3, 3))

        self.expansion_1x1_bn = BatchNormalization()
        self.expansion_1x1 = conv(filters, kernel_size=(1, 1))

        self.residual_add_bn = BatchNormalization()
        self.residual_add = Add()

    def __call__(self, inputs):
        identity = inputs
        if K.int_shape(identity)[-1] == self.filters:
            identity = self.identity_bn(identity)
        else:
            identity = self.identity_bn(identity)
            identity = self.identity_1x1(identity)

        x = inputs
        x = self.bottleneck_1x1_bn(x)
        x = self.bottleneck_1x1(x)

        x = self.bottleneck_3x3_bn(x)
        x = self.bottleneck_3x3(x)

        x = self.expansion_1x1_bn(x)
        x = self.expansion_1x1(x)

        x = self.residual_add_bn(x)
        return self.residual_add([identity, x])


class FrontModule:
    def __init__(self, filters, n_downsample, bottleneck_factor=2):
        self.filters = filters
        self.bottleneck_factor = bottleneck_factor
        n_downsample = n_downsample - 1
        self.n_downsample = int(np.maximum(0, n_downsample))

        self.conv_7x7 = Conv2D(
            filters,
            (7, 7),
            strides=(2, 2),
            padding="same",
            activation="relu",
            use_bias=False,
        )

        self.res_blocks = []
        self.pool_layers = []
        for idx in range(n_downsample):
            res_block = ResidualBlock(filters, bottleneck_factor)
            max_pool = MaxPool2D(pool_size=(2, 2), strides=(2, 2))
            self.res_blocks.append(res_block)
            self.pool_layers.append(max_pool)

        self.res_output = [
            ResidualBlock(filters, bottleneck_factor),
            ResidualBlock(filters, bottleneck_factor),
        ]

    def __call__(self, inputs):
        x = inputs
        x = self.conv_7x7(x)
        for res_block, pool_layer in zip(self.res_blocks, self.pool_layers):
            x = res_block(x)
            x = pool_layer(x)
        for layer in self.res_output:
            x = layer(x)
        return x


class Output:
    def __init__(self, n_output_channels, filters):
        self.n_output_channels = n_output_channels
        self.filters = filters

        conv = partial(
            Conv2D,
            kernel_size=(1, 1),
            activation="relu",
            padding="same",
            use_bias=False,
        )

        self.input_bn = BatchNormalization()
        self.input_conv = conv(filters)
        self.loss_bn = BatchNormalization()
        self.loss_output = conv(n_output_channels, activation="linear", use_bias=True)

        self.loss_res_bn = BatchNormalization()
        self.loss_res_conv = conv(filters)

        self.conv_1x1_bn = BatchNormalization()
        self.conv_1x1 = conv(filters)

        self.res_add_bn_loss = BatchNormalization()
        self.res_add_bn_conv = BatchNormalization()
        self.res_add_bn_identity = BatchNormalization()

        self.res_add = Add()

    def __call__(self, inputs):
        x = inputs
        x = self.input_bn(x)
        x = self.input_conv(x)

        loss_x = self.loss_bn(x)
        loss_outputs = self.loss_output(loss_x)

        loss_x = self.loss_res_bn(loss_outputs)
        loss_x = self.loss_res_conv(loss_x)

        conv_x = self.conv_1x1_bn(x)
        conv_x = self.conv_1x1(conv_x)

        loss_x = self.res_add_bn_loss(loss_x)
        conv_x = self.res_add_bn_conv(conv_x)
        identity = self.res_add_bn_identity(inputs)

        res_outputs = self.res_add([loss_x, conv_x, identity])

        return [loss_outputs, res_outputs]


class Hourglass:
    def __init__(self, filters, bottleneck_factor, n_downsample, n_upsample=None):
        self.filters = filters
        self.bottleneck_factor = bottleneck_factor
        self.n_downsample = n_downsample
        if n_upsample:
            self.n_upsample = n_upsample
        else:
            self.n_upsample = n_downsample

        self.down_res_blocks = []
        self.pool_layers = []
        for idx in range(self.n_downsample):
            res_block = ResidualBlock(filters, bottleneck_factor)
            max_pool = MaxPool2D(pool_size=(2, 2), strides=(2, 2))
            self.down_res_blocks.append(res_block)
            self.pool_layers.append(max_pool)

        self.skip_bottleneck = ResidualBlock(filters, bottleneck_factor)
        self.bottleneck_layers = [
            ResidualBlock(filters, bottleneck_factor),
            ResidualBlock(filters, bottleneck_factor),
            ResidualBlock(filters, bottleneck_factor),
        ]
        self.bottleneck_identity_bn = BatchNormalization()
        self.bottleneck_skip_bn = BatchNormalization()
        self.bottleneck_add = Add()

        self.up_res_blocks = []
        self.skip_res_blocks = []
        self.upsample_layers = []
        self.skip_bn_layers = []
        self.add_bn_layers = []
        self.add_layers = []
        for idx in range(self.n_upsample):
            res_block = ResidualBlock(filters, bottleneck_factor)
            self.up_res_blocks.append(res_block)

            res_block = ResidualBlock(filters, bottleneck_factor)
            self.skip_res_blocks.append(res_block)

            upsample = UpSampling2D(size=(2, 2))
            self.upsample_layers.append(upsample)

            add_layer = Add()
            self.add_layers.append(add_layer)

            bn_layer = BatchNormalization()
            self.skip_bn_layers.append(bn_layer)

            bn_layer = BatchNormalization()
            self.add_bn_layers.append(bn_layer)

    def __call__(self, inputs):
        x = inputs
        skip_connections = [x]

        for res_block, pool_layer in zip(self.down_res_blocks, self.pool_layers):
            x = res_block(x)
            skip_connections.append(x)
            x = pool_layer(x)

        x_identity = self.skip_bottleneck(x)

        x_bottleneck = x_identity
        for layer in self.bottleneck_layers:
            x_bottleneck = layer(x_bottleneck)

        x_identity = self.bottleneck_identity_bn(x_identity)
        x_bottleneck = self.bottleneck_skip_bn(x_bottleneck)
        x = self.bottleneck_add([x_identity, x_bottleneck])

        skip_connections = skip_connections[::-1]

        up_layers = zip(
            skip_connections,
            self.skip_res_blocks,
            self.skip_bn_layers,
            self.add_layers,
            self.add_bn_layers,
            self.up_res_blocks,
            self.upsample_layers,
        )
        for (
            skip_x,
            skip_res_block,
            skip_bn,
            add_layer,
            add_bn,
            up_res_block,
            upsample_layer,
        ) in up_layers:
            skip_x = skip_res_block(skip_x)
            skip_x = skip_bn(skip_x)
            x = upsample_layer(x)
            x = add_bn(x)
            x = add_layer([skip_x, x])
            x = up_res_block(x)

        return x


class StackedHourglass(BaseModel):
    def __init__(
        self,
        train_generator,
        n_stacks=1,
        n_transitions=-1,
        filters=256,
        bottleneck_factor=2,
        subpixel=True,
        **kwargs
    ):
        """
        Define a Stacked Hourglass model for pose estimation.
        See `References` for details on the model architecture.

        Parameters
        ----------
        train_generator : class deepposekit.io.TrainingGenerator
            A deepposekit.io.TrainingGenerator class for generating
            images and confidence maps.
        n_stacks : int, default = 1
            The number of hourglass networks to stack
            with intermediate supervision between stacks
        n_transitions : int, default = -1
            The number of transition layers (downsampling and upsampling)
            in each encoder-decoder stack. If value is <0
            the number of transitions will be automatically set
            based on image size as the maximum number of possible
            transitions minus n_transitions plus 1, or:
            n_transitions = max_transitions - n_transitions + 1.
            The default is -1, which uses the maximum number of
            transitions possible.
        bottleneck_factor : int, default = 4
            The factor for determining the number of input channels
            to 3x3 convolutional layer in each convolutional block.
            Inputs are first passed through a 1x1 convolutional layer to
            reduce the number of channels to:
            filters // bottleneck_factor
        subpixel: bool, default = True
            Whether to use subpixel maxima for calculating
            keypoint coordinates in the prediction model.

        Attributes
        -------
        train_model: keras.Model
            A model for training the network to produce confidence maps with
            one input layer for images and `n_outputs` output layers for training
            with intermediate supervision
        predict_model: keras.Model
            A model for predicting keypoint coordinates with one input and one output
            using with Maxima2D or SubpixelMaxima2D layers at the output of the network.

        Both of these models share the same computational graph, so training train_model
        updates the weights of predict_model

        References
        ----------
        [1] Newell, A., Yang, K., & Deng, J. (2016). Stacked hourglass networks
            for human pose estimation. In European Conference on Computer
            Vision (pp. 483-499). Springer, Cham.
        """

        self.n_stacks = n_stacks
        self.filters = filters
        self.bottleneck_factor = bottleneck_factor
        self.n_transitions = n_transitions
        super().__init__(train_generator, subpixel, **kwargs)

    def __init_model__(self):

        max_transitions = np.min(
            [
                image_utils.n_downsample(self.train_generator.height),
                image_utils.n_downsample(self.train_generator.width),
            ]
        )

        n_transitions = self.n_transitions
        if isinstance(self.n_transitions, (int, np.integer)):
            if n_transitions == 0:
                raise ValueError("n_transitions cannot equal zero")
            if n_transitions < 0:
                n_transitions += 1
                n_transitions = max_transitions - np.abs(n_transitions)
                self.n_transitions = n_transitions
            elif 0 < n_transitions <= max_transitions:
                self.n_transitions = n_transitions
            else:
                raise ValueError(
                    "n_transitions must be in range {0} "
                    "< n_transitions <= "
                    "{1}".format(-max_transitions + 1, max_transitions)
                )
        else:
            raise TypeError(
                "n_transitions must be integer in range "
                "{0} < n_transitions <= "
                "{1}".format(-max_transitions + 1, max_transitions)
            )
        if n_transitions <= self.train_generator.downsample_factor:
            raise ValueError(
                "`n_transitions` <= `downsample_factor`. Increase `n_transitions` or decrease `downsample_factor`."
                " If `n_transitions` is -1 (the default), check that your image resolutions can be repeatedly downsampled (are divisible by 2 repeatedly)."
            )

        batch_shape = (
            None,
            self.train_generator.height,
            self.train_generator.width,
            self.train_generator.n_channels,
        )

        input_layer = Input(batch_shape=batch_shape, dtype="uint8")
        to_float = Float()(input_layer)
        normalized = ImageNormalization()(to_float)

        n_downsample = self.train_generator.downsample_factor
        front_module = FrontModule(self.filters, n_downsample, self.bottleneck_factor)
        front_output = front_module(normalized)

        n_transitions = self.n_transitions - n_downsample
        x = front_output
        outputs = []
        for idx in range(self.n_stacks):
            x = Hourglass(self.filters, self.bottleneck_factor, n_transitions)(x)
            outputs_x, x = Output(self.train_generator.n_output_channels, self.filters)(
                x
            )
            outputs.append(outputs_x)

        self.train_model = Model(input_layer, outputs, name=self.__class__.__name__)

    def get_config(self):
        config = {
            "name": self.__class__.__name__,
            "n_stacks": self.n_stacks,
            "n_transitions": self.n_transitions,
            "bottleneck_factor": self.bottleneck_factor,
            "filters": self.filters,
            "subpixel": self.subpixel,
        }
        base_config = super(StackedHourglass, self).get_config()
        return dict(list(config.items()) + list(base_config.items()))
