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
from tensorflow.keras.layers import Conv2DTranspose, Conv2D, Concatenate
from deepposekit.models.layers.util import Float
from deepposekit.models.layers.deeplabcut import ResNet50, ResNetPreprocess
from deepposekit.models.engine import BaseModel


class DeepLabCut(BaseModel):
    def __init__(self, data_generator, subpixel=True, weights="imagenet", **kwargs):
        """
        Define a DeepLabCut model from Mathis et al., 2018 [1]
        See `References` for details on the model architecture.

        Parameters
        ----------
        data_generator : class pose.DataGenerator
            A pose.DataGenerator class for generating
            images and confidence maps.
        subpixel: bool, default = True
            Whether to use subpixel maxima for calculating
            keypoint coordinates in the prediction model.

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
        [1] Mathis, A., Mamidanna, P., Cury, K. M., Abe, T., Murthy, V. N.,
            Mathis, M. W., & Bethge, M. (2018). DeepLabCut: markerless pose
            estimation of user-defined body parts with deep learning (p. 1).
            Nature Publishing Group.

        """
        self.subpixel = subpixel
        self.weights = weights
        super(DeepLabCut, self).__init__(data_generator, subpixel, **kwargs)

    def __init_model__(self):

        batch_shape = (
            None,
            self.data_generator.height,
            self.data_generator.width,
            self.data_generator.n_channels,
        )

        input_layer = Input(batch_shape=batch_shape, dtype="uint8")
        to_float = Float()(input_layer)
        if batch_shape[-1] is 1:
            to_float = Concatenate()([to_float] * 3)
        normalized = ResNetPreprocess()(to_float)
        pretrained_model = ResNet50(
            include_top=False,
            weights=self.weights,
            input_shape=(self.data_generator.height, self.data_generator.width, 3),
        )
        pretrained_features = pretrained_model(normalized)
        if self.data_generator.downsample_factor is 4:
            x = pretrained_features
            x_out = Conv2D(self.data_generator.n_output_channels, (1, 1))(x)
        elif self.data_generator.downsample_factor is 3:
            x = pretrained_features
            x_out = Conv2DTranspose(
                self.data_generator.n_output_channels,
                (3, 3),
                strides=(2, 2),
                padding="same",
            )(x)
        elif self.data_generator.downsample_factor is 2:
            x = pretrained_features
            x = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding="same")(x)
            x_out = Conv2DTranspose(
                self.data_generator.n_output_channels,
                (3, 3),
                strides=(2, 2),
                padding="same",
            )(x)
        elif self.data_generator.downsample_factor is 1:
            x = pretrained_features
            x = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding="same")(x)
            x = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding="same")(x)
            x_out = Conv2DTranspose(
                self.data_generator.n_output_channels,
                (3, 3),
                strides=(2, 2),
                padding="same",
            )(x)
        elif self.data_generator.downsample_factor is 0:
            x = pretrained_features
            x = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding="same")(x)
            x = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding="same")(x)
            x = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same")(x)
            x_out = Conv2DTranspose(
                self.data_generator.n_output_channels,
                (3, 3),
                strides=(2, 2),
                padding="same",
            )(x)
        else:
            raise ValueError("This downsample factor is not supported for DeepLabCut")

        self.train_model = Model(input_layer, x_out, name=self.__class__.__name__)

    def get_config(self):
        config = {
            "name": self.__class__.__name__,
            "subpixel": self.subpixel,
            "weights": self.weights,
        }
        base_config = super(DeepLabCut, self).get_config()
        return dict(list(config.items()) + list(base_config.items()))
