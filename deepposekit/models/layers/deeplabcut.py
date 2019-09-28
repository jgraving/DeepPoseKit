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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tensorflow.keras.layers import Layer

from tensorflow.python.keras.applications import resnet50
from tensorflow.python.keras.applications import mobilenet_v2
from tensorflow.python.keras.applications import densenet

from deepposekit.models.layers.deeplabcut_resnet import MODELS as RESNET_MODELS
from deepposekit.models.layers.deeplabcut_mobile import MODELS as MOBILE_MODELS
from deepposekit.models.layers.deeplabcut_densenet import MODELS as DENSENET_MODELS


class ImageNetPreprocess(Layer):
    """Preprocessing layer for ImageNet inputs.
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

    def __init__(self, network, **kwargs):
        self.network = network
        if network.startswith("mobile"):
            self.preprocess_input = mobilenet_v2.preprocess_input
        elif network.startswith("resnet"):
            self.preprocess_input = resnet50.preprocess_input
        elif network.startswith("densenet"):
            self.preprocess_input = densenet.preprocess_input

        super(ImageNetPreprocess, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        return self.preprocess_input(inputs)

    def get_config(self):
        config = {"network": self.network}
        base_config = super(ImageNetPreprocess, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


MODELS = (
    list(RESNET_MODELS.items())
    + list(MOBILE_MODELS.items())
    + list(DENSENET_MODELS.items())
)
MODELS = dict(MODELS)

if __name__ == "__main__":

    from tensorflow.keras.applications.resnet50 import preprocess_input
    from tensorflow.keras.layers import Input
    from tensorflow.keras import Model

    input_layer = Input((192, 192, 3))
    model = ResNet50(include_top=False, input_shape=(192, 192, 3))
    normalized = ImageNetPreprocess(network="resnet50")(input_layer)
    pretrained_output = model(normalized)
    model = Model(inputs=input_layer, outputs=pretrained_output)
