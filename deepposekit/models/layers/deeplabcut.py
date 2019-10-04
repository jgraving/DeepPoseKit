# -*- coding: utf-8 -*-
"""Utilities for ImageNet data preprocessing & prediction decoding.
"""
# Modified from
# https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py
# All Modifications Copyright 2018-2019 Jacob M. Graving <jgraving@gmail.com>
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend

from deepposekit.models.layers.imagenet_resnet import MODELS as RESNET_MODELS
from deepposekit.models.layers.imagenet_mobile import MODELS as MOBILE_MODELS
from deepposekit.models.layers.imagenet_densenet import MODELS as DENSENET_MODELS
from deepposekit.models.layers.imagenet_xception import MODELS as XCEPTION_MODELS

from functools import partial


def _preprocess_symbolic_input(x, data_format, mode, **kwargs):
    """Preprocesses a tensor encoding a batch of images.
    # Arguments
        x: Input tensor, 3D or 4D.
        data_format: Data format of the image tensor.
        mode: One of "caffe", "tf" or "torch".
            - caffe: will convert the images from RGB to BGR,
                then will zero-center each color channel with
                respect to the ImageNet dataset,
                without scaling.
            - tf: will scale pixels between -1 and 1,
                sample-wise.
            - torch: will scale pixels between 0 and 1 and then
                will normalize each channel with respect to the
                ImageNet dataset.
    # Returns
        Preprocessed tensor.
    """

    if mode == "tf":
        x /= 127.5
        x -= 1.0
        return x

    if mode == "torch":
        x /= 255.0
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        if data_format == "channels_first":
            # 'RGB'->'BGR'
            if backend.ndim(x) == 3:
                x = x[::-1, ...]
            else:
                x = x[:, ::-1, ...]
        else:
            # 'RGB'->'BGR'
            x = x[..., ::-1]
        mean = [103.939, 116.779, 123.68]
        std = None

    mean_tensor = backend.constant(-np.array(mean))

    # Zero-center by mean pixel
    if backend.dtype(x) != backend.dtype(mean_tensor):
        x = backend.bias_add(
            x, backend.cast(mean_tensor, backend.dtype(x)), data_format=data_format
        )
    else:
        x = backend.bias_add(x, mean_tensor, data_format)
    if std is not None:
        x /= std
    return x


def preprocess_input(x, data_format=None, mode="caffe", **kwargs):
    """Preprocesses a tensor or Numpy array encoding a batch of images.
    # Arguments
        x: Input Numpy or symbolic tensor, 3D or 4D.
            The preprocessed data is written over the input data
            if the data types are compatible. To avoid this
            behaviour, `numpy.copy(x)` can be used.
        data_format: Data format of the image tensor/array.
        mode: One of "caffe", "tf" or "torch".
            - caffe: will convert the images from RGB to BGR,
                then will zero-center each color channel with
                respect to the ImageNet dataset,
                without scaling.
            - tf: will scale pixels between -1 and 1,
                sample-wise.
            - torch: will scale pixels between 0 and 1 and then
                will normalize each channel with respect to the
                ImageNet dataset.
    # Returns
        Preprocessed tensor or Numpy array.
    # Raises
        ValueError: In case of unknown `data_format` argument.
    """

    if data_format is None:
        data_format = backend.image_data_format()
    if data_format not in {"channels_first", "channels_last"}:
        raise ValueError("Unknown data_format " + str(data_format))
    return _preprocess_symbolic_input(x, data_format=data_format, mode=mode, **kwargs)


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
        super(ImageNetPreprocess, self).__init__(**kwargs)
        self.network = network
        if network.lower().startswith(("mobile", "xception")):
            self.preprocess_input = partial(preprocess_input, mode="tf")
        elif network.lower().startswith("resnet"):
            self.preprocess_input = partial(preprocess_input, mode="caffe")
        elif network.lower().startswith("densenet"):
            self.preprocess_input = partial(preprocess_input, mode="torch")

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
    + list(XCEPTION_MODELS.items())
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
