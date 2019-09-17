# -*- coding: utf-8 -*-
"""ResNet50 model for Keras.
# Reference:
- [Deep Residual Learning for Image Recognition](
    https://arxiv.org/abs/1512.03385) (CVPR 2016 Best Paper Award)
Adapted from code contributed by BigMoyan.

Modified by Jacob M. Graving from:
https://github.com/keras-team/keras-applications/blob/
master/keras_applications/resnet50.py

to match the stride 16 ResNet found here:
https://github.com/tensorflow/tensorflow/blob/
master/tensorflow/contrib/slim/python/slim/nets/resnet_v1.py

All modifications are Copyright 2018 Jacob M. Graving <jgraving@gmail.com>

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
from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Layer

# from keras.applications.imagenet_utils import decode_predictions

import os
import warnings
import tensorflow.keras as keras

# from keras import get_submodules_from_kwargs

_KERAS_BACKEND = keras.backend
_KERAS_LAYERS = keras.layers
_KERAS_MODELS = keras.models
_KERAS_UTILS = keras.utils


def set_keras_submodules(
    backend=None, layers=None, models=None, utils=None, engine=None
):
    # Deprecated, will be removed in the future.
    global _KERAS_BACKEND
    global _KERAS_LAYERS
    global _KERAS_MODELS
    global _KERAS_UTILS
    _KERAS_BACKEND = backend
    _KERAS_LAYERS = layers
    _KERAS_MODELS = models
    _KERAS_UTILS = utils


def get_keras_submodule(name):
    # Deprecated, will be removed in the future.
    if name not in {"backend", "layers", "models", "utils"}:
        raise ImportError(
            'Can only retrieve one of "backend", '
            '"layers", "models", or "utils". '
            "Requested: %s" % name
        )
    if _KERAS_BACKEND is None:
        raise ImportError(
            "You need to first `import keras` "
            "in order to use `keras_applications`. "
            "For instance, you can do:\n\n"
            "```\n"
            "import keras\n"
            "from keras_applications import vgg16\n"
            "```\n\n"
            "Or, preferably, this equivalent formulation:\n\n"
            "```\n"
            "from keras import applications\n"
            "```\n"
        )
    if name == "backend":
        return _KERAS_BACKEND
    elif name == "layers":
        return _KERAS_LAYERS
    elif name == "models":
        return _KERAS_MODELS
    elif name == "utils":
        return _KERAS_UTILS


def get_submodules_from_kwargs(kwargs):
    backend = kwargs.get("backend", _KERAS_BACKEND)
    layers = kwargs.get("layers", _KERAS_LAYERS)
    models = kwargs.get("models", _KERAS_MODELS)
    utils = kwargs.get("utils", _KERAS_UTILS)
    for key in kwargs.keys():
        if key not in ["backend", "layers", "models", "utils"]:
            raise TypeError("Invalid keyword argument: %s", key)
    return backend, layers, models, utils


def correct_pad(backend, inputs, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.
    # Arguments
        input_size: An integer or tuple/list of 2 integers.
        kernel_size: An integer or tuple/list of 2 integers.
    # Returns
        A tuple.
    """
    img_dim = 2 if backend.image_data_format() == "channels_first" else 1
    input_size = backend.int_shape(inputs)[img_dim : (img_dim + 2)]

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    return ((correct[0] - adjust[0], correct[0]), (correct[1] - adjust[1], correct[1]))


_obtain_input_shape = imagenet_utils.imagenet_utils._obtain_input_shape
preprocess_input = imagenet_utils.preprocess_input

WEIGHTS_PATH = (
    "https://github.com/fchollet/deep-learning-models/"
    "releases/download/v0.2/"
    "resnet50_weights_tf_dim_ordering_tf_kernels.h5"
)
WEIGHTS_PATH_NO_TOP = (
    "https://github.com/fchollet/deep-learning-models/"
    "releases/download/v0.2/"
    "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
)

backend = None
layers = None
models = None
keras_utils = None


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if backend.image_data_format() == "channels_last":
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"
    dilation = 2 if stage is 5 else 1  # modify for stride 16
    x = layers.Conv2D(
        filters1, (1, 1), kernel_initializer="he_normal", name=conv_name_base + "2a"
    )(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + "2a")(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(
        filters2,
        kernel_size,
        dilation_rate=dilation,
        padding="same",
        kernel_initializer="he_normal",
        name=conv_name_base + "2b",
    )(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + "2b")(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(
        filters3, (1, 1), kernel_initializer="he_normal", name=conv_name_base + "2c"
    )(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + "2c")(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation("relu")(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.
    # Returns
        Output tensor for the block.
    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    if backend.image_data_format() == "channels_last":
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    dilation = 2 if stage is 5 else 1  # modify for stride 16
    x = layers.Conv2D(
        filters1,
        (1, 1),
        strides=strides,
        kernel_initializer="he_normal",
        name=conv_name_base + "2a",
    )(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + "2a")(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(
        filters2,
        kernel_size,
        padding="same",
        kernel_initializer="he_normal",
        dilation_rate=dilation,
        name=conv_name_base + "2b",
    )(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + "2b")(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(
        filters3, (1, 1), kernel_initializer="he_normal", name=conv_name_base + "2c"
    )(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + "2c")(x)

    shortcut = layers.Conv2D(
        filters3,
        (1, 1),
        strides=strides,
        kernel_initializer="he_normal",
        name=conv_name_base + "1",
    )(input_tensor)
    shortcut = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + "1")(
        shortcut
    )

    x = layers.add([x, shortcut])
    x = layers.Activation("relu")(x)
    return x


def ResNet50(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    **kwargs
):
    """Instantiates the ResNet50 architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    global backend, layers, models, keras_utils
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    if not (weights in {"imagenet", None} or os.path.exists(weights)):
        raise ValueError(
            "The `weights` argument should be either "
            "`None` (random initialization), `imagenet` "
            "(pre-training on ImageNet), "
            "or the path to the weights file to be loaded."
        )

    if weights == "imagenet" and include_top and classes != 1000:
        raise ValueError(
            'If using `weights` as `"imagenet"` with `include_top`'
            " as true, `classes` should be 1000"
        )

    # Determine proper input shape
    input_shape = _obtain_input_shape(
        input_shape,
        default_size=224,
        min_size=32,
        data_format=backend.image_data_format(),
        require_flatten=include_top,
        weights=weights,
    )

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if backend.image_data_format() == "channels_last":
        bn_axis = 3
    else:
        bn_axis = 1

    x = layers.ZeroPadding2D(padding=(3, 3), name="conv1_pad")(img_input)
    x = layers.Conv2D(
        64,
        (7, 7),
        strides=(2, 2),
        padding="valid",
        kernel_initializer="he_normal",
        name="conv1",
    )(x)
    x = layers.BatchNormalization(axis=bn_axis, name="bn_conv1")(x)
    x = layers.Activation("relu")(x)
    x = layers.ZeroPadding2D(padding=(1, 1), name="pool1_pad")(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block="a", strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block="b")
    x = identity_block(x, 3, [64, 64, 256], stage=2, block="c")

    x = conv_block(x, 3, [128, 128, 512], stage=3, block="a")
    x = identity_block(x, 3, [128, 128, 512], stage=3, block="b")
    x = identity_block(x, 3, [128, 128, 512], stage=3, block="c")
    x = identity_block(x, 3, [128, 128, 512], stage=3, block="d")

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block="a")
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block="b")
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block="c")
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block="d")
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block="e")
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block="f")

    # modify stride to (1, 1) for conv_block5 to maintain stride 16
    x = conv_block(x, 3, [512, 512, 2048], strides=(1, 1), stage=5, block="a")
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block="b")
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block="c")

    if include_top:
        x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        x = layers.Dense(classes, activation="softmax", name="fc1000")(x)
    else:
        if pooling == "avg":
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == "max":
            x = layers.GlobalMaxPooling2D()(x)
        else:
            warnings.warn(
                "The output shape of `ResNet50(include_top=False)` "
                "has been changed since Keras 2.2.0."
            )

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = models.Model(inputs, x, name="resnet50")

    # Load weights.
    if weights == "imagenet":
        if include_top:
            weights_path = keras_utils.get_file(
                "resnet50_weights_tf_dim_ordering_tf_kernels.h5",
                WEIGHTS_PATH,
                cache_subdir="models",
                md5_hash="a7b3fe01876f51b976af0dea6bc144eb",
            )
        else:
            weights_path = keras_utils.get_file(
                "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5",
                WEIGHTS_PATH_NO_TOP,
                cache_subdir="models",
                md5_hash="a268eb855778b3df3c7506639542a6af",
            )
        model.load_weights(weights_path)
        if backend.backend() == "theano":
            keras_utils.convert_all_kernels_in_model(model)
    elif weights is not None:
        model.load_weights(weights)

    return model


class ResNetPreprocess(Layer):
    """Preprocessing layer for ResNet inputs.
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

    def __init__(self, **kwargs):
        super(ResNetPreprocess, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        return preprocess_input(inputs)

    def get_config(self):
        return super(ResNetPreprocess, self).get_config()


if __name__ == "__main__":

    from tensorflow.keras.applications.resnet50 import preprocess_input
    from tensorflow.keras.layers import Input, Lambda
    from tensorflow.keras import Model

    input_layer = Input((192, 192, 3))
    model = ResNet50(include_top=False, input_shape=(192, 192, 3))
    # for layer in model.layers:
    #    layer.trainable = False
    normalized = ResNetPreprocess()(input_layer)
    pretrained_output = model(normalized)
    model = Model(inputs=input_layer, outputs=pretrained_output)
