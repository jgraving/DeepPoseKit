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
from tensorflow.keras.layers import Conv2DTranspose, Concatenate
from deepposekit.models.layers.deeplabcut import ImageNetPreprocess, MODELS
from deepposekit.models.layers.convolutional import SubPixelUpscaling

from deepposekit.models.engine import BaseModel
from functools import partial


__docstring__ = """
    Define a DeepLabCut model from Mathis et al., 2018 [1–4]
    including MobileNetV2 backend from [4].
    See `References` for details on the model architecture.

    Parameters
    ----------
    train_generator : class deepposekit.io.TrainingGenerator
        A deepposekit.io.TrainingGenerator class for generating
        images and confidence maps.
    subpixel: bool, default = True
        Whether to use subpixel maxima for calculating
        keypoint coordinates in the prediction model.
    weights: "imagnet" or None, default is "imagenet"
        Which weights to use for initialization. "imagenet" uses
        weights pretrained on imagenet. None uses randomly initialized
        weights.
    backbone: string, default is "resnet50"
        pretrained backbone network to use. Must be one of {}. See [3].
    alpha: float, default is 1.0
        Which MobileNetV2 to use. Must be one of:
        {}
        Not used if backbone is not "mobilenetv2".

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
    1.  Insafutdinov, E., Pishchulin, L., Andres, B., Andriluka, M., & Schiele, B.
        (2016). Deepercut: A deeper, stronger, and faster multi-person pose estimation
        model. In European Conference on Computer Vision (pp. 34-50). Springer, Cham.
    2.  Mathis, A., Mamidanna, P., Cury, K. M., Abe, T., Murthy, V. N.,
        Mathis, M. W., & Bethge, M. (2018). DeepLabCut: markerless pose
        estimation of user-defined body parts with deep learning (p. 1).
        Nature Publishing Group.
    3.  Nath, T., Mathis, A., Chen, A. C., Patel, A., Bethge, M.,
        & Mathis, M. W. (2019). Using DeepLabCut for 3D markerless
        pose estimation across species and behaviors. Nature protocols,
        14(7), 2152-2176.
    4.  Mathis, A., Yuksekgonol, M., Rogers, B., Bethge, M., Mathis, M. (2019).
        Pretraining boosts out-of-domain-robustness for pose estimation.
        arXiv cs.CV https://arxiv.org/abs/1909.11229


    """.format(
    list(MODELS.keys()), [0.35, 0.50, 0.75, 1.0, 1.3, 1.4]
)


class DeepLabCut(BaseModel):
    __doc__ = __docstring__

    def __init__(
        self,
        train_generator,
        subpixel=True,
        weights="imagenet",
        backbone="resnet50",
        alpha=1.0,
        **kwargs
    ):

        self.subpixel = subpixel
        self.weights = weights
        self.backbone = backbone
        self.alpha = alpha
        super(DeepLabCut, self).__init__(train_generator, subpixel, **kwargs)

    def __init_model__(self):

        if self.input_shape[-1] is 1:
            inputs = Concatenate()([self.inputs] * 3)
        else:
            inputs = self.inputs
        if self.backbone in list(MODELS.keys()):
            normalized = ImageNetPreprocess(self.backbone)(inputs)
        else:
            raise ValueError(
                "backbone model {} is not supported. Must be one of {}".format(
                    self.backbone, list(MODELS.keys())
                )
            )
        backbone = MODELS[self.backbone]
        if self.backbone in list(MODELS.keys()):
            input_shape = None  # self.input_shape[:-1] + (3,)
        if self.backbone.startswith("mobile"):
            input_shape = None
            backbone = partial(backbone, alpha=self.alpha)
        pretrained_model = backbone(
            include_top=False, weights=self.weights, input_shape=input_shape
        )
        pretrained_features = pretrained_model(normalized)
        if self.train_generator.downsample_factor is 4:
            x = pretrained_features
            x_out = Conv2D(self.train_generator.n_output_channels, (1, 1))(x)
        elif self.train_generator.downsample_factor is 3:
            x = pretrained_features
            x_out = Conv2DTranspose(
                self.train_generator.n_output_channels,
                (3, 3),
                strides=(2, 2),
                padding="same",
            )(x)
        elif self.train_generator.downsample_factor is 2:
            x = pretrained_features
            x = SubPixelUpscaling()(x)
            x_out = Conv2DTranspose(
                self.train_generator.n_output_channels,
                (3, 3),
                strides=(2, 2),
                padding="same",
            )(x)
        else:
            raise ValueError(
                "`downsample_factor={}` is not supported for DeepLabCut. Adjust your TrainingGenerator".format(
                    self.train_generator.downsample_factor
                )
            )

        self.train_model = Model(self.inputs, x_out, name=self.__class__.__name__)

    def get_config(self):
        config = {
            "name": self.__class__.__name__,
            "subpixel": self.subpixel,
            "weights": self.weights,
            "backbone": self.backbone,
            "alpha": self.alpha if self.backbone is "mobilenetv2" else None,
        }
        base_config = super(DeepLabCut, self).get_config()
        return dict(list(config.items()) + list(base_config.items()))
