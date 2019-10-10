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
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import BatchNormalization

import deepposekit.utils.image as image_utils
from deepposekit.models.engine import BaseModel
from deepposekit.models.layers.util import ImageNormalization
from deepposekit.models.layers.deeplabcut import ImageNetPreprocess
from deepposekit.models.layers.densenet import (
    FrontEnd,
    ImageNetFrontEnd,
    DenseNet,
    OutputChannels,
    Concatenate,
)


class StackedDenseNet(BaseModel):
    def __init__(
        self,
        train_generator,
        n_stacks=1,
        n_transitions=-1,
        growth_rate=48,
        bottleneck_factor=1,
        compression_factor=0.5,
        pretrained=False,
        subpixel=True,
        **kwargs
    ):
        """
        Define a Stacked DenseNet model from Graving et al. [1]
        for pose estimation.
        This model combines elements from [2-5]
        See `References` for details on the model architecture.

        Parameters
        ----------
        train_generator : class deepposekit.io.TrainingGenerator
            A deepposekit.io.TrainingGenerator class for generating
            images and confidence maps.
        n_stacks : int, default = 1
            The number of encoder-decoder networks to stack
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
        growth_rate : int, default = 48
            The number of channels to output from each convolutional
            block.
        bottleneck_factor : int, default = 1
            The factor for determining the number of input channels
            to 3x3 convolutional layer in each convolutional block.
            Inputs are first passed through a 1x1 convolutional layer to
            reduce the number of channels to:
            growth_rate * bottleneck_factor
        compression_factor : int, default = 0.5
            The factor for determining the number of channels passed
            through a transition layer (downsampling or upsampling).
            Inputs are first passed through a 1x1 convolutional layer
            to reduce the number of channels to
            n_input_channels * compression_factor
        pretrained : bool, default = False
            Whether to use an encoder that is pretrained on ImageNet
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
        1.  Graving, J.M., Chae, D., Naik, H., Li, L., Koger, B., Costelloe, B.R.,
            Couzin, I.D. (2019) DeepPoseKit, a software toolkit for fast and robust
            animal pose estimation using deep learning. eLife, 8, e47994
        2.  Jégou, S., Drozdzal, M., Vazquez, D., Romero, A., & Bengio, Y. (2017).
            The one hundred layers tiramisu: Fully convolutional densenets for
            semantic segmentation. In Computer Vision and Pattern Recognition
            Workshops (CVPRW), 2017 IEEE Conference on (pp. 1175-1183). IEEE.
        3.  Newell, A., Yang, K., & Deng, J. (2016). Stacked hourglass networks
            for human pose estimation. In European Conference on Computer
            Vision (pp. 483-499). Springer, Cham.
        4.  Huang, G., Liu, Z., Weinberger, K. Q., & van der Maaten, L. (2017).
            Densely connected convolutional networks. In Proceedings of the IEEE
            conference on computer vision and pattern recognition
            (Vol. 1, No. 2, p. 3).
        5.  Klambauer, G., Unterthiner, T., Mayr, A., & Hochreiter, S. (2017).
            Self-normalizing neural networks. In Advances in Neural Information
            Processing Systems (pp. 972-981).
        """

        self.n_stacks = n_stacks
        self.growth_rate = growth_rate
        self.bottleneck_factor = bottleneck_factor
        self.compression_factor = compression_factor
        self.n_transitions = n_transitions
        self.pretrained = pretrained
        super(StackedDenseNet, self).__init__(train_generator, subpixel, **kwargs)

    def __init_model__(self):
        max_transitions = np.min(
            [
                image_utils.n_downsample(self.train_generator.height),
                image_utils.n_downsample(self.train_generator.width),
            ]
        )

        n_transitions = self.n_transitions
        if isinstance(n_transitions, (int, np.integer)):
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

        if self.train_generator.downsample_factor < 2:
            raise ValueError(
                "StackedDenseNet is only compatible with `downsample_factor` >= 2."
                "Adjust the TrainingGenerator or choose a different model."
            )
        if n_transitions <= self.train_generator.downsample_factor:
            raise ValueError(
                "`n_transitions` <= `downsample_factor`. Increase `n_transitions` or decrease `downsample_factor`."
                " If `n_transitions` is -1 (the default), check that your image resolutions can be repeatedly downsampled (are divisible by 2 repeatedly)."
            )
        if self.pretrained:
            if self.input_shape[-1] is 1:
                inputs = Concatenate()([self.inputs] * 3)
                input_shape = self.input_shape[:-1] + (3,)
            else:
                inputs = self.inputs
                input_shape = self.input_shape
            normalized = ImageNetPreprocess("densenet121")(inputs)
            front_outputs = ImageNetFrontEnd(
                input_shape=input_shape,
                n_downsample=self.train_generator.downsample_factor,
                compression_factor=self.compression_factor,
            )(normalized)
        else:
            normalized = ImageNormalization()(self.inputs)
            front_outputs = FrontEnd(
                growth_rate=self.growth_rate,
                n_downsample=self.train_generator.downsample_factor,
                compression_factor=self.compression_factor,
                bottleneck_factor=self.bottleneck_factor,
            )(normalized)
        n_downsample = self.n_transitions - self.train_generator.downsample_factor
        outputs = front_outputs
        model_outputs = OutputChannels(
            self.train_generator.n_output_channels, name="output_0"
        )(outputs)

        model_outputs_list = [model_outputs]
        outputs.append(BatchNormalization()(model_outputs))
        for idx in range(self.n_stacks):
            outputs = DenseNet(
                growth_rate=self.growth_rate,
                n_downsample=self.n_transitions
                - self.train_generator.downsample_factor,
                downsample_factor=self.train_generator.downsample_factor,
                compression_factor=self.compression_factor,
                bottleneck_factor=self.bottleneck_factor,
            )(outputs)
            outputs.append(Concatenate()(front_outputs))
            outputs.append(BatchNormalization()(model_outputs))
            model_outputs = OutputChannels(
                self.train_generator.n_output_channels, name="output_" + str(idx + 1)
            )(outputs)
            model_outputs_list.append(model_outputs)

        self.train_model = Model(
            self.inputs, model_outputs_list, name=self.__class__.__name__
        )

    def get_config(self):
        config = {
            "name": self.__class__.__name__,
            "n_stacks": self.n_stacks,
            "n_transitions": self.n_transitions,
            "growth_rate": self.growth_rate,
            "bottleneck_factor": self.bottleneck_factor,
            "compression_factor": self.compression_factor,
            "pretrained": self.pretrained,
            "subpixel": self.subpixel,
        }
        base_config = super(StackedDenseNet, self).get_config()
        return dict(list(config.items()) + list(base_config.items()))
