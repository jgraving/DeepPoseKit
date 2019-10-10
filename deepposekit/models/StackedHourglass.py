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

from tensorflow.keras import Model
import numpy as np

from deepposekit.models.engine import BaseModel
from deepposekit.models.layers.util import ImageNormalization
from deepposekit.utils import image as image_utils
from deepposekit.models.layers.hourglass import FrontModule, Output, Hourglass


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
        Define a Stacked Hourglass model for pose estimation from [1].
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
        1.  Newell, A., Yang, K., & Deng, J. (2016). Stacked hourglass networks
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

        normalized = ImageNormalization()(self.inputs)
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

        self.train_model = Model(self.inputs, outputs, name=self.__class__.__name__)

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
