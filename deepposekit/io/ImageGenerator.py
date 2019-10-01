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

from deepposekit.io.BaseGenerator import BaseGenerator

__all__ = ["ImageGenerator"]


class ImageGenerator(BaseGenerator):
    """
    Creates a wrapper for generating images from a data generator.

    Parameters
    ----------
    generator: deepposekit.io.BaseGenerator
        An instance of BaseGenerator (deepposekit.io.BaseGenerator) object.
        The output of the generator must be `(images, keypoints)`, where images
        are a numpy array of shape (n_images, height, width, channels), and 
        keypoints are a numpy array of shape (n_images, n_keypoints, 2), where
        2 is the row, column coordinates of the keypoints in each image.

    """

    def __init__(self, generator, **kwargs):
        self.generator = generator
        self.get_data = self.generator.get_images
        self.__len__ = self.generator.__len__
        self.compute_keypoints_shape = self.generator.compute_keypoints_shape
        self.compute_image_shape = self.generator.compute_image_shape

        super(ImageGenerator, self).__init__(**kwargs)

    def __len__(self):
        return len(self.generator)

    def get_data(self, indexes):
        return self.generator.get_images(indexes)

    def set_keypoints(self, indexes, keypoints):
        return self.generator.set_keypoints(indexes, keypoints)

    @property
    def shape(self):
        return (len(self),) + self.generator.image_shape

    def get_config(self):
        config = {}
        base_config = super(ImageGenerator, self).get_config()
        return dict(list(config.items()) + list(base_config.items()))
