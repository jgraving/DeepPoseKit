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
import imgaug.augmenters as iaa
import six.moves as sm
import h5py

from deepposekit.io.BaseGenerator import BaseGenerator

__all__ = ["FlipAxis"]


class FlipAxis(iaa.Flipud):
    """ Flips the input image and keypoints across an axis.

    A generalized class for flipping images and keypoints
    either horizontally and vertically during augmentation.
    This class requires a swap_index parameter, which indicates the
    relationships between keypoint labels when flipping the image
    (e.g. left leg is swapped with right leg, etc.)

    Parameters
    ----------
    swap_index: deepposekit.io.BaseGenerator or array
        The keypoint indices to swap when the image is flipped.
        This can be a deepposekit.io.BaseGenerator for annotations
        or an array of integers specifying which keypoint indices
        to swap.

    axis: int, default 0
        Axis over which images are flipped
        axis=0 flips up-down (np.flipud)
        axis=1 flips left-right (np.fliplr)

    name: None or str, default None
        Name given to the Augmenter object. The name is used in print().
        If left as None, will print 'UnnamedX'

    deterministic: bool, default False
        If set to true, each batch will be augmented the same way.

    random_state: None or int or np.random.RandomState, default None
        The random state for the augmenter.

    Attributes
    ----------
    axis: int
        The axis to reflect the image.

    swap_index: array
        The keypoint indices to swap when the image is flipped

    """

    def __init__(
        self,
        swap_index,
        p=0.5,
        axis=0,
        name=None,
        deterministic=False,
        random_state=None,
    ):

        super(FlipAxis, self).__init__(
            p=p, name=name, deterministic=deterministic, random_state=random_state
        )

        self.axis = axis
        if isinstance(swap_index, BaseGenerator):
            if hasattr(swap_index, "swap_index"):
                self.swap_index = swap_index.swap_index
        elif isinstance(swap_index, np.ndarray):
            self.swap_index = swap_index

    def _augment_images(self, images, random_state, parents, hooks):
        """ Augments the images

        Handles the augmentation over a specified axis

        Returns
        -------
        images: array
            Array of augmented images.

        """
        nb_images = len(images)
        samples = self.p.draw_samples((nb_images,), random_state=random_state)
        for i in sm.xrange(nb_images):
            if samples[i] == 1:
                if self.axis == 1:
                    images[i] = np.fliplr(images[i])
                elif self.axis == 0:
                    images[i] = np.flipud(images[i])
        self.samples = samples
        return images

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        """ Augments the keypoints

        Handles the augmentation over a specified axis
        and swaps the keypoint labels using swap_index.
        For example, the left leg will be swapped with the right leg
        This is accomplished by reordering the keypoints.

        Returns
        -------
        keypoints_on_images: array
            Array of new coordinates of the keypoints.

        """
        nb_images = len(keypoints_on_images)
        samples = self.p.draw_samples((nb_images,), random_state=random_state)
        for i, keypoints_on_image in enumerate(keypoints_on_images):
            if samples[i] == 1:
                for keypoint in keypoints_on_image.keypoints:
                    if self.axis == 1:
                        width = keypoints_on_image.shape[1]
                        keypoint.x = (width - 1) - keypoint.x
                    elif self.axis == 0:
                        height = keypoints_on_image.shape[0]
                        keypoint.y = (height - 1) - keypoint.y
                swapped = keypoints_on_image.keypoints.copy()
                for r in range(len(keypoints_on_image.keypoints)):
                    idx = self.swap_index[r]
                    if idx >= 0:
                        keypoints_on_image.keypoints[r] = swapped[idx]
        return keypoints_on_images
