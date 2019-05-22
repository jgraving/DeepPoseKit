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

import numpy as np
try:
    from imgaug import augmenters as iaa
except:
    from imgaug.imgaug import augmenters as iaa
from ..utils.image import check_grayscale
from ..utils.keypoints import imgaug_to_numpy, numpy_to_imgaug

__all__ = ['Augmenter']


class Augmenter(iaa.Sequential):
    '''
    Wrapper for imgaug Augmenter that augments
    images and keypoints when called.

    Parameters
    ----------
    augmenter: Augmenter class or list of Augmenters
        The augmenter to be applied to all the input images and keypoints.
        See imgaug.augmenters.Augmenter for more details
    '''
    def __init__(self, augmenter):
        if isinstance(augmenter, iaa.Augmenter):
            super(Augmenter, self).__init__(augmenter)
        elif isinstance(augmenter, list):
            if isinstance(augmenter[0], iaa.Augmenter):
                super(Augmenter, self).__init__(augmenter)
            else:
                raise TypeError('''`augmenter` must be class Augmenter
                            (imgaug.augmenters.Augmenter)
                            or list of Augmenters''')
        else:
            raise TypeError('''`augmenter` must be class Augmenter
                            (imgaug.augmenters.Augmenter)
                            or list of Augmenters''')

    def _augment(self, images, keypoints):
        """
        Returns augmented images and keypoints.

        Parameters
        ----------
        images: array, shape = (n_samples, height, width, channels)
            An array of images.

        keypoints: array, shape = (n_samples, n_keypoints, 2)
            An array of 2-D keypoints.

        Returns
        -------
        images_aug: array, shape = (n_samples, height, width, channels)
            Augmented images.

        keypoints_aug: array, shape = (n_samples, n_keypoints, 2)
            Augmented keypoints.
        """

        images_aug_list = []
        keypoints_aug_list = []

        for image_aug, keypoints_aug in zip(images, keypoints):
            image_aug, grayscale = check_grayscale(image_aug, return_color=True)

            keypoints_aug = numpy_to_imgaug(image_aug, keypoints_aug)
            aug_det = self.to_deterministic()

            image_aug = aug_det.augment_images([image_aug])[0]
            if grayscale:
                image_aug = image_aug[..., 0][..., np.newaxis]
            images_aug_list.append(image_aug)

            keypoints_aug = aug_det.augment_keypoints([keypoints_aug])[0]
            keypoints_aug = imgaug_to_numpy(keypoints_aug)
            keypoints_aug_list.append(keypoints_aug)

        images_aug = np.stack(images_aug_list)
        keypoints_aug = np.stack(keypoints_aug_list)

        return images_aug, keypoints_aug

    def __call__(self, images, keypoints):
        """
        Augment images and keypoints

        Parameters
        ----------
        images: array, shape = (n_samples, height, width, channels)
            An array of images.

        keypoints: array, shape = (n_samples, n_keypoints, 2)
            An array of 2-D keypoints.

        Returns
        -------
        images_aug: array, shape = (n_samples, height, width, channels)
            Augmented images.

        keypoints_aug: array, shape = (n_samples, n_keypoints, 2)
            Augmented keypoints.
        """
        """ TODO: flip images, keypoints, and relevant labels
        along user-specified axis with some probability """
        images, keypoints = self._augment(images,
                                          keypoints)

        return images, keypoints
