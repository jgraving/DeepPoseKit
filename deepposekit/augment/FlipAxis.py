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
from deepposekit.io.BaseGenerator import BaseGenerator
from imgaug.augmenters import meta
from imgaug import parameters as iap

__all__ = ["FlipAxis"]


class FlipAxis(meta.Augmenter):
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
        
    p: int, default 0.5
        The probability that an image is flipped

    axis: int, default 0
        Axis over which images are flipped
        axis=0 flips up-down (np.flipud)
        axis=1 flips left-right (np.fliplr)
        
    seed: None or int or np.random.RandomState, default None
        The random state for the augmenter.
        
    name: None or str, default None
        Name given to the Augmenter object. The name is used in print().
        If left as None, will print 'UnnamedX'
        
    deterministic: bool, default False
        If set to true, each batch will be augmented the same way.


    Attributes
    ----------
    p: int
        The probability that an image is flipped
    
    axis: int
        The axis to reflect the image.

    swap_index: array
        The keypoint indices to swap when the image is flipped
        

    """

    def __init__(self, swap_index, p=0.5, axis=0, seed=None, name=None, deterministic=False):
        super(FlipAxis, self).__init__(seed=seed, name=name, random_state="deprecated", deterministic=deterministic)
        self.p = iap.handle_probability_param(p, "p")
        self.axis = axis
        if isinstance(swap_index, BaseGenerator):
            if hasattr(swap_index, "swap_index"):
                self.swap_index = swap_index.swap_index
        elif isinstance(swap_index, np.ndarray):
            self.swap_index = swap_index
        
    
    def _augment_batch_(self, batch, random_state, parents, hooks):
        samples = self.p.draw_samples((batch.nb_rows,),
                                      random_state=random_state)
        for i, sample in enumerate(samples):
            if sample >= 0.5:
                
                if batch.images is not None:
                    if self.axis == 0:
                        batch.images[i] = np.flipud(batch.images[i])
                    if self.axis == 1:
                        batch.images[i] = np.fliplr(batch.images[i])


                if batch.keypoints is not None:
                    kpsoi = batch.keypoints[i]
                    if self.axis == 0:
                        height = kpsoi.shape[0]
                        for kp in kpsoi.keypoints:
                            kp.y = (height-1) - kp.y
                    if self.axis == 1:
                        width = kpsoi.shape[1]
                        for kp in kpsoi.keypoints:
                            kp.x = (width-1) - kp.x
                    swapped = kpsoi.keypoints.copy()
                    for r in range(len(kpsoi.keypoints)):
                        idx = self.swap_index[r]
                        if idx >= 0:
                            kpsoi.keypoints[r] = swapped[idx]

        return batch

    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return [self.p, self.axis, self.swap_index]