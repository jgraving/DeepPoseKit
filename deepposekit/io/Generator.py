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

from tensorflow.keras.utils import Sequence
import numpy as np

__all__ = ["BaseGenerator"]


class BaseGenerator(Sequence):
    def __init__(self, **kwargs):
        if not hasattr(self, 'tree'):
            self.tree = -np.ones(self.keypoints_shape[0])
        if not hasattr(self, 'swap_index')::
            self.swap_index = -np.ones(self.keypoints_shape[0]) 
        return

    def __len__(self):
        raise NotImplementedError()

    def compute_image_shape(self):
        raise NotImplementedError()

    def compute_keypoints_shape(self):
        raise NotImplementedError()

    def get_images(self, idx):
        raise NotImplementedError()

    def get_keypoints(self, idx):
        raise NotImplementedError()

    def set_keypoints(self, idx, keypoints):
        raise NotImplementedError()

    def __call__(self):
        return NotImplementedError()

    @property
    def image_shape(self):
        return self.compute_image_shape()

    @property
    def keypoints_shape(self):
        return self.compute_keypoints_shape() 

    @property
    def shape(self):
        return (self.image_shape, self.keypoints_shape)

    def _check_index(self, key):
        if isinstance(key, slice):
            start = key.start
            stop = key.stop
            if start is None:
                start = 0
            if stop is None:
                stop = len(self)
            if stop <= len(self):
                indexes = range(start, stop)
            else:
                raise IndexError()
        elif isinstance(key, (int, np.integer)):
            if key < len(self):
                indexes = [key]
            else:
                raise IndexError()
        elif isinstance(key, np.ndarray):
            if np.max(key) < len(self):
                indexes = key.tolist()
            else:
                raise IndexError
        elif isinstance(key, list):
            if max(key) < len(self):
                indexes = key
            else:
                raise IndexError()
        else:
            raise IndexError()
        return indexes

    def get_data(self, indexes):
        return (self.get_images(indexes), self.get_keypoints(indexes))

    def set_data(self, indexes, keypoints):
        self.set_keypoints(indexes, keypoints)

    def __getitem__(self, key):
        indexes = self._check_index(key)
        return self.get_data(indexes)

    def __setitem__(self, key, keypoints):
        indexes = self._check_index(key)
        if len(value) != len(indexes):
            raise IndexError("data shape and index do not match")
        self.set_data(indexes, keypoints)
