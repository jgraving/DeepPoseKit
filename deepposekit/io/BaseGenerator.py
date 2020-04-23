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

from tensorflow.keras.utils import Sequence
import numpy as np

__all__ = ["BaseGenerator"]


class BaseGenerator(Sequence):
    """
    BaseGenerator class for abstracting data loading and saving.
    Attributes that should be defined before use:
    __init__
    __len__
    compute_image_shape
    compute_keypoints_shape
    get_images
    get_keypoints
    set_keypoints (only needed for saving data)
    
    See docstrings for further details.
    """

    def __init__(self, **kwargs):
        """
        Initializes the BaseGenerator class.
        If graph and swap_index are not defined,
        they are set to a vector of -1 corresponding
        to keypoints shape
        """
        if not hasattr(self, "graph"):
            self.graph = -np.ones(self.keypoints_shape[0])
        if not hasattr(self, "swap_index"):
            self.swap_index = -np.ones(self.keypoints_shape[0])
        return

    def __len__(self):
        """
        Returns the number of samples in the generator as an integer (int64)
        """
        raise NotImplementedError()

    def compute_image_shape(self):
        """
        Returns a tuple of integers describing
        the image shape in the form:
        (height, width, n_channels)
        """
        raise NotImplementedError()

    def compute_keypoints_shape(self):
        """
        Returns a tuple of integers describing the
        keypoints shape in the form:
        (n_keypoints, 2), where 2 is the x,y coordinates
        """
        raise NotImplementedError()

    def get_images(self, indexes):
        """
        Takes a list or array of indexes corresponding
        to image-keypoint pairs in the dataset.
        Returns a numpy array of images with the shape:
        (n_samples, height, width, n_channels)
        """
        raise NotImplementedError()

    def get_keypoints(self, indexes):
        """
        Takes a list or array of indexes corresponding to
        image-keypoint pairs in the dataset.
        Returns a numpy array of keypoints with the shape:
        (n_samples, n_keypoints, 2), where 2 is the x,y coordinates
        """
        raise NotImplementedError()

    def set_keypoints(self, indexes, keypoints):
        """
        Takes a list or array of indexes and corresponding
        to keypoints.
        Sets the values of the keypoints corresponding to the indexes
        in the dataset.
        """
        raise NotImplementedError()

    def __call__(self):
        return NotImplementedError()

    @property
    def image_shape(self):
        return self.compute_image_shape()

    def replace_nan(self, keypoints):
        keypoints[np.isnan(keypoints)] = -99999
        return keypoints

    @property
    def keypoints_shape(self):
        return self.compute_keypoints_shape()

    @property
    def shape(self):
        """
        Returns a tuple of tuples describing the data shapes
        in the form:
        ((height, width, n_channels), (n_keypoints, 2))
        """
        return (self.image_shape, self.keypoints_shape)

    def get_data(self, indexes):
        keypoints = self.get_keypoints(indexes)
        keypoints = self.replace_nan(keypoints)
        return (self.get_images(indexes), keypoints)

    def set_data(self, indexes, keypoints):
        self.set_keypoints(indexes, keypoints)

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

    def __getitem__(self, key):
        indexes = self._check_index(key)
        return self.get_data(indexes)

    def __setitem__(self, key, keypoints):
        indexes = self._check_index(key)
        if len(keypoints) != len(indexes):
            raise IndexError("data shape and index do not match")
        self.set_data(indexes, keypoints[..., :2])

    def get_config(self):
        config = {
            "generator": self.__class__.__name__,
            "n_samples": len(self),
            "image_shape": self.image_shape,
            "keypoints_shape": self.keypoints_shape,
        }
        return config
