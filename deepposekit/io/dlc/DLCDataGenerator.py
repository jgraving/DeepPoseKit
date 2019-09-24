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
import h5py
import numpy as np
import os
import copy
import pandas as pd
import cv2

__all__ = ["DLCDataGenerator"]


class DLCDataGenerator(Sequence):
    """
    Creates a data generator for accessing a DeepLabCut annotation set.

    Parameters
    ----------
    datapath : str
        The path to the annotations file. Must be .h5
        e.g. '/path/to/file.h5'
    imagepath : str
        Path to the image dataset used in the annotations file.
        e.g. '/path/to/images/'
    """

    def __init__(self, datapath, imagepath):
        self.annotations = pd.read_hdf(datapath)
        self.imagepath = imagepath
        scorer = []
        bodyparts = []
        for column in self.annotations.columns:
            scorer.append(column[0])
            bodyparts.append(column[1])
        self.bodyparts = np.unique(bodyparts)
        self.scorer = np.unique(scorer)[0]
        self.xy = ["x", "y"]

        self.n_keypoints = len(bodyparts)
        self.n_samples = self.annotations.shape[0]
        self.index = np.arange(self.n_samples)

    def get_data(self, indexes):
        indexes = self.index[indexes]

        X = []
        Y = []
        for idx in indexes:
            row = self.annotations.iloc[idx]
            coords = []
            for part in self.bodyparts:
                x = row[(self.scorer, part, "x")]
                y = row[(self.scorer, part, "y")]
                if np.isnan(x) or np.isnan(y):
                    x = -9999999999
                    y = -9999999999
                coords.append([x, y])
            coords = np.array(coords)
            image = row.name
            image = cv2.imread(self.imagepath + image)
            height, width, channels = image.shape
            # height_pad = 800 - height if 800 - height > 0 else 0
            # width_pad = 832 - width if 800 - width > 0 else 0
            # image = np.pad(image, ((0,height_pad), (0,width_pad), (0,0)))
            X.append(image)
            Y.append(coords)

        X = np.stack(X)
        Y = np.stack(Y)

        return X, Y

    def set_data(self, indexes, y):
        return NotImplementedError
        """
        if y.shape[-1] is 3:
            y = y[..., :2]
        elif y.shape[-1] is not 2:
            raise ValueError('data shape does not match')
        if self.mode is 'annotated':
            indexes = self.annotated_index[indexes]
        elif self.mode is 'unannotated':
            indexes = self.unannotated_index[indexes]
        else:
            indexes = self.index[indexes]

        with h5py.File(self.datapath, mode='r+') as h5file:
            for idx, keypoints in zip(indexes, y):
                h5file['annotations'][idx] = keypoints
        """

    def __len__(self):
        return self.n_samples

    def _check_index(self, key):
        if isinstance(key, slice):
            start = key.start
            stop = key.stop
            if start is None:
                start = 0
            if stop is None:
                stop = len(self)
            if stop <= len(self):
                idx = range(start, stop)
            else:
                raise IndexError
        elif isinstance(key, (int, np.integer)):
            if key < len(self):
                idx = [key]
            else:
                raise IndexError
        elif isinstance(key, np.ndarray):
            if np.max(key) < len(self):
                idx = key.tolist()
            else:
                raise IndexError
        elif isinstance(key, list):
            if max(key) < len(self):
                idx = key
            else:
                raise IndexError
        else:
            raise IndexError
        return idx

    def __getitem__(self, key):
        idx = self._check_index(key)
        return self.get_data(idx)

    def __setitem__(self, key, value):

        idx = self._check_index(key)
        if isinstance(value, (np.ndarray, list)):
            if len(value) != len(idx):
                raise IndexError("data shape and " "index do not match")
            self.set_data(idx, value)


if __name__ == "__main__":
    data_generator = DLCDataGenerator(
        datapath="./deeplabcut/examples/openfield-Pranav-2018-10-30/labeled-data/m4s1/CollectedData_Pranav.h5",
        imagepath="./deeplabcut/examples/openfield-Pranav-2018-10-30/",
    )
