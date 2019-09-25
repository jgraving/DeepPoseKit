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
import pandas as pd
import cv2

from deepposekit.io.Generator import BaseGenerator

__all__ = ["DLCDataGenerator"]


class DLCDataGenerator(BaseGenerator):
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

        self.n_keypoints = len(self.bodyparts)
        self.n_samples = self.annotations.shape[0]
        self.index = np.arange(self.n_samples)

    def compute_image_shape(self):
        return self.get_images([0]).shape[1:]

    def compute_keypoints_shape(self):
        return (self.n_keypoints, 2)

    def get_images(self, indexes):
        indexes = self.index[indexes]
        images = []
        for idx in indexes:
            row = self.annotations.iloc[idx]
            image = row.name
            image = cv2.imread(self.imagepath + image)
            height, width, channels = image.shape
            images.append(image)
        return np.stack(images)

    def get_keypoints(self, indexes):
        indexes = self.index[indexes]
        keypoints = []
        for idx in indexes:
            row = self.annotations.iloc[idx]
            coords = []
            for part in self.bodyparts:
                x = row[(self.scorer, part, "x")]
                y = row[(self.scorer, part, "y")]
                if np.isnan(x) or np.isnan(y):
                    x = -1e10
                    y = -1e10
                coords.append([x, y])
            coords = np.array(coords)
            keypoints.append(coords)
        return np.stack(keypoints)

    def __len__(self):
        return self.n_samples


if __name__ == "__main__":
    data_generator = DLCDataGenerator(
        datapath="./deeplabcut/examples/openfield-Pranav-2018-10-30/labeled-data/m4s1/CollectedData_Pranav.h5",
        imagepath="./deeplabcut/examples/openfield-Pranav-2018-10-30/",
    )
