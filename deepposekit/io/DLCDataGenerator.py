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
import pandas as pd
import os
import cv2
import yaml
import glob

from deepposekit.io.BaseGenerator import BaseGenerator

__all__ = ["DLCDataGenerator"]


class DLCDataGenerator(BaseGenerator):
    """
    Creates a data generator for accessing a DeepLabCut annotation set.

    Parameters
    ----------
    project_path : str
        Path to the project with config.yaml and images.
        e.g. '/path/to/project/'
    """

    def __init__(self, project_path, **kwargs):
        self.project_path = project_path
        self.annotations_path = glob.glob(self.project_path + "/**/**/*.h5")
        annotations = [pd.read_hdf(datapath) for datapath in self.annotations_path]
        self.annotations = pd.concat(annotations)

        with open(project_path + "/config.yaml", "r") as config_file:
            self.dlcconfig = yaml.load(config_file, Loader=yaml.SafeLoader)
        self.n_keypoints = len(self.dlcconfig["bodyparts"])

        self.bodyparts = self.dlcconfig["bodyparts"]
        self.scorer = self.dlcconfig["scorer"]

        self.n_samples = self.annotations.shape[0]
        self.index = np.arange(self.n_samples)

        super(DLCDataGenerator, self).__init__(**kwargs)

    def compute_image_shape(self):
        return self.get_images([0]).shape[1:]

    def compute_keypoints_shape(self):
        return (self.n_keypoints, 2)

    def get_images(self, indexes):
        indexes = self.index[indexes]
        images = []
        for idx in indexes:
            row = self.annotations.iloc[idx]
            image_name = row.name
            filepath = self.project_path + image_name
            if os.path.exists(filepath):
                images.append(cv2.imread(filepath))
            else:
                raise IndexError("image `{}` does not exist".format(image_name))
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
                coords.append([x, y])
            coords = np.array(coords)
            keypoints.append(coords)
        return np.stack(keypoints)

    def __len__(self):
        return self.n_samples

    def get_config(self):
        config = {"project_path": self.project_path}
        base_config = super(DLCDataGenerator, self).get_config()
        return dict(list(config.items()) + list(base_config.items()))


if __name__ == "__main__":
    data_generator = DLCDataGenerator(
        project_path="./deeplabcut/examples/openfield-Pranav-2018-10-30/"
    )
