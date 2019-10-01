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
import h5py
import numpy as np
import os
import copy

from deepposekit.io.BaseGenerator import BaseGenerator

__all__ = ["DataGenerator"]


class DataGenerator(BaseGenerator):
    """
    Creates a data generator for accessing an annotation set.

    Parameters
    ----------
    datapath : str
        The path to the annotations file. Must be .h5
        e.g. '/path/to/file.h5'
    dataset : str
        The key for the image dataset in the annotations file.
        e.g. 'images'
    mode : str
        The mode for loading and saving data.
        Must be 'unannotated', 'annotated', or "full"
    """

    def __init__(self, datapath, dataset="images", mode="annotated", **kwargs):

        # Check annotations file
        if isinstance(datapath, str):
            if datapath.endswith(".h5"):
                if os.path.exists(datapath):
                    self.datapath = datapath
                else:
                    raise ValueError("datapath file or " "directory does not exist")

            else:
                raise ValueError("datapath must be .h5 file")
        else:
            raise TypeError("datapath must be type `str`")

        if isinstance(dataset, str):
            self.dataset = dataset
        else:
            raise TypeError("dataset must be type `str`")

        with h5py.File(self.datapath, mode="r") as h5file:

            # Check for annotations
            if "annotations" not in list(h5file.keys()):
                raise KeyError("annotations not found in annotations file")
            if "annotated" not in list(h5file.keys()):
                raise KeyError("annotations not found in annotations file")
            if "skeleton" not in list(h5file.keys()):
                raise KeyError("skeleton not found in annotations file")
            if self.dataset not in list(h5file.keys()):
                raise KeyError("image dataset not found in annotations file")

            # Get annotations attributes
            if mode not in ["full", "annotated", "unannotated"]:
                raise ValueError("mode must be 'full', 'annotated', or 'unannotated'")
            else:
                self.mode = mode
            self.annotated = np.all(h5file["annotated"].value, axis=1)
            self.annotated_index = np.where(self.annotated)[0]
            self.n_annotated = self.annotated_index.shape[0]
            if self.n_annotated == 0 and self.mode not in ["full", "unannotated"]:
                raise ValueError("The number of annotated images is zero")
            self.n_keypoints = h5file["annotations"].shape[1]
            self.n_samples = h5file[self.dataset].shape[0]
            self.index = np.arange(self.n_samples)
            self.unannotated_index = np.where(~self.annotated)[0]
            self.n_unannotated = self.unannotated_index.shape[0]

            # Initialize skeleton attributes
            self.graph = h5file["skeleton"][:, 0]
            self.swap_index = h5file["skeleton"][:, 1]

        super(DataGenerator, self).__init__(**kwargs)

    def compute_keypoints_shape(self):
        with h5py.File(self.datapath, mode="r") as h5file:
            return h5file["annotations"].shape[1:]

    def compute_image_shape(self):
        with h5py.File(self.datapath, mode="r") as h5file:
            return h5file[self.dataset].shape[1:]

    def get_indexes(self, indexes):
        if self.mode is "annotated":
            indexes = self.annotated_index[indexes]
        elif self.mode is "unannotated":
            indexes = self.unannotated_index[indexes]
        else:
            indexes = self.index[indexes]
        return indexes

    def get_images(self, indexes):
        indexes = self.get_indexes(indexes)
        images = []
        with h5py.File(self.datapath, mode="r") as h5file:
            for idx in indexes:
                images.append(h5file[self.dataset][idx])
        return np.stack(images)

    def get_keypoints(self, indexes):
        indexes = self.get_indexes(indexes)
        keypoints = []
        with h5py.File(self.datapath, mode="r") as h5file:
            for idx in indexes:
                keypoints.append(h5file["annotations"][idx])
        return np.stack(keypoints)

    def set_keypoints(self, indexes, keypoints):
        if keypoints.shape[-1] is 3:
            keypoints = keypoints[..., :2]
        elif keypoints.shape[-1] is not 2:
            raise ValueError("data shape does not match annotations")
        indexes = self.get_indexes(indexes)

        with h5py.File(self.datapath, mode="r+") as h5file:
            for idx, keypoints_idx in zip(indexes, keypoints):
                h5file["annotations"][idx] = keypoints_idx

    def __call__(self, mode="annotated"):
        if mode not in ["full", "annotated", "unannotated"]:
            raise ValueError("mode must be full, annotated, or unannotated")
        elif mode is "annotated" and self.n_annotated == 0:
            raise ValueError(
                "cannot return annotated samples, "
                "number of annotated samples is zero"
            )
        elif mode is "unannotated" and self.n_unannotated == 0:
            raise ValueError(
                "cannot return unannotated samples, "
                "number of unannotated samples is zero"
            )
        else:
            self.mode = mode
        return copy.deepcopy(self)

    def __len__(self):
        if self.mode is "annotated":
            return self.n_annotated
        elif self.mode is "unannotated":
            return self.n_unannotated
        else:
            return self.n_samples

    def get_config(self):
        config = {"datapath": self.datapath, "dataset": self.dataset}
        base_config = super(DataGenerator, self).get_config()
        return dict(list(config.items()) + list(base_config.items()))
