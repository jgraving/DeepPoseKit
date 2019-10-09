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
import h5py
import os
import pandas as pd

from deepposekit.io.DataGenerator import DataGenerator

__all__ = ["initialize_dataset", "initialize_skeleton", "merge_new_images"]


def initialize_skeleton(skeleton):
    """ Initialize the skeleton from input data.

    Takes in either a .csv or .xlsx file and makes a DataFrame.

    Parameters
    ----------
    skeleton: pandas.DataFrame
        Filepath of the .csv or .xlsx file that has indexed information
        on name of the keypoint (part, e.g. head), parent (the direct
        connecting part, e.g. neck connects to head, parent is head),
        and swap (swapping positions with a part when reflected over X).
    """

    if isinstance(skeleton, str):
        if skeleton.endswith(".csv"):
            skeleton = pd.read_csv(skeleton)
        elif skeleton.endswith(".xlsx"):
            skeleton = pd.read_excel(skeleton)
        else:
            raise ValueError("skeleton must be .csv or .xlsx file")
    elif isinstance(skeleton, pd.DataFrame):
        skeleton = skeleton
    else:
        raise TypeError("skeleton must be type `str` or pandas.DataFrame")

    if "name" not in skeleton.columns:
        raise KeyError("skeleton file must contain a `name` column")
    elif "parent" not in skeleton.columns:
        raise KeyError("skeleton file must contain a `parent` column")

    if "x" not in skeleton.columns:
        skeleton["x"] = -1
    if "y" not in skeleton.columns:
        skeleton["y"] = -1

    if "tree" not in skeleton.columns:
        skeleton["tree"] = -1
        for idx, name in enumerate(skeleton["parent"].values):
            branch = np.where(skeleton["name"] == name)[0]
            if branch.shape[0] > 0:
                branch = branch[0]
                skeleton.loc[idx, "tree"] = branch
    if "swap_index" not in skeleton.columns:
        skeleton["swap_index"] = -1
        for idx, name in enumerate(skeleton["name"].values):
            for jdx, swap_name in enumerate(skeleton["swap"].values):
                if swap_name == name:
                    skeleton.loc[idx, "swap_index"] = jdx
    return skeleton


def initialize_dataset(
    datapath, images, skeleton, keypoints=None, dataset="images", overwrite=False
):
    """
    Intialize an image dataset for annotation as an h5 file

    Parameters
    ----------
    datapath : str
        The path to the annotations file. Must be .h5
        e.g. '/path/to/file.h5'
    images : ndarray, shape (n_images, height, width, channels)
        A numpy array containing image data. 
        `images.dtype` should be np.uint8
    skeleton: str or pandas.DataFrame
        Filepath of the .csv or .xlsx file that has indexed information
        on name of the keypoint (part, e.g. head), parent (the direct
        connecting part, e.g. neck connects to head, parent is head),
        and swap (swapping positions with a part when reflected).
        See example files for more information.
    keypoints : None or ndarray, shape (n_images, n_keypoints, 2)
        Optionally pass keypoints for initializing annotations for the
        new images.
    dataset : str, default = "images"
        The name of the dataset within the h5 file to save the images.
    overwrite: bool, default = False
        Whether to overwrite an existing .h5 file with the same name.
    """
    if os.path.exists(datapath) and overwrite is False:
        raise OSError(
            "Annotation set {} already exists. Delete the file or set `overwrite=True`.".format(
                datapath
            )
        )
    if not isinstance(images, np.ndarray):
        raise TypeError(
            "images must be ndarray with shape (n_images, height, width, channels)"
        )
    elif images.ndim != 4:
        raise TypeError(
            "images must be ndarray with shape (n_images, height, width, channels)"
        )
    elif images.dtype != np.uint8:
        raise TypeError("`images` must be ndarray with dtype np.uint8")

    if keypoints is not None:
        if not isinstance(keypoints, np.ndarray):
            raise TypeError(
                "keypoints must be None or ndarray with shape (n_images, n_keypoints, 2)"
            )
        elif keypoints.ndim != 3:
            raise TypeError(
                "images must be ndarray with shape (n_images, n_keypoints, 2)"
            )
        elif keypoints.shape[0] != images.shape[0]:
            raise IndexError(
                "shape for `images` and `keypoints` must match along axis 0."
            )

    n_images = images.shape[0]
    height = images.shape[1]
    width = images.shape[2]
    n_channels = images.shape[3]
    skeleton = initialize_skeleton(skeleton)
    skeleton_names = skeleton["name"].values
    skeleton = skeleton[["tree", "swap_index"]].values
    n_keypoints = skeleton.shape[0]

    with h5py.File(datapath, mode="w") as h5file:
        h5file.create_dataset(
            dataset,
            shape=images.shape,
            dtype=np.uint8,
            data=images,
            maxshape=(None,) + images.shape[1:],
        )
        data = keypoints if keypoints is not None else -np.ones((n_images, n_keypoints, 2))
        h5file.create_dataset(
            "annotations",
            (n_images, n_keypoints, 2),
            dtype=np.float64,
            data=data,
            maxshape=(None,) + data.shape[1:],
        )
        data = np.zeros((n_images, n_keypoints), dtype=bool)
        h5file.create_dataset(
            "annotated",
            (n_images, n_keypoints),
            dtype=bool,
            data=data,
            maxshape=(None,) + data.shape[1:],
        )

        h5file.create_dataset("skeleton", skeleton.shape, dtype=np.int32, data=skeleton)
        h5file.create_dataset(
            "skeleton_names",
            (skeleton.shape[0],),
            dtype="S10",
            data=skeleton_names.astype("S10"),
        )


def merge_new_images(
    datapath,
    merged_datapath,
    images,
    keypoints=None,
    dataset="images",
    overwrite=False,
    mode="full",
):
    """
    Merge new images with an annotation set

    Parameters
    ----------
    datapath : str
        The path to the annotations file. Must be .h5
        e.g. '/path/to/file.h5'
    merged_datapath : str
        The path to save the merged annotations file. Must be .h5
        e.g. '/path/to/merged_file.h5'
    images : ndarray, shape (n_images, height, width, channels)
        A numpy array containing image data. 
        `images.dtype` should be np.uint8
    keypoints : None or ndarray, shape (n_images, n_keypoints, 2)
        Optionally pass keypoints for initializing annotations for the
        new images.
    dataset : str, default = "images"
        The dataset within the h5 file to save the images.
    overwrite: bool, default = False
        Whether to overwrite an existing .h5 file with the same name.
    mode : str
        The mode for loading the existing data. 
        Must be "annotated", or "full" (the full dataset)

    """

    if os.path.exists(merged_datapath) and overwrite is False:
        raise OSError(
            "Annotation set {} already exists. Delete the file or set `overwrite=True`.".format(
                merged_datapath
            )
        )
    if not isinstance(images, np.ndarray):
        raise TypeError(
            "images must be ndarray with shape (n_images, height, width, channels)"
        )
    elif images.ndim != 4:
        raise TypeError(
            "images must be ndarray with shape (n_images, height, width, channels)"
        )
    elif images.dtype != np.uint8:
        raise TypeError("`images` must be ndarray with dtype np.uint8")

    if keypoints is not None:
        if not isinstance(keypoints, np.ndarray):
            raise TypeError(
                "keypoints must be None or ndarray with shape (n_images, n_keypoints, 2)"
            )
        elif keypoints.ndim != 3:
            raise TypeError(
                "images must be ndarray with shape (n_images, n_keypoints, 2)"
            )
        elif keypoints.shape[0] != images.shape[0]:
            raise IndexError(
                "shape for `images` and `keypoints` must match along axis 0."
            )

    data_generator = DataGenerator(datapath, dataset=dataset, mode="full")

    if images.shape[1:] != data_generator.image_shape:
        raise IndexError(
            "`images` shape {} does not match existing dataset {}".format(
                images.shape[1:], data_generator.image_shape
            )
        )
    if keypoints is not None:
        if keypoints.shape[-1] == 3:
            keypoints = keypoints[:, :, :2]
        if keypoints.shape[1:] != data_generator.keypoints_shape:
            raise IndexError(
                "`keypoints` shape {} does not match existing dataset {}".format(
                    keypoints.shape[1:], data_generator.keypoints_shape
                )
            )

    h5file = h5py.File(datapath, mode="r")

    n_samples_merged = h5file[dataset].shape[0] + images.shape[0]

    merged_h5file = h5py.File(merged_datapath, "w")
    merged_h5file.create_dataset(
        dataset,
        shape=(n_samples_merged,) + data_generator.image_shape,
        dtype=np.uint8,
        maxshape=(None,) + data_generator.image_shape,
    )
    merged_h5file.create_dataset(
        "annotations",
        shape=(n_samples_merged,) + data_generator.keypoints_shape,
        dtype=np.float64,
        maxshape=(None,) + data_generator.keypoints_shape,
    )
    merged_h5file.create_dataset(
        "annotated",
        (n_samples_merged, data_generator.keypoints_shape[0]),
        dtype=bool,
        maxshape=(None, data_generator.keypoints_shape[0]),
    )
    merged_h5file.create_dataset(
        "skeleton", h5file["skeleton"].shape, dtype=np.int32, data=h5file["skeleton"][:]
    )

    for idx in range(h5file[dataset].shape[0]):
        merged_h5file[dataset][idx] = h5file[dataset][idx]
        merged_h5file["annotations"][idx] = h5file["annotations"][idx]
        merged_h5file["annotated"][idx] = h5file["annotated"][idx]

    for idx in range(h5file[dataset].shape[0], n_samples_merged):
        merged_h5file[dataset][idx] = images[idx - h5file[dataset].shape[0]]
        if keypoints is not None:
            merged_h5file["annotations"][idx] = keypoints[
                idx - h5file[dataset].shape[0]
            ]
        else:
            merged_h5file["annotations"][idx] = np.zeros(data_generator.keypoints_shape)
        merged_h5file["annotated"][idx] = np.zeros(
            data_generator.keypoints_shape[0], dtype=bool
        )

    h5file.close()
    merged_h5file.close()
