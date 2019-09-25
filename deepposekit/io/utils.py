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
import os

__all__ = ["initialize_image_set"]


def initialize_image_set(datapath, images, dataset="images", overwrite=False):
    """
    Intialize an image set for annotation as an h5 file

    Parameters
    ----------
    datapath : str
        The path to the annotations file. Must be .h5
        e.g. '/path/to/file.h5'
    images : ndarray, shape (n_images, height, width, channels)
        A numpy array containing image data. 
        `images.dtype` should be np.uint8
    dataset : str, default = "images"
        The dataset within the h5 file to save the images.
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

    with h5py.File(datapath, mode="w") as h5file:
        h5file.create_dataset(
            dataset,
            shape=images.shape,
            dtype=np.uint8,
            data=images,
            maxshape=(None,) + images.shape[1:],
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

    if keypoints:
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
    if keypoints:
        if keypoints.shape[1:] != data_generator.keypoints_shape:
            raise IndexError(
                "`keypoints` shape {} does not match existing dataset {}".format(
                    keypoints.shape, data_generator.keypoints_shape
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
        if keypoints:
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
