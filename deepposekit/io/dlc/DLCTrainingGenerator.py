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

from deepposekit.utils.image import check_grayscale
from deepposekit.io.TrainingGenerator import TrainingGenerator

import warnings

__all__ = ["DLCTrainingGenerator"]


class DLCTrainingGenerator(TrainingGenerator):
    """
    Generates training data with augmentation for a DeepLabCut annotation set.

    Automatically loads annotated data and produces
    augmented images and confidence maps for each keypoint.

    Only uses data that has been marked as annotated
    in the datapath file.

    Parameters
    ----------
    datapath : str
        The path to the annotations file. Must be .h5
        e.g. '/path/to/file.h5'
    imagepath : str
        Path to the image dataset used in the annotations file.
        e.g. '/path/to/images/'
    downsample_factor : int, default = 0
        The factor for determining the output shape of the confidence
        maps for estimating keypoints. This is determined as
        shape // 2**downsample_factor. The default is 0, which
        produces confidence maps that are the same shape
        as the input images.
    use_graph : bool, default = True
        Whether to generate confidence maps for the parent graph
        as lines drawn between connected keypoints. This can help reduce
        keypoint estimation error when training the network.
    augmenter : class or list : default = None
        A imgaug.Augmenter, or list of imgaug.Augmenter
        for applying augmentations to images and keypoints.
        Default is None, which applies no augmentations.
    shuffle : bool, default = True
        Whether to randomly shuffle the data.
    sigma : float, default = 3
        The standard deviation of the Gaussian confidence peaks.
        This is scaled to sigma // 2**downsample_factor.
    validation_split : float, default = 0.0
        Float between 0 and 1. Fraction of the training data to be used
        as validation data. The generator will set apart this fraction
        of the training data, will not generate this data unless
        the `validation` flag is set to True when the class is called.
    graph_scale : float, default = 1.0
        Float between 0 and 1. A factor to scale the edge
        confidence map values to y * edge_scale.
        The default is 1.0 which does not scale the confidence
        values. This is useful for preventing the edge channels
        from dominating the error when training a smaller network.
        This arg is not used when `use_graph` is set to False.
    random_seed : int, default = None
        set random seed for selecting validation data
    """

    def __init__(
        self,
        datapath,
        imagepath,
        downsample_factor=2,
        augmenter=None,
        shuffle=True,
        sigma=5,
        validation_split=0.1,
        random_seed=None,
    ):
        self.datapath = datapath
        self.imagepath = imagepath

        self.random_seed = random_seed
        if self.random_seed:
            np.random.seed(self.random_seed)

        self.shuffle = shuffle

        if isinstance(downsample_factor, int):
            if downsample_factor >= 0:
                self.downsample_factor = downsample_factor
            else:
                raise ValueError("""downsample factor must be >= 0""")
        else:
            raise TypeError("""downsample_factor must be type int""")
        self.sigma = sigma
        self.output_sigma = sigma / 2.0 ** downsample_factor
        self.batch_size = 32
        self.n_outputs = 1
        if 0 <= validation_split < 1:
            self.validation_split = validation_split
        else:
            raise ValueError("`validation_split` must be >=0 and <1")
        self.validation = False
        self.confidence = True
        self.use_edges = False
        self.use_graph = False
        self.edge_scale = 1
        self.graph_scale = 1
        self._init_augmenter(augmenter)
        self._init_data(datapath, imagepath)
        self.on_epoch_end()

    def _init_data(self, datapath, imagepath):

        self.generator = DLCDataGenerator(datapath, imagepath)
        self.n_samples = len(self.generator)
        if self.n_samples <= 0:
            raise AttributeError(
                "`n_samples` is 0. `datapath` or `dataset` appears to be empty"
            )

        # Get image attributes and
        # define output shape
        test_image = self.generator[0][0][0]
        self.height = test_image.shape[0]
        self.width = test_image.shape[1]
        image, grayscale = check_grayscale(test_image)
        if grayscale or test_image.ndim == 2:
            self.n_channels = 1
        else:
            self.n_channels = test_image.shape[-1]

        self.output_shape = (
            self.height // 2 ** self.downsample_factor,
            self.width // 2 ** self.downsample_factor,
        )

        # Training/validation split
        # indices for validation set in sample_index
        self.index = np.arange(self.n_samples)
        self.n_validation = int(self.validation_split * self.n_samples)
        if self.n_validation is 0:
            warnings.warn(
                "`n_validation` is 0. Increase `validation_split` to use a validation set."
            )

        self.index = np.arange(self.n_samples)
        self.n_validation = int(self.validation_split * self.n_samples)
        val_index = np.random.choice(self.index, self.n_validation, replace=False)
        self.val_index = self.index[val_index]
        # indices for training set in  sample_index
        train_index = np.invert(np.isin(self.index, self.val_index))
        self.train_index = self.index[train_index]
        self.n_train = len(self.train_index)

        # Initialize skeleton attributes
        self.graph = None
        self.swap_index = None
        self.n_keypoints = self.generator.n_keypoints
        self.n_branches = None
        self.on_epoch_end()
        X, y = self.__getitem__(0)
        self.n_output_channels = y.shape[-1]

    def get_config(self):
        if self.augmenter:
            augmenter = True
        else:
            augmenter = False
        config = {
            "shuffle": self.shuffle,
            "downsample_factor": self.downsample_factor,
            "sigma": self.sigma,
            "validation_split": self.validation_split,
            "datapath": self.datapath,
            "imagepath": self.imagepath,
            "output_shape": self.output_shape,
            "n_train": self.n_train,
            "n_validation": self.n_validation,
            "random_seed": self.random_seed,
            "n_output_channels": self.n_output_channels,
            "augmenter": augmenter,
            "n_keypoints": self.n_keypoints,
        }
        return config


if __name__ == "__main__":
    train_generator = DLCTrainingGenerator(
        datapath="./deeplabcut/examples/openfield-Pranav-2018-10-30/labeled-data/m4s1/CollectedData_Pranav.h5",
        imagepath="./deeplabcut/examples/openfield-Pranav-2018-10-30/",
    )
