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
import imgaug.augmenters as iaa

import numpy as np
import copy

from deepposekit.utils.keypoints import draw_confidence_maps, graph_to_edges
from deepposekit.utils.image import check_grayscale
from deepposekit.io.BaseGenerator import BaseGenerator

import warnings

__all__ = ["TrainingGenerator"]


class TrainingGenerator(Sequence):
    """
    Generates data for training a model.
    
    Automatically loads annotated data and produces
    augmented images and confidence maps for each keypoint.

    Parameters
    ----------
    generator: deepposekit.io.BaseGenerator
        An instance of BaseGenerator (deepposekit.io.BaseGenerator) object.
        The output of the generator must be `(images, keypoints)`, where images
        are a numpy array of shape (n_images, height, width, channels), and 
        keypoints are a numpy array of shape (n_images, n_keypoints, 2), where
        2 is the row, column coordinates of the keypoints in each image.
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
    augmenter : class or list, default = None
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
        generator,
        downsample_factor=2,
        use_graph=True,
        augmenter=None,
        shuffle=True,
        sigma=5,
        validation_split=0.0,
        graph_scale=1.0,
        random_seed=None,
    ):

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
        self.use_graph = use_graph
        self.graph_scale = graph_scale

        if 0 <= validation_split < 1:
            self.validation_split = validation_split
        else:
            raise ValueError("`validation_split` must be >=0 and <1")
        self.validation = False
        self.confidence = True
        self._init_augmenter(augmenter)
        self._init_data(generator)
        self.on_epoch_end()

    def _init_augmenter(self, augmenter):
        if isinstance(augmenter, type(None)):
            self.augmenter = augmenter
        elif isinstance(augmenter, iaa.Augmenter):
            self.augmenter = augmenter
        elif isinstance(augmenter, list):
            if isinstance(augmenter[0], iaa.Augmenter):
                self.augmenter = iaa.Sequential(augmenter)
            else:
                raise TypeError(
                    """`augmenter` must be class Augmenter
                            (imgaug.augmenters.Augmenter)
                            or list of Augmenters"""
                )
        else:
            raise ValueError(
                """augmenter must be class
                             Augmenter, list of Augmenters, or None"""
            )

    def _init_data(self, generator):

        if isinstance(generator, BaseGenerator):
            self.generator = generator
        else:
            raise TypeError(
                "`generator` must be a subclass of `deepposekit.io.BaseGenerator`"
                " such as `deepposekit.io.DataGenerator` or `deepposekit.io.DLCDataGenerator`."
            )
        self.n_samples = len(self.generator)
        if self.n_samples <= 0:
            raise AttributeError(
                "`n_samples` is 0. `datapath` or `dataset` appears to be empty"
            )

        # Get image attributes and
        # define output shape
        self.height = self.generator.image_shape[0]
        self.width = self.generator.image_shape[1]
        self.n_channels = self.generator.image_shape[2]

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

        val_index = np.random.choice(self.index, self.n_validation, replace=False)
        self.val_index = self.index[val_index]
        # indices for training set in  sample_index
        train_index = np.invert(np.isin(self.index, self.val_index))
        self.train_index = self.index[train_index]
        self.n_train = len(self.train_index)
        self.n_keypoints = self.generator.keypoints_shape[0]

        # Initialize skeleton attributes
        self.graph = self.generator.graph
        self.swap_index = self.generator.swap_index

        self.on_epoch_end()
        X, y = self.__getitem__(0)
        self.n_output_channels = y.shape[-1]

    def __len__(self):
        """The number of batches per epoch"""
        if self.validation:
            return self.n_validation // self.batch_size
        else:
            return self.n_train // self.batch_size

    def __call__(self, n_outputs=1, batch_size=32, validation=False, confidence=True):
        """ Sets the number of outputs and the batch size

        Parameters
        ----------
        n_outputs : int, default = 1
            The number of outputs to generate.
            This is needed for applying intermediate supervision
            to a network with multiple output layers.
        batch_size : int, default = 32
            Number of samples in each batch
        validation: bool, default False
            If set to True, will generate the validation set.
            Otherwise, generates the training set.
        confidence: bool, default True
            If set to True, will generate confidence maps.
            Otherwise, generates keypoints.

        """
        self.n_outputs = n_outputs
        self.batch_size = batch_size
        if validation:
            if self.n_validation is 0 and self.validation_split is 0:
                warnings.warn(
                    "`validation_split` is 0, so there will be no validation step. "
                    "callbacks that rely on `val_loss` should be switched to `loss` or removed."
                )
            if self.n_validation is 0 and self.validation_split is not 0:
                warnings.warn(
                    "`validation_split` is too small, so there will be no validation step. "
                    "`validation_split` should be increased or "
                    "callbacks that rely on `val_loss` should be switched to 'loss' or removed."
                )

        self.validation = validation
        self.confidence = confidence
        self.on_epoch_end()
        self_copy = copy.deepcopy(self)
        if self.augmenter is not None:
            self_copy.augmenter.reseed()
        return self_copy

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        idx0 = index * self.batch_size
        idx1 = (index + 1) * self.batch_size
        if self.validation:
            indexes = self.val_range[idx0:idx1]
        else:
            indexes = self.train_range[idx0:idx1]

        # Generate data
        X, y = self.generate_batch(indexes)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.train_range = np.arange(self.n_train)
        self.val_range = np.arange(self.n_validation)
        if self.shuffle:
            np.random.shuffle(self.train_range)
            np.random.shuffle(self.val_range)

    def load_batch(self, indexes):
        if self.validation:
            batch_index = self.val_index[indexes]
        else:
            batch_index = self.train_index[indexes]
        return self.generator[batch_index]

    def augment(self, images, keypoints):
        images_aug = []
        keypoints_aug = []
        for idx in range(images.shape[0]):
            images_idx = images[idx, None]
            keypoints_idx = keypoints[idx, None]
            augmented_idx = self.augmenter(images=images_idx, keypoints=keypoints_idx)
            images_aug_idx, keypoints_aug_idx = augmented_idx
            images_aug.append(images_aug_idx)
            keypoints_aug.append(keypoints_aug_idx)

        images_aug = np.concatenate(images_aug)
        keypoints_aug = np.concatenate(keypoints_aug)
        return images_aug, keypoints_aug

    def generate_batch(self, indexes):
        """Generates data containing batch_size samples"""
        X, y = self.load_batch(indexes)
        if self.augmenter is not None and not self.validation:
            X, y = self.augment(X, y)
        if self.confidence:
            y = draw_confidence_maps(
                images=X,
                keypoints=y,
                graph=self.graph,
                output_shape=self.output_shape,
                use_graph=self.use_graph,
                sigma=self.output_sigma,
            )
            y *= 255
            if self.use_graph:
                y[..., self.n_keypoints :] *= self.graph_scale
        if self.n_outputs > 1:
            y = [y for idx in range(self.n_outputs)]

        return X, y

    def get_config(self):
        if self.augmenter is not None:
            augmenter = True
        else:
            augmenter = False
        config = {
            "n_train": self.n_train,
            "n_validation": self.n_validation,
            "validation_split": self.validation_split,
            "downsample_factor": self.downsample_factor,
            "output_shape": self.output_shape,
            "n_output_channels": self.n_output_channels,
            "shuffle": self.shuffle,
            "sigma": self.sigma,
            "output_sigma": self.output_sigma,
            "use_graph": self.use_graph,
            "graph_scale": self.graph_scale,
            "random_seed": self.random_seed,
            "augmenter": augmenter,
        }
        base_config = self.generator.get_config()
        return dict(list(config.items()) + list(base_config.items()))
