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
from tensorflow.keras import Model
import warnings

from deepposekit.models.layers.subpixel import SubpixelMaxima2D
from deepposekit.models.layers.convolutional import Maxima2D
from deepposekit.utils.image import largest_factor
from deepposekit.utils.keypoints import keypoint_errors
from deepposekit.models.saving import save_model


class BaseModel:
    def __init__(self, data_generator=None, subpixel=False, **kwargs):

        self.data_generator = data_generator
        self.subpixel = subpixel
        if "skip_init" not in kwargs:
            if self.train_model is NotImplemented:
                self.__init_model__()
                self.__init_train_model__()
            if self.data_generator is not None:
                if self.subpixel:
                    output_sigma = self.data_generator.output_sigma
                else:
                    output_sigma = None
                self.__init_predict_model__(
                    self.data_generator.output_shape,
                    self.data_generator.n_keypoints,
                    self.data_generator.downsample_factor,
                    output_sigma,
                )

    train_model = NotImplemented

    def __init_train_model__(self):
        if isinstance(self.train_model, Model):
            self.compile = self.train_model.compile
            self.n_outputs = len(self.train_model.outputs)
        else:
            raise TypeError("self.train_model must be keras.Model class")

    def __init_model__(self):
        raise NotImplementedError(
            "__init_model__ method must be" "implemented to define `self.train_model`"
        )

    def __init_predict_model__(
        self, output_shape, n_keypoints, downsample_factor, output_sigma=None, **kwargs
    ):

        output = self.train_model.outputs[-1]
        if self.subpixel:
            kernel_size = np.min(output_shape)
            kernel_size = (kernel_size // largest_factor(kernel_size)) + 1
            sigma = output_sigma
            keypoints = SubpixelMaxima2D(
                kernel_size,
                sigma,
                upsample_factor=100,
                index=n_keypoints,
                coordinate_scale=2 ** downsample_factor,
                confidence_scale=255.0,
            )(output)
        else:
            keypoints = Maxima2D(
                index=n_keypoints,
                coordinate_scale=2 ** downsample_factor,
                confidence_scale=255.0,
            )(output)
        input_layer = self.train_model.inputs[0]
        self.predict_model = Model(input_layer, keypoints, name=self.train_model.name)
        self.predict = self.predict_model.predict
        self.predict_generator = self.predict_model.predict_generator
        self.predict_on_batch = self.predict_model.predict_on_batch

    def fit(
        self,
        batch_size,
        validation_batch_size=1,
        callbacks=[],
        epochs=1,
        use_multiprocessing=False,
        n_workers=1,
        steps_per_epoch=None,
        **kwargs
    ):
        """
        Trains the model for a given number of epochs (iterations on a dataset).

        Parameters
        ----------
        batch_size : int
            Number of samples per training update.
        validation_batch_size : int
            Number of samples per validation batch used when evaluating the model.
        callbacks : list or None
            List of keras.callbacks.Callback instances or deepposekit callbacks.
            List of callbacks to apply during training and validation.
        epochs: int
            Number of epochs to train the model. An epoch is an iteration over the entire dataset,
            or for `steps_per_epoch` number of batches
        use_multiprocessing: bool, default=False
            Whether to use the multiprocessing module when generating batches of data.
        n_workers: int
            Number of processes to run for generating batches of data.
        steps_per_epoch: int or None
            Number of batches per epoch. If `None` this is automatically determined
            based on the size of the dataset.
        """

        if not self.train_model._is_compiled:
            warnings.warn(
                """\nAutomatically compiling with default settings: model.compile('adam', 'mse')\n"""
                "Call model.compile() manually to use non-default settings.\n"
            )
            self.train_model.compile("adam", "mse")

        train_generator = self.data_generator(
            self.n_outputs, batch_size, validation=False, confidence=True
        )
        validation_generator = self.data_generator(
            self.n_outputs, validation_batch_size, validation=True, confidence=True
        )
        validation_generator = (
            None if len(validation_generator) == 0 else validation_generator
        )
        if validation_generator is None:
            warnings.warn(
                "No validation set detected, so validation step will not be run and `val_loss` will not be available."
            )

        activated_callbacks = self.activate_callbacks(callbacks)

        self.train_model.fit_generator(
            generator=train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            use_multiprocessing=use_multiprocessing,
            workers=n_workers,
            callbacks=activated_callbacks,
            validation_data=validation_generator,
            **kwargs
        )

    def activate_callbacks(self, callbacks):
        activated_callbacks = []
        if callbacks is not None:
            if len(callbacks) > 0:
                for callback in callbacks:
                    if hasattr(callback, "pass_model"):
                        callback.pass_model(self)
                    activated_callbacks.append(callback)
        return activated_callbacks

    def evaluate(self, batch_size):

        if self.data_generator.n_validation > 0:
            keypoint_generator = self.data_generator(
                n_outputs=1, batch_size=batch_size, validation=True, confidence=False
            )

        elif self.data_generator.n_validation == 0:
            warnings.warn(
                "`n_validation` is 0, so the training set will be used for model evaluation"
            )
            keypoint_generator = self.data_generator(
                n_outputs=1, batch_size=batch_size, validation=False, confidence=False
            )

        metrics = []
        keypoints = []
        for idx in range(len(keypoint_generator)):
            X, y_true = keypoint_generator[idx]
            y_pred = self.predict_model.predict_on_batch(X)
            y_pred = y_pred[..., :2]
            errors = keypoint_errors(y_true, y_pred)
            y_error, euclidean, mae, mse, rmse = errors
            confidence = y_pred[..., -1]
            metrics.append([euclidean, mae, mse, rmse, confidence])
            keypoints.append([y_pred, y_error])

        metrics = np.hstack(metrics)
        keypoints = np.hstack(keypoints)

        euclidean, mae, mse, rmse, confidence = metrics
        y_pred, y_error = keypoints

        evaluation_dict = {
            "y_pred": y_pred,
            "y_error": y_error,
            "euclidean": euclidean,
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "confidence": confidence,
        }

        return evaluation_dict

    def save(self, path, optimizer=True):
        save_model(self, path, optimizer)

    def get_config(self):
        config = {}
        if self.data_generator:
            base_config = self.data_generator.get_config()
            return dict(list(base_config.items()) + list(config.items()))
        else:
            return config
