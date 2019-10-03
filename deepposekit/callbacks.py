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
import json

from tensorflow.keras.callbacks import Callback
import tensorflow.keras.callbacks as callbacks
from tensorflow.python.platform import tf_logging as logging

from deepposekit.models.engine import BaseModel
from deepposekit.utils.io import get_json_type


class Logger(Callback):
    """ Saves the loss and validation metrics during training

    Parameters
    ----------
    filepath: str
        Name of the .h5 file.
    validation_batch_size: int
        Batch size for running evaluation
    """

    def __init__(
        self,
        filepath=None,
        validation_batch_size=1,
        confidence_threshold=None,
        verbose=1,
        batch_size=None,
        **kwargs
    ):

        super(Logger, self).__init__(**kwargs)
        if isinstance(filepath, str):
            if filepath.endswith(".h5"):
                self.filepath = filepath
            else:
                raise ValueError("filepath must be .h5 file")
        elif filepath is not None:
            raise TypeError("filepath must be type `str` or None")
        else:
            self.filepath = filepath

        self.verbose = verbose
        self.batch_size = validation_batch_size if batch_size is None else batch_size
        self.confidence_threshold = confidence_threshold

        if self.filepath is not None:
            with h5py.File(self.filepath, "w") as h5file:
                if "logs" not in h5file:
                    group = h5file.create_group("logs")
                    group.create_dataset(
                        "loss", shape=(0,), dtype=np.float64, maxshape=(None,)
                    )
                    group.create_dataset(
                        "val_loss", shape=(0,), dtype=np.float64, maxshape=(None,)
                    )
                    group.create_dataset(
                        "y_pred",
                        shape=(0, 0, 0, 0),
                        dtype=np.float64,
                        maxshape=(None, None, None, None),
                    )
                    group.create_dataset(
                        "y_error",
                        shape=(0, 0, 0, 0),
                        dtype=np.float64,
                        maxshape=(None, None, None, None),
                    )
                    group.create_dataset(
                        "euclidean",
                        shape=(0, 0, 0),
                        dtype=np.float64,
                        maxshape=(None, None, None),
                    )
                    group.create_dataset(
                        "confidence",
                        shape=(0, 0, 0),
                        dtype=np.float64,
                        maxshape=(None, None, None),
                    )

    def on_train_begin(self, logs):
        return

    def on_train_end(self, logs):
        return

    def on_epoch_begin(self, epoch, logs):
        return

    def on_epoch_end(self, epoch, logs):
        logs = logs or {}
        evaluation_dict = self.evaluation_model.evaluate(self.batch_size)
        y_pred = evaluation_dict["y_pred"]
        y_error = evaluation_dict["y_error"]
        euclidean = evaluation_dict["euclidean"]
        confidence = evaluation_dict["confidence"]
        if self.filepath is not None:
            with h5py.File(self.filepath) as h5file:
                values = {
                    "val_loss": np.array([logs.get("val_loss")]).reshape(1),
                    "loss": np.array([logs.get("loss")]).reshape(1),
                    "y_pred": y_pred[None, ...],
                    "y_error": y_error[None, ...],
                    "euclidean": euclidean[None, ...],
                    "confidence": confidence[None, ...],
                }
                for key, value in values.items():
                    data = h5file["logs"][key]
                    value = np.array(value)
                    data.resize(tuple(value.shape))
                    if data.shape[0] == 0:
                        data[:] = value
                    else:
                        data.resize(data.shape[0] + 1, axis=0)
                        data[-1] = value

        euclidean = euclidean.flatten()
        confidence = confidence.flatten()

        if self.confidence_threshold:
            mask = confidence >= confidence_threshold
            euclidean = euclidean[mask]
            confidence = confidence[mask]

        keypoint_percentile = np.percentile(
            [euclidean, confidence], [0, 5, 25, 50, 75, 95, 100], axis=1
        ).T
        euclidean_perc, confidence_perc = keypoint_percentile

        euclidean_mean, confidence_mean = np.mean([euclidean, confidence], axis=1)

        logs["euclidean"] = euclidean_mean
        logs["confidence"] = confidence_mean

        if self.verbose:
            print(
                "evaluation_metrics: \n"
                "euclidean - mean: {:5.2f} (0%: {:5.2f}, 5%: {:5.2f}, 25%: {:5.2f}, 50%: {:5.2f}, 75%: {:5.2f}, 95%: {:5.2f}, 100%: {:5.2f}) \n"
                "confidence - mean: {:5.2f} (0%: {:5.2f}, 5%: {:5.2f}, 25%: {:5.2f}, 50%: {:5.2f}, 75%: {:5.2f}, 95%: {:5.2f}, 100%: {:5.2f}) \n".format(
                    euclidean_mean,
                    euclidean_perc[0],
                    euclidean_perc[1],
                    euclidean_perc[2],
                    euclidean_perc[3],
                    euclidean_perc[4],
                    euclidean_perc[5],
                    euclidean_perc[6],
                    confidence_mean,
                    confidence_perc[0],
                    confidence_perc[1],
                    confidence_perc[2],
                    confidence_perc[3],
                    confidence_perc[4],
                    confidence_perc[5],
                    confidence_perc[6],
                )
            )

    def on_batch_begin(self, batch, logs):
        return

    def on_batch_end(self, batch, logs):
        return

    def pass_model(self, model):
        if isinstance(model, BaseModel):
            self.evaluation_model = model
        else:
            raise TypeError("model must be a deepposekit BaseModel class")
        if self.filepath is not None:
            with h5py.File(self.filepath, "r+") as h5file:
                # create attributes for the group based on the two dicts
                for key, value in model.get_config().items():
                    if isinstance(value, str):
                        value = value.encode("utf8")  # str not supported in h5py
                    if value is None:
                        value = "None".encode("utf8")
                    if key not in h5file.attrs:
                        h5file.attrs.create(key, value)

                if "logger_config" not in h5file.attrs:
                    h5file.attrs["logger_config"] = json.dumps(
                        model.get_config(), default=get_json_type
                    ).encode("utf8")


class ModelCheckpoint(callbacks.ModelCheckpoint):
    """Save the model after every epoch.

    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).

    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.

    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_freq: `'epoch'` or integer. When using `'epoch'`, the callback saves
            the model after each epoch. When using integer, the callback saves the
            model at end of a batch at which this many samples have been seen since
            last saving. Note that if the saving isn't aligned to epochs, the
            monitored metric may potentially be less reliable (it could reflect as
            little as 1 batch, since the metrics get reset every epoch). Defaults to
            `'epoch'`
    """

    def __init__(
        self,
        filepath,
        monitor="val_loss",
        verbose=0,
        save_best_only=True,
        mode="auto",
        save_freq="epoch",
        **kwargs
    ):
        super(ModelCheckpoint, self).__init__(
            filepath=filepath,
            monitor=monitor,
            verbose=verbose,
            save_best_only=save_best_only,
            mode=mode,
            save_freq=save_freq,
            **kwargs
        )

    def pass_model(self, model):
        if isinstance(model, BaseModel):
            self.model = model
        else:
            raise TypeError("model must be a deepposekit BaseModel class")

    def set_model(self, model):
        pass
