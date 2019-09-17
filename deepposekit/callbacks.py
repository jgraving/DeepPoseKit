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
import h5py
import json

from keras.callbacks import Callback
import keras.callbacks as callbacks
from .models.engine import BaseModel
from .utils.io import get_json_type


class Logger(Callback):
    """ Saves the loss and validation metrics during training

    Parameters
    ----------
    filepath: str
        Name of the .h5 file.
    validation_batch_size: int
        Batch size for running evaluation
    '''
    def __init__(self, filepath, validation_batch_size=1, confidence_threshold=None, verbose=1, batch_size=None, **kwargs):

    def __init__(
        self,
        filepath=None,
        validation_batch_size=1,
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
        if self.filepath is not None:
        self.confidence_threshold = confidence_threshold

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
                        "mae",
                        shape=(0, 0, 0),
                        dtype=np.float64,
                        maxshape=(None, None, None),
                    )
                    group.create_dataset(
                        "mse",
                        shape=(0, 0, 0),
                        dtype=np.float64,
                        maxshape=(None, None, None),
                    )
                    group.create_dataset(
                        "rmse",
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

                                     dtype=np.float64, maxshape=(None,))
            if 'logs' not in h5file:
                group = h5file.create_group('logs')
                group.create_dataset('loss', shape=(0,),
                group.create_dataset('val_loss', shape=(0,),
                                     dtype=np.float64, maxshape=(None,))
                group.create_dataset('y_pred', shape=(0, 0, 0, 0),
                                     dtype=np.float64,
                                     maxshape=(None, None, None, None))
                group.create_dataset('y_error', shape=(0, 0, 0, 0),
                                     dtype=np.float64,
                                     maxshape=(None, None, None, None))
                group.create_dataset('euclidean', shape=(0, 0, 0),
                                     dtype=np.float64,
                                     maxshape=(None, None, None))
                group.create_dataset('confidence', shape=(0, 0, 0),
                                     dtype=np.float64,
                                     maxshape=(None, None, None))

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        evaluation_dict = self.evaluation_model.evaluate(self.batch_size)
        y_pred = evaluation_dict['y_pred']
        y_error = evaluation_dict['y_error']
        euclidean = evaluation_dict['euclidean']
        confidence = evaluation_dict['confidence']

        with h5py.File(self.filepath) as h5file:
            values = {'loss': np.array([logs.get('loss')]).reshape(1,),
                      'val_loss': np.array([logs.get('val_loss')]).reshape(1,),
                      'y_pred': y_pred[None, ...],
                      'y_error': y_error[None, ...],
                      'euclidean': euclidean[None, ...],
                      'confidence': confidence[None, ...]}

              for key, value in values.items():
                  data = h5file["logs"][key]
                  if data.shape[0] == 0:
                      value = np.array(value)
                      data.resize(tuple(value.shape))
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

        keypoint_percentile = np.percentile([euclidean,
                                             confidence],
                                            [2.5, 25, 75, 97.5], axis=1).T
        euclidean_perc, confidence_perc = keypoint_percentile

        euclidean_mean, confidence_mean = np.mean([euclidean, confidence], axis=1)

        logs['euclidean'] = euclidean_mean
        logs['confidence'] = confidence_mean

        euclidean_median, confidence_median = np.median([euclidean, confidence], axis=1)

        if self.verbose:
            print('evaluation_metrics: mean median (2.5%, 25%, 75%, 97.5%) \n'
                  'euclidean: {:6.4f} {:6.4f} ({:6.4f}, {:6.4f}, {:6.4f}, {:6.4f}) \n'
                  'confidence: {:6.4f} {:6.4f} ({:6.4f}, {:6.4f}, {:6.4f}, {:6.4f}) \n'
                  .format(euclidean_mean, euclidean_median, euclidean_perc[0], euclidean_perc[1], euclidean_perc[2], euclidean_perc[3],
                          confidence_mean, confidence_median, confidence_perc[0], confidence_perc[1], confidence_perc[2], confidence_perc[3]
                          )
                  )

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
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
        model: pose.BaseModel class, a pose model to save
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
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(
        self,
        filepath,
        monitor="rmse",
        verbose=0,
        save_best_only=False,
        mode="auto",
        period=1,
        optimizer=True,
    ):
        super(ModelCheckpoint, self).__init__(
            filepath=filepath,
            monitor=monitor,
            verbose=verbose,
            save_best_only=save_best_only,
            mode=mode,
            period=period,
        )
        self.optimizer = optimizer

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn(
                        "Can save best model only with %s available, "
                        "skipping." % (self.monitor),
                        RuntimeWarning,
                    )
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print(
                                "\nEpoch %05d: %s improved from %0.5f to %0.5f,"
                                " saving model to %s"
                                % (
                                    epoch + 1,
                                    self.monitor,
                                    self.best,
                                    current,
                                    filepath,
                                )
                            )
                        self.best = current
                        self.save_model.save(filepath, optimizer=self.optimizer)
                    else:
                        if self.verbose > 0:
                            print(
                                "\nEpoch %05d: %s did not improve from %0.5f"
                                % (epoch + 1, self.monitor, self.best)
                            )
            else:
                if self.verbose > 0:
                    print("\nEpoch %05d: saving model to %s" % (epoch + 1, filepath))
                self.save_model.save(filepath, optimizer=self.optimizer)

    def pass_model(self, model):
        if isinstance(model, BaseModel):
            self.save_model = model
        else:
            raise TypeError("model must be a deepposekit BaseModel class")
