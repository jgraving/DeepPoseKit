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

from tensorflow.python.keras.saving import save
import h5py
import json
from deepposekit.utils.io import get_json_type


def save_model(model, path, optimizer=True):

    if isinstance(path, str):
        if path.endswith(".h5") or path.endswith(".hdf5"):
            filepath = path
        else:
            raise ValueError("file must be .h5 file")
    else:
        raise TypeError("file must be type `str`")

    save.save_model(model.train_model, path, include_optimizer=optimizer)

    with h5py.File(filepath, "r+") as h5file:

        train_generator = model.train_generator

        h5file.attrs["train_generator_config"] = json.dumps(
            {
                "class_name": train_generator.__class__.__name__,
                "config": train_generator.get_config(),
            },
            default=get_json_type,
        ).encode("utf8")

        h5file.attrs["pose_model_config"] = json.dumps(
            {"class_name": model.__class__.__name__, "config": model.get_config()},
            default=get_json_type,
        ).encode("utf8")
