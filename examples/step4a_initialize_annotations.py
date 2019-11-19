#!/usr/bin/env python3
# coding: utf-8

# # DeepPoseKit Step 4a - Initialize annotations
#
# This is step 4a of the example scripts for using DeepPoseKit.
# This script shows you how to use your trained model to initialize
# the key-point labels for the unannotated images in your annotation set.

# This script will read the trained model file "best_model_densenet.h5"
# NOTE: this script will somehow MODIFY (read and then write) the example_annotation_set.h5 file.

import os
import sys

import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    from bootstrap import bootstrap_environment

    s_deep_pose_kit_data_dir = bootstrap_environment()
    s_trained_model_fname = os.path.join(s_deep_pose_kit_data_dir, "datasets", "fly", "best_model_densenet.h5")
    # s_inout_annot_fname = os.path.join(s_deep_pose_kit_data_dir, "datasets", "fly", "example_annotation_set.h5")
    s_inout_annot_fname = "example_annotation_set.h5"

    from deepposekit.models import load_model
    from deepposekit.io import DataGenerator, ImageGenerator

    # # Load the trained model
    # This loads the trained model into memory for making predictions

    print("Loading model...")
    model = load_model(s_trained_model_fname)
    print("Loading model: done!")

    # Let's initialize our `example_annotation_set.h5` from Step 1
    data_generator = DataGenerator(s_inout_annot_fname, mode='unannotated')
    image_generator = ImageGenerator(data_generator)

    # This passes the data generator to the model to get the predicted coordinates
    predictions = model.predict(image_generator, verbose=1)

    # This saves the predicted coordinates as initial keypoint locations for the unannotated data
    data_generator[:] = predictions

    # Indexing the generator, e.g. `data_generator[0]`
    # returns an image-keypoints pair, which you can then visualize.

    image, keypoints = data_generator[0]

    plt.figure(figsize=(5,5))
    image = image[0] if image.shape[-1] is 3 else image[0, ..., 0]
    cmap = None if image.shape[-1] is 3 else 'gray'
    plt.imshow(image, cmap=cmap, interpolation='none')
    for idx, jdx in enumerate(data_generator.graph):
        if jdx > -1:
            plt.plot(
                [keypoints[0, idx, 0], keypoints[0, jdx, 0]],
                [keypoints[0, idx, 1], keypoints[0, jdx, 1]],
                'r-'
            )
    plt.scatter(
        keypoints[0, :, 0],
        keypoints[0, :, 1],
        c=np.arange(data_generator.keypoints_shape[0]),
        s=50,
        cmap=plt.cm.hsv,
        zorder=3
    )
    plt.show()
#
