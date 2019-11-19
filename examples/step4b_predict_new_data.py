#!/usr/bin/env python3
# coding: utf-8

# # DeepPoseKit Step 4b - Predict on new data
#
# This is step 4b of the example scripts for using DeepPoseKit.
# This script shows you how to use your trained model to make
# predictions on a novel video, detect outliers, merge the outliers
# with the existing annotation set, and visualize the data output.

# This script will read the trained model file "best_model_densenet.h5",
# read raw video from fly/video.avi,
# call model.predict() on that video and save resulting output into
# file called "predictions.npy" in the current directory.
# Then detect outliers and merge outliers with the annotation set and save result
# into file called "annotation_data_release_merged.h5" in the current directory.
# Then open original raw video again, draw predicted key-point positions on it
# and save resulting frames int file called "fly_posture.mp4" in the current directory.


import os
import sys

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


if __name__ == '__main__':
    from bootstrap import bootstrap_environment

    s_deep_pose_kit_data_dir = bootstrap_environment()
    s_trained_model_fname = os.path.join(s_deep_pose_kit_data_dir, "datasets", "fly", "best_model_densenet.h5")
    s_input_video_fname = os.path.join(s_deep_pose_kit_data_dir, "datasets", "fly", "video.avi")
    s_input_annot_fname = os.path.join(s_deep_pose_kit_data_dir, "datasets", "fly", "annotation_data_release.h5")
    s_out_predict_fname = "predictions.npy"
    s_out_annot_fname = "annotation_data_release_merged.h5"
    s_out_video_fname = "fly_posture.mp4"

    from deepposekit.models import load_model
    from deepposekit.io import DataGenerator, VideoReader, VideoWriter
    from deepposekit.io.utils import merge_new_images

    if not os.path.isfile(s_out_predict_fname):
        # # Makes predictions for the full video
        # This loads batches of frames and makes predictions.
        print("INFO: output file not found. Load model and predict positions...")

        oc_reader = VideoReader(s_input_video_fname, batch_size=50, gray=True)
        oc_model = load_model(s_trained_model_fname)
        predictions = oc_model.predict(oc_reader, verbose=1)
        oc_reader.close()
        np.save(s_out_predict_fname, predictions)
    else:
        print("INFO: output file found. ***SKIP prediction process ***")
        predictions = np.load(s_out_predict_fname)

    # This splits the predictions into their x-y coordinates,
    # and confidence scores from each confidence map.
    x, y, confidence = np.split(predictions, 3, -1)

    oc_reader = VideoReader(s_input_video_fname, batch_size=50, gray=True)
    frames = oc_reader[0]
    
    # Visualize the data output
    data_generator = DataGenerator(s_input_annot_fname)

    image = frames[0]
    keypoints = predictions[0]

    plt.figure(figsize=(5,5))
    image = image if image.shape[-1] is 3 else image[..., 0]
    cmap = None if image.shape[-1] is 3 else 'gray'
    plt.imshow(image, cmap=cmap, interpolation='none')
    for idx, jdx in enumerate(data_generator.graph):
        if jdx > -1:
            plt.plot(
                [keypoints[idx, 0], keypoints[jdx, 0]],
                [keypoints[idx, 1], keypoints[jdx, 1]],
                'r-'
            )
    plt.scatter(
        keypoints[:, 0],
        keypoints[:, 1],
        c=np.arange(data_generator.keypoints_shape[0]),
        s=50,
        cmap=plt.cm.hsv,
        zorder=3
    )
    plt.show()

    # # Detect outlier frames
    # This is a basic example of how to use confidence scores and temporal derivatives
    # to detect potential outliers and add them to the annotation set.
    # Plot the confidence scores

    confidence_diff = np.abs(np.diff(confidence.mean(-1).mean(-1)))

    plt.figure(figsize=(15, 3))
    plt.plot(confidence_diff)
    plt.show()

    # Use `scipy.signal.find_peaks` to detect outliers
    confidence_outlier_peaks = find_peaks(confidence_diff, height=0.1)[0]

    plt.figure(figsize=(15, 3))
    plt.plot(confidence_diff)
    plt.plot(confidence_outlier_peaks, confidence_diff[confidence_outlier_peaks], 'ro')
    plt.show()

    # Calculate the key-point derivatives and plot them
    time_diff = np.diff(predictions[..., :2], axis=0)
    time_diff = np.abs(time_diff.reshape(time_diff.shape[0], -1))
    time_diff = time_diff.mean(-1)

    plt.figure(figsize=(15, 3))
    plt.plot(time_diff)
    plt.show()

    # Use `scipy.signal.find_peaks` to detect outliers
    time_diff_outlier_peaks = find_peaks(time_diff, height=10)[0]

    plt.figure(figsize=(15, 3))
    plt.plot(time_diff)
    plt.plot(time_diff_outlier_peaks, time_diff[time_diff_outlier_peaks], 'ro')
    plt.show()

    # Combine the detected outliers into a single index
    outlier_index = np.concatenate((confidence_outlier_peaks, time_diff_outlier_peaks))
    outlier_index = np.unique(outlier_index) # make sure there are no repeats

    # Grab the frames and corresponding key-points for the selected outliers
    oc_reader = VideoReader(s_input_video_fname, batch_size=1, gray=True)
    outlier_images = []
    outlier_keypoints = []
    for idx in outlier_index:
        outlier_images.append(oc_reader[idx])
        outlier_keypoints.append(predictions[idx])

    outlier_images = np.concatenate(outlier_images)
    outlier_keypoints = np.stack(outlier_keypoints)
    oc_reader.close()

    # Visualize the outlier frames and key-point predictions
    # data_generator = DataGenerator(s_input_annot_fname)

    # The len(outlier_images) == 56. Process only first 5 images here...
    for idx in range(5):
        print(idx, len(outlier_images))
        image = outlier_images[idx]
        keypoints = outlier_keypoints[idx]

        plt.figure(figsize=(5,5))
        image = image if image.shape[-1] is 3 else image[..., 0]
        cmap = None if image.shape[-1] is 3 else 'gray'
        plt.imshow(image, cmap=cmap, interpolation='none')
        for idx, jdx in enumerate(data_generator.graph):
            if jdx > -1:
                plt.plot(
                    [keypoints[idx, 0], keypoints[jdx, 0]],
                    [keypoints[idx, 1], keypoints[jdx, 1]],
                    'r-'
                )
        plt.scatter(
            keypoints[:, 0],
            keypoints[:, 1],
            c=np.arange(data_generator.keypoints_shape[0]),
            s=50,
            cmap=plt.cm.hsv,
            zorder=3
        )
        plt.show()

    # # Merge outliers with the annotation set
    # Here we'll use a utility function `merge_new_images` to merge the outliers
    # with our existing annotation set. You can then go annotate them with
    # `deepposekit.annotate.Annotator`. Make sure to use the merged output file!
    merge_new_images(
        datapath=s_input_annot_fname,
        merged_datapath=s_out_annot_fname,
        images=outlier_images,
        keypoints=outlier_keypoints,
        # overwrite=True # This overwrites the merged dataset if it already exists
    )

    # Load the data with `DataGenerator` and check that the merged data are there
    merged_generator = DataGenerator(s_out_annot_fname, mode="unannotated")
    image, keypoints = merged_generator[0]

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

    # # Visualize the data as a video
    # This is an example of how to visualize the predicted posture data on the original video.

    data_generator = DataGenerator(s_input_annot_fname)
    predictions = predictions[...,:2]
    predictions *= 2
    resized_shape = (data_generator.image_shape[0]*2, data_generator.image_shape[1]*2)
    cmap = plt.cm.hsv(np.linspace(0, 1, data_generator.keypoints_shape[0]))[:, :3][:, ::-1] * 255

    oc_writer = VideoWriter(s_out_video_fname, (192*2,192*2), 'MP4V', 30.0)
    oc_reader = VideoReader(s_input_video_fname, batch_size=1)

    for frame, keypoints in zip(oc_reader, predictions):
        frame = frame[0]
        frame = frame.copy()
        frame = cv.resize(frame, resized_shape)
        for idx, node in enumerate(data_generator.graph):
            if node >= 0:
                pt1 = keypoints[idx]
                pt2 = keypoints[node]
                cv.line(frame, (pt1[0], pt1[1]), (pt2[0], pt2[1]), (0,0,255), 2, cv.LINE_AA)
        for idx, keypoint in enumerate(keypoints):
            keypoint = keypoint.astype(int)
            cv.circle(frame, (keypoint[0], keypoint[1]), 5, tuple(cmap[idx]), -1, lineType=cv.LINE_AA)

        oc_writer.write(frame)

    oc_writer.close()
    oc_reader.close()
#

