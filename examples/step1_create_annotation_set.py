#!/usr/bin/env python3
# coding: utf-8

# # DeepPoseKit Step 1 - Create an annotation set
#
# This is step 1 of the example scripts for using DeepPoseKit.
# This script shows you how to load and sample images from a video,
# define a key-point skeleton, and save the data to a file for labellings with key-points.

# This script will read raw video (fly/video.avi) and skeleton definition file (fly/skeleton.csv)
# and generate file called "example_annotation_set.h5" in the current directory.

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sample_frames_randomly(oc_reader):
    l_frames = []
    for i_idx, _ in enumerate(oc_reader):
        batch = oc_reader[i_idx]
        random_sample = batch[np.random.choice(batch.shape[0], 10, replace=False)]
        l_frames.append(random_sample)
    na_frames = np.concatenate(l_frames)
    return na_frames
#


if __name__ == '__main__':
    from bootstrap import bootstrap_environment

    s_deep_pose_kit_data_dir = bootstrap_environment(b_verbose=True)
    s_input_video_fname = os.path.join(s_deep_pose_kit_data_dir, "datasets", "fly", "video.avi")
    s_input_skel_fname = os.path.join(s_deep_pose_kit_data_dir, "datasets", "fly", "skeleton.csv")
    # s_out_annot_fname = os.path.join(s_deep_pose_kit_data_dir, "datasets", "fly", "example_annotation_set.h5")
    s_out_annot_fname = "example_annotation_set.h5" # local output file for testing

    from deepposekit.io import VideoReader, DataGenerator, initialize_dataset
    from deepposekit.annotate import KMeansSampler

    # # A note on image resolutions
    # Currently DeepPoseKit only supports image resolutions that can be repeatedly divided by 2.

    # # Open a video
    # The `VideoReader` class allows you to load in single video frames
    # or batches of frames from nearly any video format.

    oc_reader = VideoReader(s_input_video_fname, batch_size=100, gray=True)

    # # Sample video frames
    # This loads batches of 100 frames from the video, and then randomly samples frames
    # from the batches to hold them in memory. You can use any method for sampling frames.

    na_sampled_frames = sample_frames_randomly(oc_reader)
    oc_reader.close()
    print("na_sampled_frames.shape:", na_sampled_frames.shape)

    # # Apply k-means to reduce correlation
    # This applies the k-means algorithm to the images using `KMeansSampler` to even out sampling
    # across the distribution of images and reduce correlation within the annotation set.

    kmeans = KMeansSampler(n_clusters=10, max_iter=1000, n_init=10, batch_size=100, verbose=True)
    kmeans.fit(na_sampled_frames)
    kmeans.plot_centers(n_rows=2)
    plt.show()

    kmeans_sampled_frames, kmeans_cluster_labels = kmeans.sample_data(na_sampled_frames, n_samples_per_label=10)

    # # Define a keypoints skeleton file
    # You must create a .xlsx or .csv file with keypoint names, parent relationships,
    # and swapping relationships for bilaterally symmetric parts (only relevant if using
    # flipping augmentations). If you leave out the `parent` and `swap` columns,
    # then these will simply not be used for annotating data and training the model.
    # See example skeleton.csv files for more details

    oc_skeleton = pd.read_csv(s_input_skel_fname)
    print("\nINFO: loaded skeleton definition:\n", oc_skeleton)

    # # Initialize a new data set for annotations
    # You can use any method for sampling images to create a numpy array with the shape
    # (n_images, height, width, channels) and then initialize an annotation set.

    # NOTE: the 'datapath' kwarg is the OUTPUT file name here
    initialize_dataset(
        images=kmeans_sampled_frames,
        datapath=s_out_annot_fname,
        skeleton=s_input_skel_fname,
        # overwrite=True # This overwrites the existing datapath
    )

    # # Create a data generator
    # This creates a `DataGenerator` for loading annotated data.
    # Indexing the generator returns an image-keypoints pair,
    # which you can then visualize. Right now all the key-points
    # are set to zero, because they haven't been annotated.

    data_generator = DataGenerator(s_out_annot_fname, mode="full")
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

