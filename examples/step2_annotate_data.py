#!/usr/bin/env python3
# coding: utf-8

# # DeepPoseKit Step 2 - Annotate your data
#
# This is step 2 of the example scripts for using DeepPoseKit.
# This script shows you how to annotate your training data with
# user-defined key-points using the saved data from step 1.

# This script will read skeleton definition file (fly/skeleton.csv)
# and the file "example_annotation_set.h5" generated during step1 in the current directory
# NOTE: this script will modify and save the "example_annotation_set.h5" file.

import os


if __name__ == '__main__':
    from bootstrap import bootstrap_environment

    s_deep_pose_kit_data_dir = bootstrap_environment()
    s_input_skel_fname = os.path.join(s_deep_pose_kit_data_dir, "datasets", "fly", "skeleton.csv")
    s_input_annot_fname = "example_annotation_set.h5" # This file was generated during Step 1.

    from deepposekit import Annotator

    # # Annotate data
    # Annotations are saved automatically.
    # The skeleton in each frame will turn blue when the frame is fully annotated.
    # If there are no visible key-points, this means the frame hasn't been annotated,
    # so try clicking to position the key-point in the frame.

    app = Annotator(
        datapath=s_input_annot_fname,
        dataset='images',
        skeleton=s_input_skel_fname,
        shuffle_colors=False,
        text_scale=0.2
    )
    app.run()
#
