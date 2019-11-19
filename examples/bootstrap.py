#!/usr/bin/env python3
# coding: utf-8


import os
import sys


def bootstrap_environment(b_verbose=False):
    # Try to find the place into which https://github.com/jgraving/DeepPoseKit.git was cloned
    if 'DEEP_POS_KIT_REPO' in os.environ:
        if not os.path.isdir(os.environ['DEEP_POS_KIT_REPO']):
            raise ValueError("DEEP_POS_KIT_REPO: the value is not an accessible directory: %s" % os.environ['DEEP_POS_KIT_REPO'])
        else:
            sys.path.append(os.environ['DEEP_POS_KIT_REPO'])
            if b_verbose:
                print("INFO: DEEP_POS_KIT_REPO set to %s" % os.environ['DEEP_POS_KIT_REPO'])
    else:
        if 'HOME' not in os.environ:
            raise ValueError("Unexpected environment")

    # Try to find the place into which https://github.com/aleju/imgaug.git was cloned
    if 'IMGAUG_REPO_DIR' in os.environ:
        if not os.path.isdir(os.environ['IMGAUG_REPO_DIR']):
            raise ValueError("IMGAUG_REPO_DIR: the value is not an accessible directory: %s" % os.environ['IMGAUG_REPO_DIR'])
        else:
            s_imgaug_build_dir = os.path.join(os.environ['IMGAUG_REPO_DIR'], 'build', 'lib')
            if not os.path.isdir(s_imgaug_build_dir):
                raise ValueError("The imgaug repository was found but does not seems to be build. Run 'python setup.py build' there")
            sys.path.append(s_imgaug_build_dir)
            if b_verbose:
                print("INFO: IMGAUG_REPO_DIR set to %s" % os.environ['IMGAUG_REPO_DIR'])
    else:
        if 'HOME' not in os.environ:
            raise ValueError("Unexpected environment")

    if 'DEEP_POS_KIT_DATA_REPO' in os.environ:
        if not os.path.isdir(os.environ['DEEP_POS_KIT_DATA_REPO']):
            raise ValueError("DEEP_POS_KIT_DATA_REPO: the value is not an accessible directory: %s" % os.environ['DEEP_POS_KIT_DATA_REPO'])
        s_deep_pose_kit_data_dir = os.environ['DEEP_POS_KIT_DATA_REPO']
        if b_verbose:
            print("INFO: DEEP_POS_KIT_DATA_REPO set to %s" % os.environ['DEEP_POS_KIT_DATA_REPO'])
    else:
        if 'HOME' not in os.environ:
            raise ValueError("Unexpected environment")
        s_deep_pose_kit_data_dir = os.path.join(os.environ['HOME'], 'DeepPoseKit-Data')
        if b_verbose:
            print("INFO: DeepPoseKit-Data directory found at: %s" % s_deep_pose_kit_data_dir)
        if not os.path.isdir(s_deep_pose_kit_data_dir):
            raise ValueError("Not a directory: %s" % s_deep_pose_kit_data_dir)

    return s_deep_pose_kit_data_dir
#
