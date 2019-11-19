#!/usr/bin/env python3
# coding: utf-8

# # DeepPoseKit Step 3 - Train a model
#
# This is step 3 of the example scripts for using DeepPoseKit.
# This script shows you how to use your annotated data to train a deep learning
# model applying data augmentation and using callbacks for logging the training
# process and saving the best model during training.

# This script will read fully annotated data file (fly/annotation_data_release.h5)
# and generate file called "best_model_densenet.h5" in the current directory.
# The output file contain best trained model.

import os
import time

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping


def create_data_augmenter(data_generator):
    import imgaug as ia
    import imgaug.augmenters as iaa

    augmenter = []

    augmenter.append(FlipAxis(data_generator, axis=0))  # flip image up-down
    augmenter.append(FlipAxis(data_generator, axis=1))  # flip image left-right 

    sometimes = []
    sometimes.append(iaa.Affine(scale={"x": (0.95, 1.05), "y": (0.95, 1.05)},
                                translate_percent={'x': (-0.05, 0.05), 'y': (-0.05, 0.05)},
                                shear=(-8, 8),
                                order=ia.ALL,
                                cval=ia.ALL,
                                mode=ia.ALL)
                     )
    sometimes.append(iaa.Affine(scale=(0.8, 1.2),
                                mode=ia.ALL,
                                order=ia.ALL,
                                cval=ia.ALL)
                     )
    augmenter.append(iaa.Sometimes(0.75, sometimes))
    augmenter.append(iaa.Affine(rotate=(-180, 180),
                                mode=ia.ALL,
                                order=ia.ALL,
                                cval=ia.ALL)
                     )
    augmenter = iaa.Sequential(augmenter)
    return augmenter
#


if __name__ == '__main__':
    from bootstrap import bootstrap_environment

    s_deep_pose_kit_data_dir = bootstrap_environment()
    s_input_annot_result_fname = os.path.join(s_deep_pose_kit_data_dir, "datasets", "fly", "annotation_data_release.h5")
    s_out_best_model_fname = "best_model_densenet.h5"

    from deepposekit.augment import FlipAxis
    from deepposekit.io import TrainingGenerator, DataGenerator
    from deepposekit.models import DeepLabCut, StackedDenseNet, StackedHourglass, LEAP
    from deepposekit.callbacks import Logger, ModelCheckpoint

    # # Create a `DataGenerator`
    # This creates a `DataGenerator` for loading annotated data.
    # Indexing the generator, e.g. `data_generator[0]` returns
    # an image-keypoints pair, which you can then visualize.

    data_generator = DataGenerator(s_input_annot_result_fname)

    # # Create an augmentation pipeline
    # DeepPoseKit works with augmenters from the imgaug: https://github.com/aleju/imgaug
    # This is a short example using spatial augmentations with axis flipping and affine transforms
    # See https://github.com/aleju/imgaug for more documentation on augmenters.
    # `deepposekit.augment.FlipAxis` takes the `DataGenerator` as an argument to get the key-point
    # swapping information defined in the annotation set. When the images are mirrored key-points
    # for left and right sides are swapped to avoid "confusing" the model during training.

    oc_augmenter = create_data_augmenter(data_generator)

    # Load an image-keypoints pair, apply augmentation, visualize it.

    image, keypoints = data_generator[0]
    image, keypoints = oc_augmenter(images=image, keypoints=keypoints)

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

    # # Create a `TrainingGenerator`
    # This creates a `TrainingGenerator` from the `DataGenerator` for training the model with annotated data.
    # The `TrainingGenerator` uses the `DataGenerator` to load image-keypoints pairs and then applies
    # the augmentation and draws the confidence maps for training the model.

    # If you're using `StackedDenseNet`, `StackedHourglass`, or `DeepLabCut`
    # you should set `downsample_factor=2` for 1/4x outputs or `downsample_factor=3`
    # for 1/8x outputs (1/8x is faster). Here it is set to `downsample_factor=3` to maximize speed.
    # If you are using `LEAP` you should set the `downsample_factor=0` for 1x outputs.

    # The `validation_split` argument defines how many training examples to use for validation
    # during training. If your dataset is small (such as initial annotations for active learning),
    # you can set this to `validation_split=0`, which will just use the training set for model fitting.
    # However, when using callbacks, make sure to set `monitor="loss"` instead of `monitor="val_loss"`.

    # Visualizing the outputs in the next section also works best with `downsample_factor=0`.

    train_generator = TrainingGenerator(
        generator=data_generator,
        downsample_factor=3,
        augmenter=oc_augmenter,
        sigma=5,
        validation_split=0.1, 
        use_graph=True,
        random_seed=1,
        graph_scale=1
    )
    train_generator.get_config()

    # # Check the `TrainingGenerator` output
    # This plots the training data output from the `TrainingGenerator` to ensure
    # that the augmentation is working and the confidence maps look good.

    n_keypoints = data_generator.keypoints_shape[0]
    batch = train_generator(batch_size=1, validation=False)[0]
    inputs = batch[0]
    outputs = batch[1]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10,10))
    ax1.set_title('image')
    ax1.imshow(inputs[0,...,0], cmap='gray', vmin=0, vmax=255)

    ax2.set_title('posture graph')
    ax2.imshow(outputs[0,...,n_keypoints:-1].max(-1))

    ax3.set_title('keypoints confidence')
    ax3.imshow(outputs[0,...,:n_keypoints].max(-1))

    ax4.set_title('posture graph and keypoints confidence')
    ax4.imshow(outputs[0,...,-1], vmin=0)
    plt.show()

    train_generator.on_epoch_end()

    # # Define a model
    # Here you can define a model to train with your data. You can use our
    # `StackedDenseNet` model, `StackedHourglass` model, `DeepLabCut` model, or the `LEAP` model.
    # The default settings for each model should work well for most datasets,
    # but you can customize the model architecture. The `DeepLabCut` model
    # has multiple pretrained (on ImageNet) backbones available for using
    # transfer learning, including the original ResNet50 (He et al. 2015)
    # as well as the faster MobileNetV2 (Sandler et al. 2018; see also Mathis et al. 2019)
    # and DenseNet121 (Huang et al. 2017). We'll select `StackedDenseNet` and set `n_stacks=2`
    # for 2 hourglasses, with `growth_rate=32` (32 filters per convolution).
    # Adjust the `growth_rate` and/or `n_stacks` to change model performance (and speed).
    # You can also set `pretrained=True` to use transfer learning with `StackedDenseNet`,
    # which uses a DenseNet121 pretrained on ImageNet to encode the images.

    # The pre-trained model will be downloaded
    # during this call (only once) and saved into:
    # %USERPROFILE%\\.keras\\models (on Windows) or
    # $HOME/.keras/models (*nix)

    model = StackedDenseNet(train_generator, n_stacks=2, growth_rate=32, pretrained=True)
    # model = DeepLabCut(train_generator, backbone="resnet50")
    # model = DeepLabCut(train_generator, backbone="mobilenetv2", alpha=0.35) # Increase alpha to improve accuracy
    # model = DeepLabCut(train_generator, backbone="densenet121")
    # model = LEAP(train_generator)
    # model = StackedHourglass(train_generator)

    model.get_config()

    # # Test the prediction speed
    # This generates a random set of input images for the model to test
    # how fast the model can predict keypoint locations.

    data_size = (10000,) + data_generator.image_shape
    x = np.random.randint(0, 255, data_size, dtype="uint8")
    y = model.predict(x[:100], batch_size=100) # make sure the model is in GPU memory
    t0 = time.time()
    y = model.predict(x, batch_size=100, verbose=1)
    t1 = time.time()
    print("Prediction speed (FPS):", x.shape[0] / (t1 - t0))

    # # Define callbacks to enhance model training
    # Here you can define callbacks to pass to the model for use during training.
    # You can use any callbacks available in `deepposekit.callbacks` or `tensorflow.keras.callbacks`

    # Remember, if you set `validation_split=0` for your `TrainingGenerator`,
    # which will just use the training set for model fitting,
    # make sure to set `monitor="loss"` instead of `monitor="val_loss"`.

    # `Logger` evaluates the validation set (or training set if `validation_split=0` in the `TrainingGenerator`)
    # at the end of each epoch and saves the evaluation data to a HDF5 log file (if `filepath` is set).

    oc_logger = Logger(
        validation_batch_size=10,
        # filepath saves the logger data to a .h5 file
        # filepath=HOME + "/deepposekit-data/datasets/fly/log_densenet.h5"
    )

    # `ReduceLROnPlateau` automatically reduces the learning rate of the optimizer
    # when the validation loss stops improving. This helps the model to reach
    # a better optimum at the end of training.

    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, verbose=1, patience=20)

    # `ModelCheckpoint` automatically saves the model when the validation loss improves
    # at the end of each epoch. This allows you to automatically save the best performing
    # model during training, without having to evaluate the performance manually.

    model_checkpoint = ModelCheckpoint(
        s_out_best_model_fname,
        monitor="val_loss",
        # monitor="loss" # use if validation_split=0
        verbose=1,
        save_best_only=True,
    )

    # `EarlyStopping` automatically stops the training session when the validation loss stops
    # improving for a set number of epochs, which is set with the `patience` argument.
    # This allows you to save time when training your model if there's not more improvement.

    early_stop = EarlyStopping(
        monitor="val_loss",
        # monitor="loss" # use if validation_split=0
        min_delta=0.001,
        patience=100,
        verbose=1
    )

    # Create a list of callbacks to pass to the model
    l_callbacks = [early_stop, reduce_lr, model_checkpoint, oc_logger]

    # # Fit the model
    # This fits the model for a set number of epochs with small batches of data.
    # If you have a small dataset initially you can set `batch_size` to a small value
    # and manually set `steps_per_epoch` to some large value, e.g. 500,
    # to increase the number of batches per epoch, otherwise this is automatically
    # determined by the size of the dataset.
    # The number of `epochs` is set to `epochs=200` for demonstration purposes.
    # **Increase the number of epochs to train the model longer, for example `epochs=1000`**.
    # The `EarlyStopping` callback will then automatically end training if there is no improvement.

    model.fit(
        batch_size=16,
        validation_batch_size=10,
        callbacks=l_callbacks,
        #epochs=1000, # Increase the number of epochs to train the model longer
        epochs=200,
        n_workers=8,
        steps_per_epoch=None,
    )
#


