Development Roadmap
==========

IO module `deepposekit.io`
------
- [x] Abstract data IO with `deepposekit.io.BaseGenerator`
- [x] Support for custom data sets by subclassing `deepposekit.io.BaseGenerator`
- [x] Support for loading DeepLabCut formatted data `deepposekit.io.DLCDataGenerator`
- [x] Utility function for initializing a new image set for annotation `deepposekit.io.utils.initialize_image_set`
- [x] Utility function for merging a new image set to an existing dataset `deepposekit.io.utils.merge_new_images`
- [ ] Add methods for appending new images to `deepposekit.io.BaseGenerator` with 
`deepposekit.io.BaseGenerator.append_images()`
- [ ] Utility function for merging multiple arbitrary `deepposekit.io.BaseGenerator` with `deepposekit.io.utils.merge_data`
- [ ] Utility function for converting `deepposekit.io.DLCDataGenerator` data to `deepposekit.io.DataGenerator` data and vice-versa
- [ ] Support more DLC features within `deepposekit.io.DLCDataGenerator`. 
- [ ] Support passing multiple `deepposekit.io.BaseGenerator` for `deepposekit.io.TrainingGenerator`, but ensure all are compatible before training the model.

Annotation module `deepposekit.annotate`
------
- [ ] Add support for `deepposekit.annotate.Annotator` to edit DeepLabCut formatted data `deepposekit.io.DLCDataGenerator`. Ensure this does not destroy compatibility with DLC.
- [x] Remove extra step of initializing a skeleton and remove `deepposekit.annotate.Skeleton`, as this is confusing and not all that helpful.
- [ ] Abstract `deepposekit.annotate.gui.GUI` and `deepposekit.annotate.Annotator` to use new `deepposekit.io.BaseGenerator` with abstracted data IO
- [ ] Develop submodule `deepposekit.annotate.outliers` with tools for identifying outlier data for adding to data sets

Models modules `deepposekit.models`
------
- [x] Add `MobileNetV2` and `DenseNet` backbones to `deepposekit.models.DeepLabCut`
- [x] Add pretrained `DenseNet` frontend to `StackedDenseNet` model
- [ ] Support arbitrary image sizes (not just powers of 2) with `tf.keras.layers.ZeroPaddding2D` 
- [ ] Support dynamic image sizes with with automatic padding at inference. **Is this possible without reducing functionality?**

Examples and Documentation
------
- [ ] Improve and update docstrings across the package
- [x] Add example notebook for using custom data sets 
- [x] Add example notebook for using DeepLabCut formatted data
- [x] Add example for identifying outliers and appending new images to a training set
- [x] Add html documentation

Tests (once API has stabilized)
------
- [ ] Import all modules and submodules
- [ ] Download example data
- [ ] Run training for all models
- [ ] Save model
- [ ] Load model
- [ ] Resume training
- [ ] Predict on new data

Future
------
- [x] Put `deepposekit` on PyPI
- [x] Update to tf.keras (stand-alone keras will be deprecated)
- [x] Update to Tensorflow 2.0
- [ ] `deepposekit.visualize` module with functions for making videos and plotting data
- [ ] `deepposekit.pose3d` module? Does it make sense to support this, or just make the API abstract enough to let others use their own solution for 3D?
- [ ] `deepposekit.localize` module. Train models that localize individuals using confidence maps. Update and further abstract `deepposekit.annotate`, `deepposekit.models`, etc.
- [ ] `deepposekit.multiple` module. Add support for small groups of multiple individuals? Does it make sense to support this or focus on `deepposekit.localize`?
