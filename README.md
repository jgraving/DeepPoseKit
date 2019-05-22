DeepPoseKit: an API for pose estimation
============

<p align="center">
<img src="https://github.com/jgraving/jgraving.github.io/blob/master/files/images/Figure1video1.gif" max-height:256px>
</p>


You have just found DeepPoseKit.
------------
DeepPoseKit is a high-level API for 2D pose estimation with deep learning written in Python and built using [Keras](https://github.com/keras-team/keras) and [Tensorflow](https://github.com/tensorflow/tensorflow). Use DeepPoseKit if you need:

- tools for annotating images or video frames with user-defined keypoints
- a straightforward but flexible data augmentation pipeline
- a Keras-based interface for initializing, training, and evaluating pose estimation models
- easy-to-use methods for saving and loading models and making predictions on new data

DeepPoseKit is designed with a focus on *usability* and *extensibility*, as being able to go from idea to result with the least possible delay is key to doing good research. [Check out our preprint](https://www.biorxiv.org/content/10.1101/620245v3) to find out more.

<p align="center">
<img src="https://github.com/jgraving/jgraving.github.io/blob/master/files/images/zebra.gif" max-height:256px>
<img src="https://github.com/jgraving/jgraving.github.io/blob/master/files/images/locust.gif" max-height:256px>
</p>

How to use DeepPoseKit
------------
DeepPoseKit is designed for easy use. For example, training and saving a model requires only a few lines of code:
```python
train_generator = TrainingGenerator('/path/to/data.h5')
model = StackedDenseNet(train_generator)
model.compile('adam', 'mse')
model.fit(batch_size=16, n_workers=8)
model.save('/path/to/model.h5')
```
Loading a trained model and running predictions on new data is also straightforward:
```python
model = load_model('/path/to/model.h5')
new_data = load_new_data('/path/to/new/data.h5')
predictions = model.predict(new_data)
```
[See our example notebooks](https://github.com/jgraving/deepposekit/blob/master/examples/) for more details on how to use DeepPoseKit.


Installation
------------

DeepPoseKit requires [Tensorflow](https://github.com/tensorflow/tensorflow) and [Keras](https://github.com/keras-team/keras) for training and using pose estimation models. These should be manually installed, along with dependencies such as CUDA and cuDNN, before installing DeepPoseKit:

- [Keras Installation Instructions](https://github.com/keras-team/keras#Installation)
- [Tensorflow Installation Instructions](https://www.tensorflow.org/install)
- Note: [Tensorflow 2.0](https://www.tensorflow.org/alpha) is not yet supported.

DeepPoseKit has only been tested on Ubuntu 18.04, which is the recommended system for using the toolkit. 

Install the development version with pip:
```bash
pip install git+https://www.github.com/jgraving/deepposekit.git
```

To use the annotation toolkit you must install this separately (see [DeepPoseKit Annotator](https://www.github.com/jgraving/deepposekit-annotator) for details):
```bash
pip install git+https://www.github.com/jgraving/deepposekit-annotator.git
```

You can download example datasets for DeepPoseKit from our [DeepPoseKit Data](https://github.com/jgraving/deepposekit-data) repository:
```bash
git clone https://www.github.com/jgraving/deepposekit-data
```


Citation
---------
If you use DeepPoseKit for your research please cite our preprint:

    @article{graving2019fast,
             title={Fast and robust animal pose estimation},
             author={Graving, Jacob M and Chae, Daniel and Naik, Hemal and Li, Liang and Koger, Benjamin and Costelloe, Blair R and Couzin, Iain D},
             journal={bioRxiv},
             pages={620245},
             year={2019},
             publisher={Cold Spring Harbor Laboratory}
             }

License
------------
Released under a Apache 2.0 License. See [LICENSE](https://github.com/jgraving/deepposekit/blob/master/LICENSE.md) for details.

Development
-------------
Please submit bugs or feature requests to the [GitHub issue tracker](https://github.com/jgraving/deepposekit/issues/new). Please limit reported issues to the DeepPoseKit codebase and provide as much detail as you can with a minimal working example if possible. 

If you experience problems with [Tensorflow](https://github.com/tensorflow/tensorflow) or [Keras](https://github.com/keras-team/keras), such as installing CUDA or cuDNN dependencies, then please direct issues to those development teams.

Contributors
------------
DeepPoseKit was developed by [Jake Graving](https://github.com/jgraving) and [Daniel Chae](https://github.com/dchaebae). We welcome public contributions to the toolkit. If you wish to contribute, please [fork the repository](https://help.github.com/en/articles/fork-a-repo) to make your modifications and [submit a pull request](https://help.github.com/en/articles/creating-a-pull-request-from-a-fork).