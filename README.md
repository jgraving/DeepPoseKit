<p align="center">
<img src="https://github.com/jgraving/DeepPoseKit/blob/master/assets/deepposekit_logo.svg" height="512px">
</p>


You have just found DeepPoseKit.
---------------------------------
<p align="center">
<img src="https://github.com/jgraving/jgraving.github.io/blob/master/files/images/Figure1video1.gif" max-height:256px>
</p>

DeepPoseKit is a software toolkit with a high-level API for 2D pose estimation of user-defined keypoints using deep learning—written in Python and built using [Tensorflow](https://github.com/tensorflow/tensorflow) and [Keras](https://www.tensorflow.org/guide/keras). Use DeepPoseKit if you need:

- tools for annotating images or video frames with user-defined keypoints
- a straightforward but flexible data augmentation pipeline using the [imgaug package](https://github.com/aleju/imgaug)
- a Keras-based interface for initializing, training, and evaluating pose estimation models
- easy-to-use methods for saving and loading models and making predictions on new data

DeepPoseKit is designed with a focus on *usability* and *extensibility*, as being able to go from idea to result with the least possible delay is key to doing good research.

DeepPoseKit is currently limited to *individual pose esimation*. If individuals can be easily distinguished visually (i.e., they have differently colored bodies or are marked in some way), then multiple individuals can simply be labeled with separate keypoints (head1, tail1, head2, tail2, etc.). Otherwise DeepPoseKit can be extended to multiple individuals by first localizing, tracking, and cropping individuals with additional software such as [idtracker.ai](https://idtracker.ai/), [pinpoint](https://github.com/jgraving/pinpoint), or [Tracktor](https://github.com/vivekhsridhar/tracktor).

Localization (without tracking) can also be achieved with deep learning software like [keras-retinanet](https://github.com/fizyr/keras-retinanet), the [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection), or [MatterPort's Mask R-CNN](https://github.com/matterport/Mask_RCNN).

[Check out our preprint](https://doi.org/10.1101/620245) to find out more.

**NOTE:** This software is still in early-release development. *Expect some adventures.*

<p align="center">
<img src="https://github.com/jgraving/jgraving.github.io/blob/master/files/images/zebra.gif" max-height:256px>
<img src="https://github.com/jgraving/jgraving.github.io/blob/master/files/images/locust.gif" max-height:256px>
</p>

How to use DeepPoseKit
---------------------------------
DeepPoseKit is designed for easy use. For example, training and saving a model requires only a few lines of code:
```python
from deepposekit.io import DataGenerator, TrainingGenerator
from deepposekit.models import StackedDenseNet

data_generator = DataGenerator('/path/to/annotation_data.h5')
train_generator = TrainingGenerator(data_generator)
model = StackedDenseNet(train_generator)
model.fit(batch_size=16, n_workers=8)
model.save('/path/to/saved_model.h5')
```
Loading a trained model and running predictions on new data is also straightforward. For example, running predictions on a new video:
```python
from deepposekit.models import load_model
from deepposekit.io import VideoReader

model = load_model('/path/to/saved_model.h5')
reader = VideoReader('/path/to/video.mp4')
predictions = model.predict(reader)
```

Using DeepPoseKit is a 4-step process:
- **1.** [Create an annotation set](https://github.com/jgraving/DeepPoseKit/blob/master/examples/step1_create_annotation_set.ipynb) <a href="https://colab.research.google.com/github/jgraving/deepposekit/blob/master/examples/step1_create_annotation_set.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
- **2.** [Annotate your data](https://github.com/jgraving/DeepPoseKit/blob/master/examples/step2_annotate_data.ipynb) with our built-in GUI (no Colab support)
- **3.** [Select and train a model](https://github.com/jgraving/DeepPoseKit/blob/master/examples/step3_train_model.ipynb) including our `StackedDenseNet` model and the `DeepLabCut` model. <a href="https://colab.research.google.com/github/jgraving/deepposekit/blob/master/examples/step3_train_model.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
- **4.** Use the trained model to:
	- a) [Initialize keypoints for unannotated data in the annotation set](https://github.com/jgraving/DeepPoseKit/blob/master/examples/step4a_initialize_annotations.ipynb) for faster annotations with *active learning*. <a href="https://colab.research.google.com/github/jgraving/deepposekit/blob/master/examples/step4a_initialize_annotations.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

	- b) [Predict on new data and refine the training set](https://github.com/jgraving/DeepPoseKit/blob/master/examples/step4b_predict_new_data.ipynb) to improve performance. <a href="https://colab.research.google.com/github/jgraving/deepposekit/blob/master/examples/step4b_predict_new_data.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

[See our example notebooks](https://github.com/jgraving/deepposekit/blob/master/examples/) and [read our preprint](https://doi.org/10.1101/620245) for more details.

"I already have annotated data"
---------------------------------
DeepPoseKit is designed to be extensible, so loading data in other formats is possible.

If you have data from DeepLabCut (http://deeplabcut.org), try [our (experimental) example notebook ](https://github.com/jgraving/DeepPoseKit/blob/master/examples/deeplacut_data_example.ipynb) for loading data in this format. <a href="https://colab.research.google.com/github/jgraving/deepposekit/blob/master/examples/deeplabcut_data_example.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

Have data in another format? You can write your own custom generator to load it.
Check out the [example for writing custom data generators](https://github.com/jgraving/DeepPoseKit/blob/master/examples/custom_data_generator.ipynb). <a href="https://colab.research.google.com/github/jgraving/deepposekit/blob/master/examples/custom_data_generator.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


Installation
---------------------------------

DeepPoseKit requires [Tensorflow](https://github.com/tensorflow/tensorflow) for training and using pose estimation models. [Tensorflow](https://github.com/tensorflow/tensorflow) should be manually installed, along with dependencies such as CUDA and cuDNN, before installing DeepPoseKit:

- [Tensorflow Installation Instructions](https://www.tensorflow.org/install)
- **Note**: [Tensorflow 2.0](https://www.tensorflow.org/beta) is not yet supported, but an update is in the works.

DeepPoseKit has only been tested on Ubuntu 18.04, which is the recommended system for using the toolkit. 

Install the latest stable release with pip:
```bash
pip install --update deepposekit
```

Install the latest development version with pip:
```bash
pip install --update git+https://www.github.com/jgraving/deepposekit.git
```

You can download example datasets from our [DeepPoseKit Data](https://github.com/jgraving/deepposekit-data) repository:
```bash
git clone https://www.github.com/jgraving/deepposekit-data
```

### Installing with Anaconda
Anaconda cannot install the [imgaug package](https://github.com/aleju/imgaug) using pip, therefore as a temporary workaround we recommend installing imgaug manually:
```bash
conda config --add channels conda-forge
conda install imgaug -c conda-forge
```
We also recommend installing DeepPoseKit from within Python rather than using the command line, either from within Jupyter or another IDE, to ensure it is installed in the correct working environment:
```python
import sys
!{sys.executable} -m pip install --update deepposekit
```
Contributors and Development  
---------------------------------   
DeepPoseKit was developed by [Jake Graving](https://github.com/jgraving) and [Daniel Chae](https://github.com/dchaebae), and is still being actively developed. .

We welcome community involvement and public contributions to the toolkit. If you wish to contribute, please [fork the repository](https://help.github.com/en/articles/fork-a-repo) to make your modifications and [submit a pull request](https://help.github.com/en/articles/creating-a-pull-request-from-a-fork).

If you'd like to get involved with developing DeepPoseKit, get in touch (jgraving@gmail.com) and check out [our development roadmap](https://github.com/jgraving/DeepPoseKit/blob/master/DEVELOPMENT.md) to see future plans for the package.  

Please submit bugs or feature requests to the [GitHub issue tracker](https://github.com/jgraving/deepposekit/issues/new). Please limit reported issues to the DeepPoseKit codebase and provide as much detail as you can with a minimal working example if possible.

If you experience problems with [Tensorflow](https://github.com/tensorflow/tensorflow), such as installing CUDA or cuDNN dependencies, then please direct issues to those development teams.

License
---------------------------------
Released under a Apache 2.0 License. See [LICENSE](https://github.com/jgraving/deepposekit/blob/master/LICENSE) for details.

References
---------------------------------
If you use DeepPoseKit for your research please cite [our preprint](https://doi.org/10.1101/620245):

    @article{graving2019deepposekit,
             title={DeepPoseKit, a software toolkit for fast and robust pose estimation using deep learning},
             author={Graving, Jacob M and Chae, Daniel and Naik, Hemal and Li, Liang and Koger, Benjamin and Costelloe, Blair R and Couzin, Iain D},
             journal={bioRxiv},
             pages={620245},
             year={2019},
             publisher={Cold Spring Harbor Laboratory}
             }


News
---------------------------------

- **September 2019:** v0.3.0 is released. See [the release notes](https://github.com/jgraving/DeepPoseKit/releases/tag/v0.3.0).
- **April 2019:** The DeepPoseKit preprint is on biorxiv (http://preprint.deepposekit.org)
