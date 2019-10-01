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

import numpy as np
import cv2

from deepposekit.annotate.gui.GUI import GUI
from deepposekit.annotate.utils import hotkeys as keys

__all__ = ["Skeleton"]


class Skeleton(GUI):
    """
    A GUI for initializing a skeleton for a new dataset.

    ------------------------------------------------------------
         Keys             |   Action
    ------------------------------------------------------------
    >    +,-              |   Rescale the image
    >    Left mouse       |   Move active keypoint
    >    W, A, S, D       |   Move active keypoint
    >    space            |   Changes W,A,S,D mode (swaps between 1px or 10px)
    >    J, L             |   Load previous or next image
    >    <, >             |   Jump 10 images backward or forward
    >    I, K or          |
         tab, shift+tab   |   Switch active keypoint
    >    R                |   Mark frame as unannotated, or "reset"
    >    F                |   Mark frame as annotated or "finished"
    >    Esc, Q           |   Quit the GUI
    ------------------------------------------------------------

    Parameters
    ----------
    image: str
        Filepath of the image to be labeled.

    skeleton: str
        Filepath of the .csv or .xlsx file that has indexed information
        on name of the keypoint (part, e.g. head), parent (the direct
        connecting part, e.g. neck connects to head, parent is head),
        and swap (swapping positions with a part when reflected over X)

        Consult example of such a file for more information

    scale: int/float, default 1
        Scaling factor for the GUI (e.g. used in zooming).

    text_scale: float
        Scaling factor for the GUI font.
        A text_scale of 1 works well for 1920x1080 (1080p) images

    shuffle_colors: bool, default = True
        Whether to shuffle the color order for drawing keypoints

    refresh: int, default 100
        Delay on receiving next keyboard input in milliseconds.

    Attributes
    ----------
    window_name: str
        Name of the Annotation window when running program.
        Set to be 'Annotation' unless otherwise changed.

    n_images: int
        Number of images in question (1 in this case).

    key: int
        The key that is pressed on the keyboard.

    image_idx: int
        Index of a specific image in the .h5 file.

    save: method
        Output method is set to be to_csv

    Examples
    --------
    >>> from deepposekit import Skeleton
    >>> app = Skeleton('input_image.png', 'skeleton.csv')
    >>> app.run()
    >>>
    >>> app.save('skeleton_initialized.csv') # save the labels in skeleton.csv file

    Note: must use app.save('file.csv') to save! Unlike the
    Annotator, will not automatically save until that line runs.

    """

    def __init__(
        self,
        image,
        skeleton,
        scale=1,
        text_scale=0.15,
        shuffle_colors=True,
        refresh=100,
    ):

        if isinstance(image, str):
            self.image = cv2.imread(image)
        elif isinstance(image, np.ndarray):
            self.image = image

        super(GUI, self).__init__()
        self.image_idx = 0
        self.n_images = 1
        self.window_name = "Skeleton Creator"
        self.shuffle_colors = shuffle_colors
        self._init_skeleton(skeleton)
        self._init_gui(scale, text_scale, shuffle_colors, refresh)
        self.save = self.skeleton.to_csv

    def _hotkeys(self):
        """ Activates all key bindings.
        
        Enables all the key functionalities described at the
        start of the file.

        """
        if self.key != keys.NONE:
            self._wasd()
            self._move_idx()
            self._zoom()
            self._update_canvas()
