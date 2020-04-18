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

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from deepposekit.annotate.utils import hotkeys as keys


def _mouse_click(event, x, y, flags, param):
    """ Handles mouse click by annotating at the point and updating canvas

    Parameters
    ----------
    event: int
        OpenCV mouse event

    x: int
        x-coordinate of the mouse on click

    y: int
        y-coordinate of the mouse on click

    flags: int
        Flags for OpenCV

    param:
        Pass `self` for callback
    """

    self = param
    if event is cv2.EVENT_LBUTTONDOWN:
        self.point = np.array([x, y])
        self.skeleton.loc[self.idx, ["x", "y"]] = self.point / self.scale
        self._set_annotated()
        self._update_canvas()
    # elif event is cv2.EVENT_LBUTTONDOWN:
    #    self.point = np.array([x, y])
    #    keypoints = self.skeleton.loc[:, ['x', 'y']].values
    #    distances = (keypoints - self.point[None])**2
    #    self.idx = np.argmin(distances.sum(1))
    #    self.skeleton.loc[self.idx, ['x', 'y']] = self.point / self.scale
    #    self._set_annotated()
    #    self._update_canvas()


class GUI:
    def __init__(self):
        """ A GUI for annotating or marking up image(s).

        The GUI class works with a subclass to run a program that could be
        used to annotate or markup an image or a series of images.
        In order to extend the GUI class to make a subclass, a few things
        must be defined in the subclass.
        ------------------------------------------------------------------------
            Method      |   
        ------------------------------------------------------------------------
        >   _hotkeys()  |   Activates all the hotkey bindings
        ------------------------------------------------------------------------
            Attribute   |   
        ------------------------------------------------------------------------
        >   window_name |   Name of application window
        >   image_idx   |   Index of active image, important for multiple images
        >   n_images    |   Number of total images
        ------------------------------------------------------------------------

        See the Annotator.py or Skeleton.py for an example of how this is done

        Attributes
        ----------
        scale: float
            Scaling factor for the GUI (e.g. used in zooming).

        text_scale: float
            Scaling factor for the GUI font.
            A text_scale of 1 is good for 1920x1080 (1080p) images

        refresh: int
            Delay on receiving next keyboard input in milliseconds.

        point: numpy.ndarray
            The coordinates of the mouse on the GUI.

        image: numpy.ndarray
            One image accessed using image_idx.

        canvas: numpy.ndarray
            Canvas for the GUI itself.

        skeleton: pandas.DataFrame
            Store information from the skeleton data input.

        idx: int
            Index of the keypoint in question.

        keypoint_idx: numpy.ndarray
            Keeps track of the keypoints array.

        n_keypoints: int
            Total number of keypoints in an image.

        text_locs: list
            List of text locations.

        key: int
            The key that is pressed on the keyboard.

        """
        pass

    def _init_gui(self, scale, text_scale, shuffle_colors=True, refresh=100):
        """ Initializes the GUI

        Takes in the scaling factor and keyboard input delay. In addition,
        this function creates the window, names it, and updates the
        canvas and text.

        Parameters
        ----------
        scale: float
            Scaling factor for the GUI (e.g. used in zooming).
        text_scale: float
            Scaling factor for the GUI font.
            A text_scale of 1 is good for 1920x1080 (1080p) images
        shuffle_colors: bool
            Whether to shuffle the color order for keypoint drawing
        refresh: int
            Delay on receiving next keyboard input in milliseconds.
        
        """
        self.scale = float(scale)
        self.text_scale = float(text_scale)
        self.refresh = refresh
        self.point = np.array([-1, -1])
        self.shuffle_colors = shuffle_colors

        if max(self.image.shape) * self.scale < 512:
            self.scale = 512.0 / max(self.image.shape)
        if max(self.image.shape) * self.scale > 1024:
            self.scale = 1024.0 / max(self.image.shape)
        self.wasd_mode = True
        self.move_len = 1.0

        cv2.namedWindow(self.window_name)
        cv2.moveWindow(self.window_name, 0, 0)
        self._update_canvas_size()
        self._update_text_locs()
        self._update_canvas()

    def _check_grayscale(self):
        """ Convert image to RGB if it is not in RGB

        An image can be checked if it is in color based on dimensions.
        If the image shape is less than 3, then it does not have 
        color channels. If there is only 1 color channel, it is in
        grayscale.

        """
        if len(self.image.shape) == 2 or self.image.shape[-1] == 1:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)

    def _set_annotated(self):
        """ Set the specific keypoint as annotated.

        """
        self.skeleton.loc[self.idx, "annotated"] = True

    def _init_skeleton(self, skeleton):
        """ Initialize the skeleton from input data.

        Takes in either a .csv or .xlsx file and makes a DataFrame.

        Parameters
        ----------
        skeleton: pandas.DataFrame
            Filepath of the .csv or .xlsx file that has indexed information
            on name of the keypoint (part, e.g. head), parent (the direct
            connecting part, e.g. neck connects to head, parent is head),
            and swap (swapping positions with a part when reflected over X).

            Consult example of such a file for more information.
            
        """

        if isinstance(skeleton, str):
            if skeleton.endswith(".csv"):
                skeleton = pd.read_csv(skeleton)
            elif skeleton.endswith(".xlsx"):
                skeleton = pd.read_excel(skeleton)
            else:
                raise ValueError("skeleton must be .csv or .xlsx file")
        else:
            raise TypeError("skeleton must be type `str`")

        if "name" not in skeleton.columns:
            raise KeyError("skeleton file must contain a `name` column")
        elif "parent" not in skeleton.columns:
            raise KeyError("skeleton file must contain a `parent` column")

        if "x" not in skeleton.columns:
            skeleton["x"] = -1
        if "y" not in skeleton.columns:
            skeleton["y"] = -1
        if "annotated" not in skeleton.columns:
            skeleton["annotated"] = False

        if "tree" not in skeleton.columns:
            skeleton["tree"] = -1
            for idx, name in enumerate(skeleton["parent"].values):
                branch = np.where(skeleton["name"] == name)[0]
                if branch.shape[0] > 0:
                    branch = branch[0]
                    skeleton.loc[idx, "tree"] = branch
        if "swap_index" not in skeleton.columns:
            skeleton["swap_index"] = -1
            for idx, name in enumerate(skeleton["name"].values):
                for jdx, swap_name in enumerate(skeleton["swap"].values):
                    if swap_name == name:
                        skeleton.loc[idx, "swap_index"] = jdx

        self.skeleton = skeleton
        self.keypoint_index = self.skeleton.index
        self.n_keypoints = self.skeleton.index.shape[0]
        colors = (
            plt.cm.hsv(np.linspace(0, 1, self.n_keypoints))[..., :-1] * 255
        ).astype(np.uint8)
        if self.shuffle_colors:
            np.random.shuffle(colors)
        self.colors = colors
        self.inv_colors = np.bitwise_not(colors)
        self.idx = 0

    def _init_canvas(self):
        """ Initialize the canvas of the GUI.
        
        Create the canvas when the program runs.

        """
        self.canvas = cv2.resize(
            self.image.copy(), (0, 0), None, self.scale, self.scale, cv2.INTER_NEAREST
        )
        empty_size = (self.canvas.shape[0], int(self.canvas.shape[1] / 4.0), 3)
        empty = np.zeros(empty_size, dtype=np.uint8)
        self.canvas = np.concatenate((self.canvas, empty), axis=1)

    def _last_keypoint(self):
        """ Check if the idx is on the last keypoint

        Returns
        -------
        bool
            True if the index is on the last keypoint. False otherwise.

        """
        return self.idx == self.n_keypoints - 1

    def _update_text_locs(self):
        """ Update the text locations.

        Update the location based on the canvas.

        """
        self.text_locs = [
            (int(self.canvas_size[0] * 1.025), int(self.canvas_size[1] * 0.05) * idx)
            for idx in range(1, self.n_keypoints + 2)
        ]

    def _draw_text(self):
        """ Draw in the texts
        
        Go through the skeleton to get the appropriate texts.

        """

        text = str(self.image_idx) + "/" + str(self.n_images - 1)
        if len(text) > 9:
            text = text[:6] + "..."
        loc = self.text_locs[0]
        fontscale = self.text_scale * self.scale
        cv2.putText(
            img=self.canvas,
            text=text,
            org=loc,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=fontscale,
            color=(0, 0, 0),
            thickness=4,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            img=self.canvas,
            text=text,
            org=loc,
            fontFace=cv2.FONT_ITALIC,
            fontScale=fontscale,
            color=(255, 255, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        self.text_locs = self.text_locs[1:]
        for idx, loc in enumerate(self.text_locs):
            idx %= len(self.text_locs)
            text_idx = self.idx + idx
            text_idx %= self.n_keypoints

            if idx == 0:
                if np.all(self.skeleton.loc[:, "annotated"]):
                    color = (254, 79, 48)
                    border_color = (255, 255, 255)
                else:
                    color = self.colors[self.idx]  # (34, 87, 255)
                    border_color = np.bitwise_not(color)
                if self.skeleton.loc[self.idx, ["x", "y"]].isnull()[0]:
                    color = (127, 127, 127)

                # color = (3, 255, 118)
                thickness = 8

            else:
                if np.all(self.skeleton.loc[:, "annotated"]):
                    color = (254, 79, 48)
                else:
                    color = self.colors[idx]  # (34, 87, 255)
                if self.skeleton.loc[text_idx, ["x", "y"]].isnull()[0]:
                    color = (127, 127, 127)
                thickness = 2
                border_color = (0, 0, 0)
            

            text = self.skeleton.loc[text_idx, "name"]
            loc = self.text_locs[(idx + len(self.text_locs) // 4) % len(self.text_locs)]
            cv2.putText(
                img=self.canvas,
                text=text,
                org=loc,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=fontscale,
                color=(
                    int(border_color[0]),
                    int(border_color[1]),
                    int(border_color[2]),
                ),
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                img=self.canvas,
                text=text,
                org=loc,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=fontscale,
                color=(int(color[0]), int(color[1]), int(color[2])),
                thickness=1,
                lineType=cv2.LINE_AA,
            )

    def _draw_crosshairs(self, center, radius, color, thickness):
        """ Draws the crosshair on the point

        Draws an '+' crosshair on the point of interest
        
        Parameters
        ----------
        center: tuple
            (x,y) coordinates of the point.

        radius: int
            The radius of the crosshair.

        color: tuple
            The color of the crosshair in BGR.

        thickness: int
            Thickness of the drawing.

        """

        ypt1 = (center[0], center[1] + radius)
        ypt2 = (center[0], center[1] - radius)

        xpt1 = (center[0] + radius, center[1])
        xpt2 = (center[0] - radius, center[1])
        color = (int(color[0]), int(color[1]), int(color[2]))
        cv2.line(self.canvas, ypt1, ypt2, color, thickness, cv2.LINE_AA)
        cv2.line(self.canvas, xpt1, xpt2, color, thickness, cv2.LINE_AA)

    def _get_scaled_coords(self, idx):
        """ Get the scaled coordinates

        Parameters
        ----------
        idx: int
            Index of the keypoint in the skeleton DataFrame.

        """
        coords = self.skeleton.loc[idx, ["x", "y"]].values * self.scale
        coords = tuple([int(x) for x in coords])
        return coords

    def _draw_point(self, center, radius, color, thickness=1):
        """ Draw a single point.

        Draws a single point at the center with specified radius and color.

        Parameters
        ----------
        center: tuple
            (x,y) coordinates of the point.

        radius: int
            The radius of the point.

        color: tuple
            The color of the crosshair in BGR.

        """
        cv2.circle(
            img=self.canvas,
            center=center,
            radius=radius,
            color=(int(color[0]), int(color[1]), int(color[2])),
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )
        cv2.circle(
            img=self.canvas,
            center=center,
            radius=1,
            color=(int(color[0]), int(color[1]), int(color[2])),
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

    def _draw_points(self):
        """ Draws multiple points.

        Draws all the annotated points in addition to the new point

        """
        for idx in self.keypoint_index:
            if idx != self.idx:
                if not self.skeleton.loc[idx, ["x", "y"]].isnull()[0]:
                    if np.all(self.skeleton.loc[:, "annotated"]):
                        color = self.colors[idx]
                        inv_color = (254, 79, 48)
                        # inv_color = None
                    else:
                        # color = (34, 87, 255)
                        color = self.colors[idx]
                        inv_color = self.inv_colors[idx]
                    center = self._get_scaled_coords(idx)
                    radius = 5
                    if inv_color is not None:
                        self._draw_point(center, radius, inv_color, 2)
                    self._draw_point(center, radius, color)
        
        if not self.skeleton.loc[self.idx, ["x", "y"]].isnull()[0]:
            center = self._get_scaled_coords(self.idx)
            radius = 8
            # color = (3, 255, 118)
            color = self.colors[self.idx]
            inv_color = self.inv_colors[self.idx]
            self._draw_point(center, radius, inv_color, 2)
            self._draw_crosshairs(center, radius + 3, inv_color, 2)
            self._draw_point(center, radius, color)
            self._draw_crosshairs(center, radius + 3, color, 1)

    def _draw_lines(self):
        """ Draw lines

        Connect every keypoint with a line if they are annotated.

        """
        if np.all(self.skeleton.loc[:, "annotated"]):
            color = (254, 79, 48)
        else:
            color = (34, 87, 255)
        for idx in self.keypoint_index:
            tree = self.skeleton.loc[idx, "tree"]
            if (tree >= 0 and
                not self.skeleton.loc[tree, ["x", "y"]].isnull()[0] and
                not self.skeleton.loc[idx, ["x", "y"]].isnull()[0]):
                pt1 = self._get_scaled_coords(idx)
                pt2 = self._get_scaled_coords(tree)
                cv2.line(
                    img=self.canvas,
                    pt1=pt1,
                    pt2=pt2,
                    color=color,
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )

    def _update_canvas(self):
        """ Update the canvas

        Make a canvas, draw in text, lines, and points over 
        the existing window.

        """
        self._update_canvas_size()
        self._update_text_locs()
        self._init_canvas()
        self._draw_text()
        self._draw_lines()
        self._draw_points()
        cv2.imshow(self.window_name, self.canvas)

    def _update_canvas_size(self):
        """ Update the size of the canvas.

        Change the canvas size based on the scale attribute.

        """
        self.canvas_size = tuple([x * self.scale for x in self.image.shape[:2][::-1]])

    def _zoom(self):
        """ Key bindings for zooming.

        Creates additional key bindings for the program.
        The bindings are as follows:
        
        ------------------------------------------------------------
             Keys             |   Action                           
        ------------------------------------------------------------
        >    +                |   Zoom in by scale factor
        >    -                |   Zoom out by scale factor
        ------------------------------------------------------------

        Also, if the scale is less than 1, then it sets the scale to 1.

        """

        if self.key is keys.PLUS:
            self.scale += 0.1

        if self.key is keys.MINUS:
            self.scale -= 0.1
            self._update_text_locs()
            self._update_canvas_size()

        # if self.scale < 1:
        #    self.scale = 1.

    def _wasd(self):
        """ Key bindings for WASD.

        Creates additional key bindings for the program.
        The bindings are as follows:
        
        ------------------------------------------------------------
             Keys             |   Action                           
        ------------------------------------------------------------
        >    W/A/S/D          |   Move active keypoint 1px
        >    Shift + W/A/S/D  |   Move active keypoint 10px
        ------------------------------------------------------------

        This allows the user to make finer adjustments as needed.

        """
        if self.key is keys.SPACE:
            if self.wasd_mode:
                self.wasd_mode = False
                self.move_len = 10.0
            else:
                self.wasd_mode = True
                self.move_len = 1.0

        if self.key is keys.W:
            self.skeleton.loc[self.idx, "y"] -= self.move_len / self.scale
        if self.key is keys.S:
            self.skeleton.loc[self.idx, "y"] += self.move_len / self.scale
        if self.key is keys.A:
            self.skeleton.loc[self.idx, "x"] -= self.move_len / self.scale
        if self.key is keys.D:
            self.skeleton.loc[self.idx, "x"] += self.move_len / self.scale

    def _move_idx(self):
        """ Key bindings for WASD.

        Creates additional key bindings for the program.
        The bindings are as follows:
        
        ------------------------------------------------------------
             Keys             |   Action                           
        ------------------------------------------------------------
        >    Up key, Ctrl-Y,  |   
             Shift-Tab        |   Move up in active keypoint list
        >    Down key, Ctrl-I,|
             Tab              |   Move down in active keypoint list
        ------------------------------------------------------------

        This allows the user to make finer adjustments as needed.

        """

        # Moving down
        if self.key in [keys.DOWNARROW, keys.TAB]:
            if self._last_keypoint():
                self.idx = 0
            else:
                self.idx += 1
            self._set_annotated()

        # Moving up
        if self.key in [keys.UPARROW, keys.SHIFT_TAB]:
            if self.idx == 0:
                self.idx = self.n_keypoints - 1
            else:
                self.idx -= 1
            self._set_annotated()

    def _exit(self):
        """ Key bindings for WASD.

        Creates additional key bindings for the program.
        The bindings are as follows:
        
        ------------------------------------------------------------
             Keys             |   Action                           
        ------------------------------------------------------------
        >    Esc, 0, Q        |   Exit the program
        ------------------------------------------------------------

        """
        return self.key in [keys.Q, keys.ESC]

    def _hotkeys(self):
        raise NotImplementedError

    def run(self):
        """ Run the program.

        Runs the program by continually calling for hotkeys function
        defined in the subclasses.

        """
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, _mouse_click, self)
        self._update_canvas()
        while True:
            self.key = cv2.waitKey(self.refresh) & 0xFF
            self._hotkeys()
            if self._exit():
                break
        cv2.destroyAllWindows()
        cv2.waitKey(1)
