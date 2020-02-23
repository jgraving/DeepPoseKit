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

from tensorflow.keras.utils import Sequence
import cv2
import numpy as np
import os

__all__ = ["VideoReader", "VideoWriter"]


class VideoReader(cv2.VideoCapture, Sequence):

    """Read a video in batches.

    Parameters
    ----------
    videopath: str
        Path to the video file.
    batch_size: int, default = 1
        Batch size for reading frames
    gray: bool, default = False
        If gray, return only the middle channel
    """

    def __init__(self, videopath, batch_size=1, gray=False):

        if isinstance(videopath, str):
            if os.path.exists(videopath):
                super(VideoReader, self).__init__(videopath)
                self.videopath = videopath
            else:
                raise ValueError("file or path does not exist")
        else:
            raise TypeError("videopath must be str")
        self.batch_size = batch_size
        self.n_frames = int(self.get(cv2.CAP_PROP_FRAME_COUNT))
        self.idx = 0
        self.fps = self.get(cv2.CAP_PROP_FPS)
        self.height = int(self.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.finished = False
        self.gray = gray
        self._read = super(VideoReader, self).read

    def read(self):
        """ Read one frame

        Returns
        -------
        frame: array
            Image is returned of the frame if a frame exists.
            Otherwise, return None.

        """
        ret, frame = self._read()
        if ret:
            self.idx += 1
            if self.gray:
                frame = frame[..., 1][..., None]
            return frame
        else:
            self.finished = True
            return None

    def read_batch(self, batch_size=1, asarray=False):
        """ Read in a batch of frames.

        Parameters
        ----------
        batch_size: int, default 1
            Number of frames to pull from the video.

        asarray: bool, default False
            If True, stack the frames (in numpy).

        Returns
        -------
        frames: list or array
            A batch of frames from the video.

        """
        frames = []
        for idx in range(batch_size):
            frame = self.read()
            if not self.finished:
                frames.append(frame)
        empty = len(frames) == 0
        if asarray and not empty:
            frames = np.stack(frames)
        if not empty:
            return frames
        else:
            return

    def close(self):
        """ Close the VideoReader.

        Returns
        -------
        bool
            Returns True if successfully closed.

        """
        self.release()
        return not self.isOpened()

    def __len__(self):
        return int(np.ceil(self.n_frames / float(self.batch_size)))

    def __getitem__(self, index):

        if self.finished:
            raise StopIteration
        if isinstance(index, (int, np.integer)):
            idx0 = index * self.batch_size
            if self.idx != idx0:
                self.set(cv2.CAP_PROP_POS_FRAMES, index - 1)
                self.idx = idx0
            return self.read_batch(self.batch_size, True)
        else:
            raise NotImplementedError

    def __next__(self):

        if self.finished:
            raise StopIteration
        else:
            return self.read_batch(self.batch_size, True)

    def __del__(self):
        self.close()

    @property
    def current_frame(self):
        return int(self.get(cv2.CAP_PROP_POS_FRAMES))

    @property
    def current_time(self):
        return self.get(cv2.CAP_PROP_POS_MSEC)

    @property
    def percent_finished(self):
        return self.get(cv2.CAP_PROP_POS_AVI_RATIO) * 100


class VideoWriter:

    """Read a video in batches.

    Parameters
    ----------
    path: str
        Path to save the video file.
    frame_size: tuple
        Size of the frame as (width, height)
    codec: str, default = 'FFV1'
        FourCC video codec for encoding the video.
        Options include `MP4V',  'X264', 'H264', etc.
    fps: int or float, default = 30.0
        Frame rate for the encoded video
    color: bool, default = True
        Whether or not the video is color
    """

    def __init__(self, path, frame_size, codec="FFV1", fps=30.0, color=True):
        codec = cv2.VideoWriter_fourcc(*codec)
        self.stream = cv2.VideoWriter(path, codec, fps, frame_size, color)

    def write(self, frame):
        self.stream.write(frame)

    def write_batch(self, batch):
        for frame in batch:
            self.write(frame)

    def close(self):
        self.stream.release()
        return not self.stream.isOpened()
