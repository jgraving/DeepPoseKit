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
import imgaug as ia

MACHINE_EPSILON = np.finfo(np.float64).eps

__all__ = [
    "draw_confidence_maps",
    "draw_confidence_map",
    "graph_to_edges",
    "draw_keypoints",
    "draw_graph",
    "numpy_to_imgaug",
    "imgaug_to_numpy",
    "keypoint_errors",
]


def graph_to_edges(graph):
    edges = graph.copy()
    parents = set()
    edge = {}
    for idx in range(len(edges)):
        if edges[idx] == -1:
            parents.add(idx)
        else:
            edge[idx] = edges[idx]

    for idx in range(len(edges)):
        if idx in parents:
            edges[idx] = idx
        else:
            idx0 = idx
            while idx0 not in parents:
                idx0 = edge[idx0]
            edges[idx] = idx0

    return edges


def draw_graph(keypoints, height, width, output_shape, graph, sigma=1, linewidth=1):
    # One channel for each edge
    keypoints = keypoints.copy()
    edge_labels = graph_to_edges(graph)
    labels = np.unique(edge_labels)
    out_height = output_shape[0]
    out_width = output_shape[1]
    sigma *= height / out_height
    output_shape = (out_height, out_width, labels.shape[0])
    confidence = np.zeros(output_shape, dtype=np.float64)
    edge_confidence_list = []
    for idx, label in enumerate(labels):
        lines = graph[edge_labels == label]
        lines_idx = np.where(edge_labels == label)[0]
        edge_confidence = np.zeros((out_height, out_width, lines.shape[0]))
        zeros = np.zeros((height, width, 1), dtype=np.uint8)
        for jdx, (line_idx, line) in enumerate(zip(lines_idx, lines)):
            if line >= 0:
                pt1 = keypoints[line_idx]
                pt2 = keypoints[line]
                line_map = cv2.line(
                    zeros.copy(),
                    (int(pt1[0]), int(pt1[1])),
                    (int(pt2[0]), int(pt2[1])),
                    1,
                    linewidth,
                    lineType=cv2.LINE_AA,
                )
                blurred = cv2.GaussianBlur(
                    line_map.astype(np.float64), (height + 1, width + 1), sigma
                )
                resized = cv2.resize(blurred, (out_width, out_height)) + MACHINE_EPSILON
                edge_confidence[..., jdx] = resized
        edge_confidence = edge_confidence[..., 1:]
        edge_confidence_list.append(edge_confidence)
        confidence[..., idx] = edge_confidence.sum(-1)
    edge_confidence = np.concatenate(edge_confidence_list, -1)
    confidence = np.concatenate((confidence, edge_confidence), -1)
    return confidence


def draw_keypoints(keypoints, height, width, output_shape, sigma=1, normalize=True):
    keypoints = keypoints.copy()
    n_keypoints = keypoints.shape[0]
    out_height = output_shape[0]
    out_width = output_shape[1]
    keypoints[:, 1] *= out_height / height
    keypoints[:, 0] *= out_width / width
    confidence = np.zeros((out_height, out_width, n_keypoints))
    xv = np.arange(out_width)
    yv = np.arange(out_height)
    xx, yy = np.meshgrid(xv, yv)
    for idx in range(n_keypoints):
        keypoint = keypoints[idx]
        gaussian = (yy - keypoint[1]) ** 2
        gaussian += (xx - keypoint[0]) ** 2
        gaussian *= -1
        gaussian /= 2 * sigma ** 2
        gaussian = np.exp(gaussian)
        confidence[..., idx] = gaussian
    if not normalize:
        confidence /= sigma * np.sqrt(2 * np.pi)
    return confidence


def draw_confidence_map(
    image, keypoints, graph=None, output_shape=None, use_graph=True, sigma=1
):
    height = image.shape[0]
    width = image.shape[1]

    if not output_shape:
        output_shape = image.shape[:2]
    keypoints_confidence = draw_keypoints(keypoints, height, width, output_shape, sigma)
    if use_graph and isinstance(graph, np.ndarray):
        edge_confidence = draw_graph(
            keypoints, height, width, output_shape, graph, sigma
        )
        sum_keypoints = keypoints_confidence.sum(-1, keepdims=True)
        idx = np.unique(graph_to_edges(graph)).shape[0]
        sum_edges = edge_confidence[..., :idx].sum(-1, keepdims=True)
        sum_edges_keypoints = np.concatenate((sum_edges, sum_keypoints), -1)
        sum_edges_keypoints = sum_edges_keypoints.sum(-1, keepdims=True)

        confidence = (
            keypoints_confidence,
            edge_confidence,
            sum_edges,
            sum_edges_keypoints,
        )
        confidence = np.concatenate(confidence, -1)
    else:
        confidence = keypoints_confidence

    return confidence


def draw_confidence_maps(
    images, keypoints, graph=None, output_shape=None, use_graph=True, sigma=1
):
    n_samples = keypoints.shape[0]
    confidence_maps = []
    for idx in range(n_samples):
        confidence = draw_confidence_map(
            images[idx], keypoints[idx], graph, output_shape, use_graph, sigma
        )
        confidence_maps.append(confidence)
    confidence_maps = np.stack(confidence_maps)

    return confidence_maps


def numpy_to_imgaug(image, keypoints):
    """Returns imgaug keypoints"""
    keypoints_list = []
    for keypoint in keypoints:
        keypoints_list.append(ia.Keypoint(x=keypoint[0], y=keypoint[1]))
    keypoints = ia.KeypointsOnImage(keypoints_list, shape=image.shape)

    return keypoints


def imgaug_to_numpy(keypoints):
    """Returns numpy keypoints"""
    keypoints_list = []
    for keypoint in keypoints.keypoints:
        keypoints_list.append([keypoint.x, keypoint.y])
    keypoints = np.stack(keypoints_list)
    return keypoints


def keypoint_errors(y_true, y_pred):
    y_error = y_true - y_pred

    euclidean = np.sqrt(np.sum(y_error ** 2, axis=-1))
    mae = np.mean(np.abs(y_error), axis=-1)
    mse = np.mean(y_error ** 2, axis=-1)
    rmse = np.sqrt(mse)

    return [y_error, euclidean, mae, mse, rmse]
