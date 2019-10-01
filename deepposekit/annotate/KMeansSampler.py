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
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.utils.validation import check_is_fitted
from deepposekit.annotate.utils.image import check_image_array


class KMeansSampler(MiniBatchKMeans):
    def __init__(
        self,
        n_clusters=10,
        init="k-means++",
        max_iter=100,
        batch_size=100,
        verbose=0,
        compute_labels=True,
        random_state=None,
        tol=0.0,
        max_no_improvement=10,
        init_size=None,
        n_init=3,
        reassignment_ratio=0.01,
    ):

        super(KMeansSampler, self).__init__(
            n_clusters=n_clusters,
            init=init,
            max_iter=max_iter,
            batch_size=batch_size,
            verbose=verbose,
            compute_labels=compute_labels,
            random_state=random_state,
            tol=tol,
            max_no_improvement=max_no_improvement,
            init_size=init_size,
            n_init=n_init,
            reassignment_ratio=reassignment_ratio,
        )
        self._fit = super(KMeansSampler, self).fit
        self._partial_fit = super(KMeansSampler, self).partial_fit
        self._predict = super(KMeansSampler, self).predict

    def sample_idx(self, X, n_samples_per_label=100):
        labels = self.predict(X)

        X_new = []
        y_new = []
        index = np.arange(X.shape[0])
        for idx in np.unique(labels):
            label_idx = index[labels == idx]
            if label_idx.shape[0] > 0:
                if label_idx.shape[0] < n_samples_per_label:
                    n_samples = label_idx.shape[0]
                else:
                    n_samples = n_samples_per_label
                sample_idx = np.random.choice(label_idx, n_samples, replace=False)
                X_new.append(sample_idx)
                y_new.append(np.ones_like(sample_idx, dtype=np.int32) * idx)
        X_new = np.concatenate(X_new)
        y_new = np.concatenate(y_new)

        return X_new, y_new

    def sample_data(self, X, n_samples_per_label=100):
        """Sample evenly from each cluster for X.
        Parameters
        ----------
        X : array-like, shape = [n_samples, rows, cols, channels]
            Coordinates of the data points to cluster.
        n_samples_per_label : int
            Number of samples per cluster label.
            If X does not contain enough samples in
            a cluster, all samples for that cluster
            are used without replacement.
        Returns
        -------
        X_new : array-like, shape = [n_samples, rows, cols, channels]
            The sampled data
        y_new : array-like, shape = [n_samples,]
            The cluster labels each sample

        """
        labels = self.predict(X)

        X_new, y_new = self.sample_idx(X, n_samples_per_label)
        X_new = X[X_new]
        return X_new, y_new

    def fit(self, X, y=None):
        """Compute the centroids on X by chunking it into mini-batches.
        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, rows, cols, channels]
            Training instances to cluster.
        y : Ignored
        """
        X = check_image_array(self, X)

        return self._fit(X, y)

    def partial_fit(self, X, y=None):
        """Update k means estimate on a single mini-batch X.
        Parameters
        ----------
        X : array-like, shape = [n_samples, rows, cols, channels]
            Coordinates of the data points to cluster.
        y : Ignored
        """
        X = check_image_array(self, X)

        return self._partial_fit(X, y)

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.
        In the vector quantization literature, `cluster_centers_` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, rows, cols, channels]
            New data to predict.
        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        check_is_fitted(self, "cluster_centers_")
        X = check_image_array(self, X)

        return self._predict(X)

    def plot_centers(self, n_rows=2, figsize=(20, 20)):

        check_is_fitted(self, "cluster_centers_")

        n_cols = self.n_clusters // n_rows

        mean = self.cluster_centers_.mean(0)
        centers = self.cluster_centers_ - mean[None, ...]
        centers = centers.reshape(n_rows, n_cols, self.rows, self.cols, self.channels)
        centers = centers.swapaxes(1, 2).reshape(
            n_rows * self.rows, n_cols * self.cols, self.channels
        )
        if self.channels == 1:
            centers = centers[..., 0]
        fig = plt.figure(figsize=figsize)
        plt.imshow(centers, cmap="seismic", vmin=-255, vmax=255)

        return fig
