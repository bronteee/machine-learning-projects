# Copyright © 2023 "Bronte" Sihan Li
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
This module implements the KNearestNeighbor class.
"""

import numpy as np
import pandas as pd
from project2.utility.distance import get_distance


class KNearestNeighbor:
    """
    This class implements the k nearest neighbor algorithm.
    """
    def __init__(self, distance_metric: str = 'euclidean', k: int = 1) -> None:
        self.distance_metric = distance_metric
        self.k = k

    def fit_predict(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.DataFrame,
    ) -> np.ndarray:
        """
        Fits the model to the training data and predicts the labels for the test data using
        the nearest neighbor for each point.
        """

        self.X_train = X_train
        self.y_train = y_train

        # Calculate the distance matrix
        distance_matrix = get_distance(
            X_train.to_numpy(), X_test.to_numpy(), self.distance_metric
        )
        # Get the index of the nearest neighbor for each test data point
        nearest_neighbors = self._get_nearest_neighbors(distance_matrix)
        # Return the labels of the nearest neighbors
        return self._get_labels(nearest_neighbors)

    def _get_nearest_neighbors(self, distance_matrix: np.ndarray) -> np.ndarray:
        """
        Get the index of nearest neighbor for each vector in the test data
        """
        # Get minimum k values of each column in distance matrix as a matrix
        return np.argpartition(distance_matrix, self.k, axis=0)[: self.k, :].T

    def _get_labels(self, X: np.ndarray) -> np.ndarray:
        """
        Get the labels of the nearest neighbors for each vector in X using
        majority voting
        """
        n_classes = self.y_train.nunique()[0]
        # Construct a matrix to store the number of votes for each label
        votes = np.zeros((X.shape[0], n_classes))
        for row, _ in enumerate(X):
            # Get the labels of the nearest neighbors
            labels = self.y_train.iloc[X[row, :]].to_numpy()
            # Count the number of votes for each label
            for label in labels:
                votes[row, label] += 1
        # Return the label with the most votes
        return np.argmax(votes, axis=1, keepdims=True)

    def get_accuracy(self, y_pred: np.ndarray, y_test: pd.DataFrame) -> float:
        """
        Calculate the accuracy of the prediction
        """
        # Check that the shapes of the prediction and the test data are the same
        assert (
            y_pred.shape == y_test.shape
        ), f'Expected {y_test.shape} but got {y_pred.shape}'

        a = y_pred
        b = y_test.to_numpy()
        return np.count_nonzero(np.equal(a, b)) / b.size

    def confusion_matrix(
        self, y_pred: np.ndarray, y_test: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute the confusion matrix of the prediction
        """
        # Check that the shapes of the prediction and the test data are the same
        assert (
            y_pred.shape == y_test.shape
        ), f'Expected {y_test.shape} but got {y_pred.shape}'

        predicted = pd.Series(y_pred.flatten(), name='Predicted')
        actual = pd.Series(y_test.to_numpy().flatten(), name='Actual')
        return pd.crosstab(predicted, actual)
