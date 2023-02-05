# Copyright © 2023 "Bronte" Sihan Li

"""
This module contains functions to calculate the distance metric between two
vectors in a matrix.
"""

import numpy as np


def get_distance(
    data: np.ndarray, unknown: np.ndarray, metric: str = "euclidean", **kwargs
):
    """
    Takes in a N x m data matrix to evaluate against the E x m exemplar data matrix and
    returns a E x N matrix with E distances for each of the N points to be evaluated.
    E is the number of exemplars and N is the number of points to be evaluated, m is the number of features.
    """
    DISTANCE_METRIC_MAPPING = {
        "euclidean": euclidean,
        "cosine": cosine,
        "minkowski": minkowski,
    }

    assert (
        unknown.shape[1] == data.shape[1]
    ), "Data and unknown must have the same number of features."

    return DISTANCE_METRIC_MAPPING[metric](unknown, data, **kwargs)


def euclidean(A, B):
    """
    Calculates the euclidean distance between two matrices.
    """
    # Use distance matrix formula
    A_squared = np.sum(A**2, axis=1)[:, np.newaxis]
    B_squared = np.sum(B**2, axis=1)
    AB = np.dot(A, B.T)

    return np.sqrt(A_squared + B_squared - 2 * AB).T


def cosine(a, b):
    """
    Calculates the cosine distance between two vectors.
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def minkowski(a, b, p=2):
    """
    Calculates the minowski distance between two vectors.
    """
    return np.sum(np.abs(a - b) ** p) ** (1 / p)


