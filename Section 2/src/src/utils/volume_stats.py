"""
Contains various functions for computing statistics over 3D volumes
"""
import numpy as np


def Dice3d(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the Dice Similarity coefficient for two 3-dimensional volumes.
    Volumes are expected to be of the same size. We are expecting binary masks -
    0's are treated as background and anything else is counted as data.

    Arguments:
        a {Numpy array} -- 3D array with first volume
        b {Numpy array} -- 3D array with second volume

    Returns:
        float -- Dice coefficient in range [0, 1], or -1 if both volumes are empty
    """
    if len(a.shape) != 3 or len(b.shape) != 3:
        raise ValueError(f"Expecting 3 dimensional inputs, got {a.shape} and {b.shape}")

    if a.shape != b.shape:
        raise ValueError(f"Expecting inputs of the same shape, got {a.shape} and {b.shape}")

    overlap = np.count_nonzero((a != 0) & (b != 0))
    vol = np.count_nonzero(a) + np.count_nonzero(b)

    if vol == 0:
        return -1

    return 2.0 * float(overlap) / float(vol)


def Jaccard3d(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the Jaccard Similarity coefficient for two 3-dimensional volumes.
    Volumes are expected to be of the same size. We are expecting binary masks -
    0's are treated as background and anything else is counted as data.

    Arguments:
        a {Numpy array} -- 3D array with first volume
        b {Numpy array} -- 3D array with second volume

    Returns:
        float -- Jaccard coefficient in range [0, 1], or -1 if both volumes are empty
    """
    if len(a.shape) != 3 or len(b.shape) != 3:
        raise ValueError(f"Expecting 3 dimensional inputs, got {a.shape} and {b.shape}")

    if a.shape != b.shape:
        raise ValueError(f"Expecting inputs of the same shape, got {a.shape} and {b.shape}")

    intersection = np.count_nonzero((a != 0) & (b != 0))
    union = np.count_nonzero((a != 0) | (b != 0))

    if union == 0:
        return -1

    return float(intersection) / float(union)
