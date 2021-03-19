import numpy as np


def verify_numpy_array(arr):
    """Convert numpy array to string for compatability with approvaltests."""
    with np.printoptions(threshold=np.inf):
        return np.array2string(arr, max_line_width=1e3)
