import numpy as np


# Some generic loss functions
def square_loss(a: np.array, b: np.array) -> np.array:
    return np.square(a - b)


def absolute_loss(a: np.array, b: np.array) -> np.array:
    return np.absolute(a - b)