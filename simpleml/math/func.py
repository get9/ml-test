import sys

import numpy as np
from math import log

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def log_likelihood(h, xs, ys, ws, r):
    if not callable(h):
        raise TypeError('h must be callable')
    hvals = h(xs, ws)
    m = len(xs[0])
    return -(ys @ np.log(hvals) + (1 - ys) @ np.log(1 - hvals)) / m + r * np.sum(ws[1:] ** 2) / (2 * m)
