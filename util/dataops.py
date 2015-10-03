import numpy as np

# Divides {dataset} into {k} equal (or nearly equal) partitions
def kdivide(dataset, k):
    assert k > 1, "Must have k > 1"
    return list(map(np.squeeze, np.array_split(dataset, k)))

# Divides {dataset} into features and labels
# NOTE: Assumes labels are the last column
def fldivide(dataset):
    return [dataset[:, :-1], dataset[:, -1]]

# Makes {dataset} into an nth-order dataset by horizontally appending columns
# e.g. [x^0 x^1 x^2 ... x^n]
def make_nth_order(dataset, n):
    return np.hstack([dataset ** i for i in range(n+1)])
