import numpy as np

# Divides {dataset} into {k} equal (or nearly equal) partitions
def kdivide(dataset, k):
    assert k > 1, "Must have k > 1"
    return list(map(np.squeeze, np.array_split(dataset, k)))

# Divides {dataset} into features and labels
# NOTE: Assumes labels are the last column
def fldivide(dataset):
    return [dataset[:, :-1], dataset[:, -1].astype(np.int)]
