import numpy as np

# Scales dataset to be values between [-1, 1]
def unit_scale(dataset):
    scalevals = 1 / np.max(np.abs(dataset), axis=0)
    return dataset * scalevals, scalevals
