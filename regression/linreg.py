import numpy as np
import csv
import sys

def read_data(filename: str) -> 
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        mat = []
        for line in reader:
            mat.append(list(map(float, line)))
    return np.array(mat)

# NOTE: Assumes label column is the last column
def split_feats_vals(mat):
    return mat[:, :-1], mat[:, -1]

# Prepends a column of 1's to x
def prepend_ones(x):
    return np.hstack((np.ones((len(x), 1)), x))

# Fits using the normal equations
# TODO: Implement regularization
def fit_normal(x, y):
    x = prepend_ones(x)
    return x.T.dot(x).inv().dot(x.T).dot(y)

# Fits using gradient descent
# TODO: Implement regularization
def fit_gradient_descent(x, y, a: float=1.0, convergence: float=1e-6):
    # Need to put column of 1's in front of x
    x = prepend_ones(x)
    iters = 0
    m, n = x.shape
    w = np.ones(m)
    new_w = np.empty_like(w)

    # Perform gradient descent until convergence
    while np.linalg.norm(w - y) > convergence:
        for j in range(n):
            diff = sum((w.dot(y[i]) - x[i]) * x[i, j] for i in range(m))
            new_w[j] = w - (a / m) * diff
        w = new_j
        iters += 1

    return w

# Predicts using model from data fitting
def predict(model, data):
    return prepend_ones(data).dot(model)

# Compute mean squared error
def mean_square_error(ground_truth, estimated):
    return np.mean((ground_truth - estimated) ** 2)
