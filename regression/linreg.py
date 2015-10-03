import numpy as np
import matplotlib.pyplot as plt
import csv
import sys

def read_data(filename: str):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        mat = []
        for line in reader:
            mat.append(list(map(float, line)))
    return np.array(mat)

# NOTE: Assumes label column is the last column
def split_feats_vals(mat):
    return mat[:, :-1], mat[:, -1]

# Makes data into an nth-order set of data (with 1's column prepended)
# n+1 to account for 0th order
# Assumes x is column vector of observations
def make_nth_order(x, n: int):
    return np.hstack([x ** i for i in range(n+1)])

# Fits using the normal equations
# TODO: Implement regularization
def fit_normal(x, y):
    return np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)

# Fits using gradient descent
# TODO: Implement regularization
def fit_gradient_descent(x, y, a: float=1.0, convergence: float=1e-6):
    iters = 0
    m, n = x.shape
    w = np.empty(n)
    new_w = np.ones_like(w)

    # Perform gradient descent until convergence
    while np.linalg.norm(w - new_w) > convergence:
        np.copyto(w, new_w)
        for j in range(n):
            diff = sum((w.dot(x[i]) - y[i]) * x[i, j] for i in range(m))
            new_w[j] = w[j] - (a / m) * diff
        iters += 1
    print("iters={}".format(iters))
    return new_w

# Predicts using model from data fitting
def predict(model, data):
    return data.dot(model)

# Compute mean squared error
def mean_square_error(ground_truth, estimated):
    return np.mean((ground_truth - estimated) ** 2)

# Plot scatter plot of data points as well as regression curve
def plot_scatter_curve(x, y, model, fignum=0, title=''):
    plt.figure(fignum)
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    plt.xlim(0.70*xmin, 1.3*xmax)
    plt.ylim(0.70*ymin, 1.3*ymax)
    plt.scatter(x, y)
    r = np.linspace(xmin, xmax, num=100).reshape((100, 1))
    s = make_nth_order(r, len(model)-1).dot(model)
    plt.plot(r, s, 'g-')
    plt.axis([xmin, xmax, ymin, ymax])
    plt.title(title)

def main():
    if len(sys.argv) < 2:
        print("Usage:\n\t{} [trainfile] [testfile]".format(sys.argv[0]))
        sys.exit(1)

    train_file, test_file = sys.argv[1:]
    train_data, train_labels = split_feats_vals(read_data(train_file))
    test_data, test_labels   = split_feats_vals(read_data(test_file))

    for i in range(5):
        nth_train_data = make_nth_order(train_data, i)
        nth_test_data  = make_nth_order(test_data, i)
        w = fit_gradient_descent(nth_train_data, train_labels, a=0.4, convergence=1e-9)
        print(w)
        estimated = predict(w, nth_test_data)
        mse = mean_square_error(test_labels, estimated)
        print("order={}, mse = {}".format(i, mse))
        plot_scatter_curve(test_data, test_labels, w, fignum=i,
                title="Gradient descent, order {}, mse={}, alpha={}".format(i, mse, 0.4))

    plt.show()

main()
