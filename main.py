#!/usr/bin/env python3

import sys
import os

import numpy as np
import matplotlib.pyplot as plt

from simpleml.preprocess import reader, scaling
from simpleml.classifiers.logreg import GradientDescentLogisticRegressor
from simpleml.util.dataops import add_feature_bias, basis_expand, fljoin, fldivide
from simpleml.containers import SimpleDataset


def plot_misclass_err(ks, err, fig, dsetname):
    plt.figure(fig)
    plt.plot(ks, err, 'go')
    plt.grid(True)
    plt.xlabel('k')
    plt.ylabel('Training error')
    xmin, xmax = np.min(ks), np.max(ks)
    ymin, ymax = np.min(err), np.max(err)
    plt.xlim([xmin-1, xmax+1])
    plt.ylim([0.70 * ymin, 1.3 * ymax])
    plt.title("Misclassification error for {}".format(dsetname))

# Generates a grid of points that encompass {dataset} at a resolution of
# {resolution}
def generate_grid(dataset, resolution=50):
    d = dataset.features()
    xmin, xmax = np.min(d[:, 0]), np.max(d[:, 0])
    ymin, ymax = np.min(d[:, 1]), np.max(d[:, 1])
    X = np.linspace(xmin-0.1*abs(xmin), xmax+0.1*abs(xmax), resolution)
    Y = np.linspace(ymin-0.1*abs(ymin), ymax+0.1*abs(ymax), resolution)
    xx, yy = np.meshgrid(X, Y)
    return X, Y, np.dstack((xx, yy))

def plot_contour(dataset, clf, scalevals, figure_num=0, title=''):
    plt.figure(figure_num)
    # Number of points on each axis of grid
    resolution = 100
    gridx, gridy, grid = generate_grid(dataset, resolution=resolution)

    # Train classifier
    clf.train(dataset.features(), dataset.labels())

    # Predict over grid
    predicted_l = np.empty((resolution, resolution))
    for i in range(resolution):
        gridvals = basis_expand(grid[i], *dataset.expand_funcs)
        #gridvals = grid[i]
        #gridvals *= scalevals
        predicted_l[i, :] = clf.predict(gridvals)

    # Plot contour + dataset points
    plt.contourf(gridx, gridy, predicted_l)
    ds = dataset.features()
    l = dataset.labels()
    zero, one = ds[l == 0], ds[l == 1]
    plt.plot(zero[:, 0], zero[:, 1], 'yo', label='0')
    plt.plot(one[:, 0], one[:, 1], 'g^', label='1')
    xmin, xmax = np.min(gridx), np.max(gridx)
    ymin, ymax = np.min(gridy), np.max(gridy)
    plt.axis([xmin, xmax, ymin, ymax])
    plt.legend(loc='upper left')
    plt.xlabel("x values")
    plt.ylabel("y values")
    plt.title(title)

def main():
    if len(sys.argv) < 2:
        print("usage: {} dataset [dataset, ...]".format(sys.argv[0]))
        sys.exit(1)

    filenames = sys.argv[1:]
    for i, fname in enumerate(filenames, 1):
        dsetname = os.path.basename(fname)
        dataset = SimpleDataset(reader.read(fname))
        dataset.expand_basis(
            lambda x: x ** 2,
            lambda x: (x[:, 0] * x[:, 1]).reshape(len(x), 1),
        )
        #scalevals = dataset.scale_features()
        scalevals = 1
        clf = GradientDescentLogisticRegressor(bias=True, regularization=1)
        clf.train(dataset.features(), dataset.labels())
        predicted_labels = clf.predict(dataset.features())
        misclass_err = clf.error(predicted_labels, dataset.labels())
        plot_contour(dataset, clf, scalevals, figure_num=i, \
                title='Contour plot for {}, misclass err = {}'.format(dsetname, misclass_err))
    plt.show()

main()
