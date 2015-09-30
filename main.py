#!/usr/bin/env python3

import numpy as np
import os
import dt
import sys
import knn
import reader
import dataops
import crossval
import matplotlib.pyplot as plt

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
    xmin, xmax = np.min(dataset[:, 0]), np.max(dataset[:, 0])
    ymin, ymax = np.min(dataset[:, 1]), np.max(dataset[:, 1])
    X = np.linspace(xmin-0.1*abs(xmin), xmax+0.1*abs(xmax), resolution)
    Y = np.linspace(ymin-0.1*abs(ymin), ymax+0.1*abs(ymax), resolution)
    xx, yy = np.meshgrid(X, Y)
    return X, Y, np.dstack((xx, yy))

def plot_contour(dataset, dsetname, clf, fignum):
    plt.figure(fignum)
    # Number of points on each axis of grid
    resolution = 50
    gridx, gridy, grid = generate_grid(dataset)

    # Train classifier
    clf.train(dataset)

    # Predict over grid
    predicted_l = np.empty((resolution, resolution))
    for i in range(resolution):
        predicted_l[i, :] = clf.predict(grid[i])

    # Plot contour + dataset points
    plt.contourf(gridx, gridy, predicted_l)
    zero, one = dataset[dataset[:, 2] == 0], dataset[dataset[:, 2] == 1]
    plt.plot(zero[:, 0], zero[:, 1], 'yo', label='0')
    plt.plot(one[:, 0], one[:, 1], 'g^', label='1')
    xmin, xmax = np.min(gridx), np.max(gridx)
    ymin, ymax = np.min(gridy), np.max(gridy)
    plt.axis([xmin, xmax, ymin, ymax])
    plt.legend(loc='upper left')
    plt.xlabel("x values")
    plt.ylabel("y values")
    if (isinstance(clf, knn.KNN)):
        plt.title("dataset: {}; k = {}".format(dsetname, clf.k))
    else:
        plt.title("dataset: {}".format(dsetname))

def main():
    if len(sys.argv) < 2:
        print("usage: {} dataset [dataset, ...]".format(sys.argv[0]))
        sys.exit(1)

    filenames = sys.argv[1:]
    bestk = [1,6,7,7]
    for i in range(len(filenames)):
        dsetname = os.path.basename(filenames[i])
        dataset = reader.read(filenames[i])
        np.random.shuffle(dataset)
        cv = crossval.CrossValidator()
        ks = []
        errs = []
        for j in range(1, 11):
            clf = knn.KNN(j)
            cv.clf = clf
            err = cv.kfold(dataset)
            print("{} {}".format(j, err))

main()
