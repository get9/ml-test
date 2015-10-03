import numpy as np
import matplotlib.pyplot as plt
import csv
import sys

from .baseregressor import BaseRegressor

class NormalEquationLinearRegressor(BaseRegressor):
    def __init__(self):
        self.w = np.array([])

    def train(self, x, y):
        self.w = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)

    def predict(self, x):
        return x.dot(self.w)

    def error(self, predicted, actual):
        return np.mean((predicted - actual) ** 2)

class GradientDescentLinearRegressor(BaseRegressor):
    def __init__(self, learn_rate=1.0, convergence=1e-6):
        self.w = np.array([])
        self.learn_rate = learn_rate
        self.convergence = convergence

    def train(self, x, y):
        iters = 0
        m, n = x.shape
        self.w = np.empty(n)
        new_w = np.ones_like(self.w)

        while np.linalg.norm(self.w - new_w) > self.convergence:
            np.copyto(self.w, new_w)
            for j in range(n):
                diff = sum((self.w.dot(x[i]) - y[i]) * x[i, j] for i in range(m))
                new_w[j] = self.w[j] - (self.learn_rate / m) * diff
            iters += 1
        self.w = new_w

    def predict(self, x):
        return x.dot(self.w)

    def error(self, predicted, actual):
        return np.mean((predicted - actual) ** 2)
