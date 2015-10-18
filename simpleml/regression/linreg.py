import numpy as np
import matplotlib.pyplot as plt
import csv
import sys

import util
from .baseregressor import BaseRegressor

class NormalEquationLinearRegressor(BaseRegressor):
    def __init__(self, regularization=0):
        self.w = np.array([])
        self.l = regularization

    def train(self, x):
        x, y = util.fldivide(x)
        
        # Check if any columns of x are all zeros (bad news for inversion)
        try:
            self.w = np.linalg.inv(x.T.dot(x) + self.l * np.eye(len(x[0]))).dot(x.T).dot(y)
        except np.linalg.linalg.LinAlgError as e:
            print(e)
            self.w = np.empty(len(x[0]))
            self.w[:] = np.NAN

    def predict(self, x):
        return x.dot(self.w)

    def error(self, predicted, actual):
        return np.mean((predicted - actual) ** 2)

class GradientDescentLinearRegressor(BaseRegressor):
    def __init__(self, learn_rate=1.0, convergence=1e-6, regularization=0):
        self.w = np.array([])
        self.learn_rate = learn_rate
        self.convergence = convergence
        self.l = regularization

    def train(self, x):
        x, y = util.fldivide(x)
        iters = 0
        m, n = x.shape
        self.w = np.empty(n)
        new_w = np.ones_like(self.w)

        while np.linalg.norm(self.w - new_w) > self.convergence:
            np.copyto(self.w, new_w)
            for j in range(n):
                diff = sum((self.w.dot(x[i]) - y[i]) * x[i, j] for i in range(m))
                new_w[j] = self.w[j] - (self.learn_rate / m) * (diff + self.l * self.w[j])
            iters += 1
        self.w = new_w

    def predict(self, x):
        return x.dot(self.w)

    def error(self, predicted, actual):
        return np.mean((predicted - actual) ** 2)
