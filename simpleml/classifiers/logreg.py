import sys

import numpy as np
from scipy.special import expit

from simpleml.util import fldivide, add_feature_bias
from simpleml.optimize.gradient import GradientDescentOptimizer
from simpleml.math import sigmoid, log_likelihood
from simpleml.classifiers.baseclassifier import BaseClassifier


class GradientDescentLogisticRegressor(BaseClassifier):
    def __init__(self, bias=False, learn_rate=None, convergence=None, regularization=None):
        self.ws = np.array([])
        self.bias = bias
        self.learn_rate = learn_rate
        self.convergence = convergence
        self.l = regularization
        self.hfunc = lambda x, y: sigmoid(x @ y)
        self.costfunc = log_likelihood

    def train(self, xs):
        xs, ys = fldivide(xs)
        if self.bias:
            xs = add_feature_bias(xs)
        g = GradientDescentOptimizer(self.hfunc, self.costfunc, convergence=1e-6, \
                learning_rate=0.001, regularization=self.l, max_iters=1e4)
        self.ws = g.optimize(xs, ys)

    def predict(self, xs):
        if self.bias:
            xs = add_feature_bias(xs)
        return np.rint(self.hfunc(xs, self.ws))

    def error(self, predicted, actual):
        return np.count_nonzero(predicted - actual) / len(predicted)
