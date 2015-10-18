import sys

import numpy as np
from scipy.special import expit

from simpleml.math import sigmoid


class GradientDescentOptimizer:

    def __init__(self, hfunc, costfunc, learning_rate=1.0, convergence=1e-6,
            regularization=0, max_iters=10e3):
        if not callable(hfunc):
            raise TypeError('hfunc must be callable')
        elif not callable(costfunc):
            raise TypeError('costfunc must be callable')

        self.hfunc = hfunc
        self.costfunc = costfunc
        self.alpha = learning_rate
        self.convergence = convergence
        self.r = regularization
        self.max_iters = max_iters

    def optimize(self, xs, ys):
        m, n = xs.shape
        ws = np.zeros(len(xs[0]))
        iters = 0
        grad = 1

        while iters < self.max_iters: #and np.linalg.norm(sigmoid(xs @ ws) - ys) > self.convergence:
            grad = (expit(xs @ ws) - ys) @ xs + self.r * ws
            ws -= self.alpha * grad
            iters += 1

        print('Gradient descent finished: {} iters, cost = {}'.format(iters, \
                np.linalg.norm(sigmoid(xs @ ws) - ys)))
        print(ws)
        return ws
