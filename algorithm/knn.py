import sys
import base
import numpy as np
from ..util import dataops

class KNN(base.BaseClassifier):
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.train_f = np.array([])
        self.train_l = np.array([])

    def train(self, trainingset):
        self.trained = True
        self.train_f, self.train_l = dataops.fldivide(trainingset)

    def predict(self, testset):
        test_labels = np.empty((len(testset)), dtype=np.int)
        for i in range(len(test_labels)):
            test_labels[i] = self.test_single(testset[i])
        return test_labels

    # Performs a single round
    def test_single(self, test_point):
        # Calculate distances to each of the training points
        distances = np.empty_like(self.train_l, dtype=np.float)
        for i in range(len(distances)):
            distances[i] = np.linalg.norm(test_point - self.train_f[i])
        assert len(self.train_l) == len(distances)
    
        # Catenate distances/labels for sorting
        pairs = np.dstack((distances, self.train_l)).squeeze()
    
        # Take top k points, histogram them and find the most common value
        # NOTE: values should only be [0, 1]
        # argsort() method taken from: http://stackoverflow.com/questions/2828059
        sorted_pairs = pairs[pairs[:, 0].argsort()]
        kpairs = sorted_pairs[:self.k, 1].astype(np.int)
        assert kpairs.shape == (self.k,)
        idx = np.argmax(np.bincount(kpairs))
        assert idx == 0 or idx == 1
    
        return idx

    def error(self, predicted, actual):
        return np.count_nonzero(np.abs(predicted - actual)) / len(predicted)
