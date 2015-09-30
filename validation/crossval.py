from ..util import dataops
import numpy as np

class CrossValidator:
    def __init__(self, clf=None):
        self.clf = clf

    def kfold(self, dataset, k=10):
        # Divide data into k-groups
        kgroups = dataops.kdivide(dataset, k)
        err = []
        for i in range(len(kgroups)):
            # Catenate testing samples (leave out group i)
            train_set = np.vstack(kgroups[:i] + kgroups[i+1:])
            test_set = kgroups[i]
            
            # Run classifier, test with test set
            self.clf.train(train_set)
            test_f, test_l = dataops.fldivide(test_set)
            predicted_l = self.clf.predict(test_f)
            err.append(self.clf.error(predicted_l, test_l))

        # Return mean error from all k-groups
        return sum(err) / len(err)

    # NOTE: Only used with housing data
    def best_2features(self, dataset, k=10):
        d_f, d_l = dataops.fldivide(dataset)
        N = len(d_f[0])
        err = np.ones((N, N))
        for i in range(len(d_f[0])):
            for j in range(i+1, len(d_f[0])):
                f1, f2, l = d_f[:, i], d_f[:, j], d_l
                d = np.dstack((f1, f2, l)).squeeze()
                err[i, j] = self.kfold(d)
        minpos = np.unravel_index(err.argmin(), err.shape)
        return minpos[0], minpos[1], err[minpos[0], minpos[1]]
