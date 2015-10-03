from util import dataops
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
        feats, labels = dataops.fldivide(dataset)
        N = len(feats[0])
        err = np.ones((N, N))
        for i in range(len(feats[0])):
            for j in range(i+1, len(feats[0])):
                f1, f2, l = feats[:, i], feats[:, j], labels
                d = np.dstack((f1, f2, l)).squeeze()
                err[i, j] = self.kfold(d)
        minpos = np.unravel_index(err.argmin(), err.shape)
        return minpos[0], minpos[1], err[minpos[0], minpos[1]]

    # NOTE: Only used with housing data
    def best_3features_topN(self, dataset, n=1, k=10):
        feats, labels = dataops.fldivide(dataset)
        N = len(feats[0])
        err = np.empty((N, N, N))
        err[:] = np.NAN
        for i in range(len(feats[0])):
            for j in range(i+1, len(feats[0])):
                for k in range(j+1, len(feats[0])):
                    f1, f2, f3, l = feats[:, i], feats[:, j], feats[:, k], labels
                    d = np.dstack((f1, f2, f3, l)).squeeze()
                    err[i, j, k] = self.kfold(d)

        # Return indices of top N sets of three features + their error
        indices = []
        errors  = []
        for i in range(n):
            minpos = np.unravel_index(np.nanargmin(err), err.shape)
            indices.append(minpos)
            errors.append(err[minpos])
            err[minpos] = np.NAN
        return indices, errors

