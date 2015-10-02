import sys
import numpy as np

from .baseclassifier import BaseClassifier

# Calculate gini impurity of vector {f}
def gini(f):
    return 1.0 - np.sum(np.square(f))

# Score the split given by {splitval} on {dim} according to gini impurity
def score_split(splitval, dim, left, right):
    N = len(left) + len(right)
    fleft = np.bincount(left[:, 2].astype(np.int)) + np.finfo(float).eps
    fleft /= np.sum(fleft)
    fright = np.bincount(right[:, 2].astype(np.int)) + np.finfo(float).eps
    fright /= np.sum(fright)
    return (len(left) * gini(fleft) + len(right) * gini(fright)) / N

# Splits dataset {d} according to {splitval} on {dim}
def split(splitval, dim, d):
    assert dim < len(d[0] - 1)
    return d[d[:, dim] < splitval], d[d[:, dim] >= splitval]


# A node in the decision tree
class DTNode:
    def __init__(self, val, left=None, right=None, feature_idx=-1):
        self.val = val
        self.left = left
        self.right = right
        self.feature_idx = feature_idx

    def __str__(self):
        return "val = {}; feature_idx = {}".format(self.val, self.feature_idx)

# Main decision tree class
class DecisionTree(BaseClassifier):
    def __init__(self, depth=3, nsplits=100):
        super().__init__()
        self.t = None
        self.depth = depth
        self.nsplits = nsplits

    # Main entry point for recursive call that builds the tree
    def train(self, dataset):
        self.t = self.train_internal(dataset, self.depth, list(range(len(dataset[0]) - 1)))

    # Recursive training method. Stops when no more dimensions to condition
    def train_internal(self, dataset, depth, dims):
        # Base case: we're at a leaf, so return whichever label is more common
        labelcount = np.bincount(dataset[:, 2].astype(np.int))
        if depth == 0 or labelcount.max() == len(dataset):
            ret = DTNode(np.argmax(labelcount))
            assert ret.val == 0 or ret.val == 1
            return ret

        best_split_val = None
        best_split_dim = None
        best_left      = np.array([])
        best_right     = np.array([])
        best_score     = 1.0

        # For each dimension, check every possible split and see if it's best
        # When we find the best split after going through all dimensions and
        # all possible splits, remove that dimension from further consideration
        for i in range(len(dims)):
            col = dataset[:, i]
            splits = np.linspace(np.min(col), np.max(col), self.nsplits)
            for j in range(len(splits)):
                left, right = split(splits[j], dims[i], dataset)
                score = score_split(splits[j], dims[i], left, right)
                if score < best_score:
                    best_score     = score
                    best_split_val = splits[j]
                    best_split_dim = dims[i]
                    best_left      = left
                    best_right     = right

        # Recurse
        assert score != 1.0
        left_node  = self.train_internal(best_left, depth-1, dims)
        right_node = self.train_internal(best_right, depth-1, dims)
        return DTNode(best_split_val, left=left_node, right=right_node,
                feature_idx=best_split_dim)

    def print_tree(self):
        self.print_tree_r(self.t)

    def print_tree_r(self, node, indent=''):
        if not node:
            return
        print(indent + str(node))
        self.print_tree_r(node.left, indent+'  ')
        self.print_tree_r(node.right, indent+'  ')

    def predict(self, points):
        predicted_l = np.empty((len(points)), dtype=np.int)
        for i in range(len(points)):
            predicted_l[i] = self.predict_single(points[i])
        return predicted_l

    def predict_single(self, point):
        assert self.t != None
        currnode = self.t
        while currnode.feature_idx > -1:
            if point[currnode.feature_idx] <= currnode.val:
                currnode = currnode.left
            else:
                currnode = currnode.right
        return currnode.val

    def error(self, predicted, actual):
        return np.count_nonzero(np.abs(predicted - actual)) / len(predicted)

