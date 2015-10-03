import sys
import numpy as np

import util
from regression.linreg import NormalEquationLinearRegressor
from preprocess import reader, scaling
from validation import CrossValidator

def main():
    if len(sys.argv) < 2:
        print("Usage:\n\t{} [housing-data]".format(sys.argv[0]))
        sys.exit(1)

    dataset = reader.read(sys.argv[1], delim=' ')

    # Exapand features with nonlinear functions
    # Need to put them back together to handle 
    features, labels = util.fldivide(dataset)
    features, scale = scaling.unit_scale(features)
    features = util.basis_expand(features, lambda x: x ** 2, lambda x: np.exp(x))
    dataset = np.hstack([features, labels.reshape((len(labels), 1))])
    
    reg = NormalEquationLinearRegressor()
    cv  = CrossValidator(reg)

    feat_indices, feat_errors = cv.best_3features_topN(dataset, n=5)
    for indices, err in zip(feat_indices, feat_errors):
        print("indices = {}, err = {}".format(indices, err))

main()
