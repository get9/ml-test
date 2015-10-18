import sys
import numpy as np

import util
from regression.linreg import NormalEquationLinearRegressor, GradientDescentLinearRegressor
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
    features = util.basis_expand(features, lambda x: x ** 2, lambda x: x ** 3)
    features = np.hstack([features, np.ones((len(features), 1))])
    dataset = util.fljoin(features, labels)
    
    reg = NormalEquationLinearRegressor(regularization=1e-8)
    cv  = CrossValidator(reg)

    feat_indices, feat_errors = cv.best_3features_topN(dataset, n=5)
    for indices, err in zip(feat_indices, feat_errors):
        bestfeats = np.dstack([features[:, i] for i in indices]).squeeze()
        data = util.fljoin(bestfeats, labels)
        reg.train(data)
        print(reg.w)
        print("indices = {}, err = {}".format(indices, err))

main()
