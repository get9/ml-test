import sys
import numpy as np
import matplotlib.pyplot as plt

import util

from preprocess.reader import read
from regression.linreg import NormalEquationLinearRegressor, GradientDescentLinearRegressor

# Plot scatter plot of data points as well as regression curve
def plot_scatter_curve(x, y, model, fignum=0, title=''):
    plt.figure(fignum)
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    plt.xlim(0.70*xmin, 1.3*xmax)
    plt.ylim(0.70*ymin, 1.3*ymax)
    plt.scatter(x, y)
    r = np.linspace(xmin, xmax, num=100).reshape((100, 1))
    s = util.make_nth_order(r, len(model)-1).dot(model)
    plt.plot(r, s, 'g-')
    plt.axis([xmin, xmax, ymin, ymax])
    plt.title(title)

def main():
    if len(sys.argv) < 2:
        print("Usage:\n\t{} [trainfile] [testfile]".format(sys.argv[0]))
        sys.exit(1)
    
    train_file, test_file = sys.argv[1:]
    train_data, train_labels = util.fldivide(read(train_file))
    test_data, test_labels   = util.fldivide(read(test_file))
    
    for i in range(5):
        nth_train_data = util.make_nth_order(train_data, i)
        nth_train = np.hstack((nth_train_data, train_labels.reshape((len(train_labels), 1))))
        nth_test_data  = util.make_nth_order(test_data, i)

        model = GradientDescentLinearRegressor(learn_rate=0.4, regularization=1e1)
        model.train(nth_train)
        predicted = model.predict(nth_test_data)

        mse = model.error(predicted, test_labels)

        plot_scatter_curve(test_data, test_labels, model.w, fignum=i,
                title="Gradient Descent, order {}, alpha={}, lambda={}, mse={}".format(i, model.learn_rate, model.l, mse))

    plt.show()

main()
