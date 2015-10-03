import numpy as np
import csv

# Returns (testpoints, labels) as np arrays
def read(filename, delim=','):
    with open(filename, 'r') as f:
        observations = []
        reader = csv.reader(f, delimiter=delim, skipinitialspace=True)
        for line in reader:
            observations.append(list(map(float, line)))
    return np.array(observations)
