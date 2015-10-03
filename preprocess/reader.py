import numpy as np
import csv

# Returns (testpoints, labels) as np arrays
def read(filename):
    with open(filename, 'r') as f:
        observations = []
        reader = csv.reader(f)
        for line in reader:
            observations.append(list(map(float, line)))
    return np.array(observations)
