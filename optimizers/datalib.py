import os
import numpy as np

def writeNumpyData(data, filename):
    # Writing HDS output.
    np.save(filename, data)