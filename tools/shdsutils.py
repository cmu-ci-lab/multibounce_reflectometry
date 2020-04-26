import struct
import sys
import numpy as np

# Load a SHDS per-vertex data file.
def loadSHDS(filename):
    f = open(filename, "r")
    alldata = f.read()
    numvals = struct.unpack('iii',alldata[0:12])

    num = numvals[0]
    print ("Items: ", numvals[0])
    print ("Dimensions: ", numvals[1], numvals[2])

    alldata = alldata[12:]
    npdata = np.zeros((numvals[1]*numvals[2], 3))
    for i in range(len(alldata)/(16)):
        vals = struct.unpack('ifff',alldata[i*16:i*16+16])
        npdata[vals[0]] = np.array(vals[1:])

    return npdata