import struct
import sys
import numpy as np

# Load a HIST file containing the histogram of sampled angles.
def loadHistogramFileOLD(filename):
    f = open(filename, "r")
    alldata = f.read()
    resolution = struct.unpack('ii',alldata[0:8])

    alldata = alldata[8:]

    npdata = np.zeros((resolution[0], resolution[1], 3))

    k=0
    for i in range(resolution[0]):
        for j in range(resolution[1]):
            vals = struct.unpack('fff',alldata[k*12:k*12+12])
            k+=1
            npdata[i,j,:] = vals

    return npdata

def loadHistogramFile(filename):
    f = open(filename, "r")
    alldata = f.read()
    resolution = struct.unpack('ii',alldata[0:8])
    numVals = struct.unpack('i',alldata[8:12])[0]

    print(resolution[0], "X", resolution[1])
    print(numVals)

    alldata = alldata[12:]

    npdata = np.zeros((7, 7, resolution[0], resolution[1], 2))

    for k in range(numVals):
        vals = struct.unpack('iiiifff',alldata[k*28:k*28+28])
        s, t, hn, hv, wtd, unwtd, _reserved = vals
        if s >= 7 or t >= 7:
            # Path too long.. ignore
            continue

        npdata[s,t,hn,hv,0] = wtd
        npdata[s,t,hn,hv,1] = unwtd

    return npdata