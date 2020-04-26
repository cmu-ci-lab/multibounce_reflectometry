import struct
import sys
import numpy as np

# Load a RAW tensor file
def loadRAW(filename, numSlices=0):
    f = open(filename, "r")
    alldata = f.read()
    numvals = struct.unpack('iii',alldata[0:12])

    num = numvals[0]
    print ("Items: ", numvals[0])
    print ("Dimensions: ", numvals[1], numvals[2])

    alldata = alldata[12:]
    npdataCount = len(alldata)/24
    print ("Total elements: ", npdataCount)
    print ("Excess elements: ",  npdataCount - numvals[0])

    data = np.zeros((numSlices, numvals[1], numvals[2]))

    maxelement = 0
    numelements = 0
    for i in range(len(alldata)/(24)):
        vals = struct.unpack('iiifff',alldata[i*24:i*24+24])
        if vals[2] < numSlices:
            data[vals[2], vals[1], vals[0]] = vals[3]

        numelements += 1
    print("Elements read: ", numelements)

    return data