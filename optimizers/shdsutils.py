import struct
import sys
import numpy as np

# Load a SHDS per-vertex data file.
def loadSHDS(filename, numWeights=0, tabularSize=(0,0,0)):
    f = open(filename, "r")
    alldata = f.read()
    numvals = struct.unpack('iii',alldata[0:12])

    num = numvals[0]
    #print ("Items: ", numvals[0])
    #print ("Dimensions: ", numvals[1], numvals[2])

    alldata = alldata[12:]
    npdataCount = len(alldata)/16
    #print ("Total elements: ", npdataCount)
    #print ("Excess elements: ",  npdataCount - numvals[0])

    ddata = {}
    if numWeights > 0:
        wtdata = np.zeros((numWeights,))
    else:
        wtdata = None
    
    if tabularSize != (0,0,0):
        tabdata = np.zeros(tabularSize)
    else:
        tabdata = None

    maxelement = 0
    numelements = 0
    numTabularIndices = tabularSize[0] * tabularSize[1] * tabularSize[2]
    for i in range(len(alldata)/(16)):
        vals = struct.unpack('ifff',alldata[i*16:i*16+16])
        if vals[0] < numWeights:
            wtdata[vals[0]] = vals[1]
        elif vals[0] < numTabularIndices + numWeights:
            tabdata[np.unravel_index([vals[0] - numWeights], tabularSize)] = np.array([vals[1]])
        else:
            idx = vals[0] - (numWeights + numTabularIndices)
            maxelement = (idx) if ((idx) > maxelement) else maxelement
            ddata[idx] = np.array(vals[1:])
        numelements += 1
    #print("Elements read: ", numelements)

    npdata = np.zeros((maxelement+1, 3))
    for k in ddata:
        npdata[k,:] = ddata[k]

    return npdata, wtdata, tabdata