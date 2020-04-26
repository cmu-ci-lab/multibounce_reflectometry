# MERL I/O
import struct
import os
import sys
import numpy as np

# Write out into .binary (MERL format)
def merl_write(filename, bsdf):
    # TODO: Add stuff here.
    dims = bsdf.shape

    binary_data = ""
    binary_data += struct.pack('iii', dims[0], dims[1], dims[2])

    bsdf = bsdf.reshape((dims[0] * dims[1] * dims[2],))
    bsdf = np.tile(bsdf, (3,))
    for b in bsdf:
        binary_data += struct.pack('d', b)

    f = open(filename, "wb")
    f.write(binary_data)

def merl_read_raw(filename):
    f = open(filename, "r")
    alldata = f.read()
    resolution = struct.unpack('iii',alldata[0:12])

    print ("Resolution: ", resolution)

    alldata = alldata[12:]
    print ("Items: ", len(alldata)/(8))
    npdata = np.zeros((resolution[0] * resolution[1] * resolution[2] * 3,))
    for i in range(len(alldata)/(8)):
        vals = struct.unpack('d',alldata[i*8:i*8+8])
        npdata[i] = vals[0]

    return npdata

def merl_read(filename):
    f = open(filename, "r")
    alldata = f.read()
    resolution = struct.unpack('iii',alldata[0:12])

    print ("Resolution: ", resolution)

    alldata = alldata[12:]
    print ("Items: ", len(alldata)/(8))
    npdata = np.zeros((resolution[0] * resolution[1] * resolution[2] * 3,))
    for i in range(len(alldata)/(8)):
        vals = struct.unpack('d',alldata[i*8:i*8+8])
        npdata[i] = vals[0]
        #if npdata[i] < 0:
            #print("FATAL: BRDF values cannot be negative: ", npdata[i])
            #sys.exit(1)
    
    print(npdata[0])

    npdata = npdata.reshape((3, resolution[0], resolution[1], resolution[2]))
    npdata = npdata[0,:,:,:]
    #npdata = np.mean(npdata, axis=0)

    return npdata