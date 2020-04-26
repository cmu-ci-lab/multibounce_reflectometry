import struct
import numpy as np
import time

def loadHDSImage(filename):
    data = open(filename, "r").read()

    (width, height, channels) = struct.unpack('iii', data[0:12])

    data = data[12:]

    npdata = np.zeros((width,height,channels), dtype=np.float32)

    k = 0
    for x in range(width):
        for y in range(height):
            for z in range(channels):
                npdata[x,y,z] = (struct.unpack('f', data[(k*4) : (k*4 + 4)]))[0]
                k += 1

    return npdata

def writeHDSImage(filename, width, height, channels, npdata):
    fstream = open(filename, "w")

    whcheader = struct.pack('iii', width, height, channels)
    fstream.write(whcheader)

    k = 0
    for x in range(width):
        for y in range(height):
            for z in range(channels):
                fstream.write(struct.pack('f', npdata[x,y,z]))
                k += 1

    return npdata