import numpy as np
import cv2
import scipy.ndimage
import math
from remesher.sct_createtarget import normalsToMesh
import os
import OpenEXR
import hdsutils
import Imath
import array
import remesher.plyutils
import matplotlib.pyplot as plt

# Upscaling functions.

def loadEXR(fname, channelNames):
    # Open the input file
    f = OpenEXR.InputFile(fname)

    # Compute the size
    dw = f.header()['dataWindow']
    sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    # Read the three  color channels as 32-bit floats
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    channels = [array.array('f', f.channel(Chan, FLOAT)).tolist() for Chan in channelNames ]

    channels = [ np.array(channel).reshape(sz) for channel in channels ]
    return np.array(channels)

def renderMask(meshfile, W, H, oW=1, oH=1):
    # Re-render mesh to upscale it.
    dirpath = os.path.dirname(__file__)
    exrfile = "/tmp/normals.exr"
    os.system(
        "mitsuba " +\
         dirpath +\
        "/data/normals.xml -o " +\
        exrfile + " -Dmesh=" +\
        meshfile + " -Dwidth=" +\
        format(W) + " -Dheight=" +\
        format(H) + " -DsampleCount=64" +\
        " -DorthoW=" + format(oW) + \
        " -DorthoH=" + format(oH))
    colors = loadEXR(exrfile, ('R','G','B'))

    colors = colors.transpose([1,2,0])
    #plt.imshow(colors)
    #plt.show()

    kernel = np.ones((5,5),np.uint8)

    #mesh = normalsToMesh(normals, directive="none")
    mask = (np.linalg.norm(colors, axis=2, keepdims=False) != 0.0)
    mask = cv2.erode(mask * 1.0, kernel)
    #mask = (np.linalg.norm(normals, axis=2, keepdims=False) == 0.0)
    return mask