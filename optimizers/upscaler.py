import numpy as np
import cv2
import scipy.ndimage
import math
from remesher.sct_createtarget import normalsToMesh
import os
import OpenEXR
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

def rescaleMesh(meshfile, W, H, omeshfile=None, edge_protect=0, mask=None, silent=False):
    # Re-render mesh to upscale it.
    suffix = ""
    if silent:
        suffix = " > /dev/null 2> /dev/null"

    dirpath = os.path.dirname(__file__)
    exrfile = "/tmp/normals.exr"
    os.system("mitsuba " + dirpath + "/data/normals.xml -o " + exrfile + " -Dmesh=" + meshfile + " -Dwidth=" + format(W) + " -Dheight=" + format(H) + " -DsampleCount=64" + " -DorthoW=1 -DorthoH=1 " + suffix)
    colors = loadEXR(exrfile, ('R','G','B'))

    colors = colors.transpose([1,2,0])

    if mask is None:
        mask = (np.linalg.norm(colors, axis=2, keepdims=True) != 0.0)
    else:
        mask = mask[:,:,np.newaxis]

    normals = 2 * colors - 1
    #normals = np.transpose(normals, [1, 2, 0])
    print("RESCALE:", normals.shape)
    W,H,_ = normals.shape
    normals[:,:,2] = normals[:,:,2]
    normals[:,:,1] = normals[:,:,1]
    normals[:,:,0] = normals[:,:,0]

    normals = normals / (np.linalg.norm(normals, axis=2, keepdims=True) + 1e-5)
    print("RESCALE:", normals.shape)
    #plt.imshow(np.squeeze(mask) * 1.0)
    #plt.show()

    mask = np.tile(mask, [1,1,3])
    normals = normals * mask + np.tile(np.array([0,0,1]).reshape([1,1,3]), [normals.shape[0], normals.shape[1], 1]) * (1 - mask)
    print("RESCALE:", normals.shape)

    # Fill empty spaces.
    bad_normals = np.logical_or((np.linalg.norm(normals, axis=2, keepdims=True) > 1.20), (np.linalg.norm(normals, axis=2, keepdims=True) < 0.80))
    bad_normals = np.logical_or(bad_normals,(normals[:,:,2] < 0)[:,:,np.newaxis])
    normals = normals * (1 - bad_normals) + np.tile(np.array([0,0,1]).reshape([1,1,3]), [normals.shape[0], normals.shape[1], 1]) * (bad_normals)
    print("RESCALE:", normals.shape)

    #print(bad_normals.shape)
    #plt.imshow(np.squeeze(bad_normals))
    #plt.show()

    #plt.imshow(np.squeeze(mask) * 1.0 * normals)
    #plt.show()

    mask = np.any(mask, axis=2)
    mesh = normalsToMesh(normals, directive="none", edge_protect=edge_protect, mask=np.squeeze(mask))

    if omeshfile is None:
        omeshfile = meshfile

    # Write the mesh to mesh file.
    remesher.plyutils.writePLY(omeshfile, *mesh)

def downsampleImage(image, factor):
    for i in range(int(math.log(factor,2))):
        scipy.ndimage.gaussian_filter(image, sigma=math.sqrt(2))
        if len(image.shape) == 3:
            image = 0.25 * image[::2,::2,:] +\
                    0.25 * image[1::2,::2,:] +\
                    0.25 * image[::2,1::2,:] +\
                    0.25 * image[1::2,1::2,:]
        elif len(image.shape) == 2:
            image = 0.25 * image[::2,::2] +\
                    0.25 * image[1::2,::2] +\
                    0.25 * image[::2,1::2] +\
                    0.25 * image[1::2,1::2]
        else:
            print("Invalid image dimensions.")
            assert(False)

    return image