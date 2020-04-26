# Render a depth image and create a target mesh.

# Render normals and reconstruct.
import sys
import os
import OpenEXR
import numpy as np
import Imath
import array

import remesh
import z2mesh
#from integrate import integrate
import plyutils

from mpl_toolkits.mplot3d import Axes3D

from algorithms.poisson.integrator import integrate

import matplotlib.pyplot as plt


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

def normalsToMesh(normals, directive=None, edge_protect=0, mask=None):
    onormals = np.array(normals)
    onormals = np.flip(onormals, 0)
    #onormals = np.flip(onormals, 1)
    normals = np.array(normals)

    normals[:, :, 0] = -onormals[:, :, 0]
    normals[:, :, 1] = -onormals[:, :, 1]
    normals[:, :, 2] = -onormals[:, :, 2]

    nans = (np.abs(normals) == np.array([0,0,0])).all(axis=2, keepdims=True)
    nans = np.tile(nans,[1, 1, 3])

    infs = (np.linalg.norm(normals, axis=2, keepdims=True) > 2)

    if mask is None:
        mask = (np.abs(normals) == np.array([1,1,1])).all(axis=2)
    else:
        mask = mask[::-1,:]
    #plt.imshow(np.abs(infs[:,:,0])*1.0)
    #plt.show()

    #print(normals[0:10, 0:10, :])
    print normals.shape
    W,H,_ = normals.shape
    print("Dims: ", W, H)

    for x in range(0, W):
        for y in range(0, H):
            if abs(normals[x,y,1]) > 1000:
                normals[x,y,:] = np.array([0, 0, -1])

    normals = normals * (1 - nans) + np.tile(np.array([0, 0, -1]).reshape([1,1,3]), [W, H, 1]) * nans
    normals = normals * (1 - infs) + np.tile(np.array([0, 0, -1]).reshape([1,1,3]), [W, H, 1]) * infs
    #plt.quiver(normals[::3,::3,0], normals[::3,::3,1], scale=160.0, width=0.0005)
    #plt.show()

    # Invert normals if requested.
    if directive == "invert":
        normals[:,:,0] = -normals[:,:,0]
        normals[:,:,1] = -normals[:,:,1]

    if np.sum(normals[:,:,2]) > 0:
        zflipped = False
    else:
        zflipped = True

    # Integrate normals into a heightfield.
    zfield = integrate(normals, 0.0, zflipped=zflipped, edge_protect=edge_protect, mask=mask)
    zfield = -zfield

    # Some stats.
    print("zmax: ", np.max(zfield))
    print("zmin: ", np.min(zfield))


    # Create a mesh from the heightfield.
    mesh = z2mesh.z2mesh(zfield, -1.0, 1.0, -1.0, 1.0)

    new_vertices, new_normals, new_indices = mesh

    normals[:, :, 2] = -normals[:, :, 2]
    normals[:, :, 1] = -normals[:, :, 1]
    normals[:, :, 0] = -normals[:, :, 0]
    normals = normals.reshape((W*H, 3))

    print("normals: ", np.sum(normals[:,2]))

    return new_vertices, normals, new_indices

if __name__ == "__main__":
    directory = sys.argv[1]
    if len(sys.argv) > 2:
        directive = sys.argv[2]
    else:
        directive = None

    os.system("mitsuba " + directory + "/originals/normals-scene.xml -o " + directory + "/originals/normals.exr -DsampleCount=128")

    colors = loadEXR(directory + "/originals/normals.exr", ("R", "G", "B"))

    normals = 2 * colors - 1
    normals = np.transpose(normals, [1, 2, 0])
    print(normals.shape)
    W,H,_ = normals.shape

    mesh = normalsToMesh(normals, directive=directive)

    # Write the mesh to mesh file.
    plyutils.writePLY(directory + "/originals/targetmesh.ply", *mesh)