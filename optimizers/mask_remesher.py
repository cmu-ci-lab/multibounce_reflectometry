# Masked remdsher.
from upscaler import rescaleMesh
from remesher import plyutils
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt

def toXY(i, W, H):
    return (H - 1) - (int(i) / W), (int(i) % W)

def remesh(imeshfile, omeshfile, W, H, mask, edge_protect=0, rescale=True, tempfile=None):

    if tempfile is None:
        tempfile = omeshfile

    if rescale:
        # Rerender the mesh using those files.
        rescaleMesh(imeshfile, W, H, tempfile, edge_protect=edge_protect, mask=mask)

    if rescale:
        # Reload mesh.
        vertices, normals, indices = plyutils.readPLY(tempfile)
    else:
        vertices, normals, indices = plyutils.readPLY(imeshfile)

    #normals = normals.reshape((W,H,3))
    #vertices = vertices.reshape((W,H,3))

    new_vertices = []
    new_normals = []

    new_indices = []

    rejected = []
    index_remap = []

    #plt.imshow(mask * 256)
    #plt.show()

    for i, vn in enumerate(zip(vertices,normals)):
        v, n = vn
        x,y = toXY(i, W, H)

        if mask[x,y] > 0:
            index_remap.append(len(new_vertices))
            new_vertices.append(v)
            new_normals.append(n)
        else:
            index_remap.append(-1)
            rejected.append(i)

    for idxs in indices:
        reject = False
        for idx in idxs:
            #print idxs
            x,y = toXY(idx, W, H)
            #print x,y
            #print type(x), type(y)
            #print mask.shape
            if mask[x,y] == 0:
                reject = True
                break

        if not reject:
            assert(index_remap[int(idx)] != -1)
            new_indices.append([ index_remap[int(idx)] for idx in idxs ])

    plyutils.writePLY(omeshfile, np.array(new_vertices), np.array(new_normals), np.array(new_indices))