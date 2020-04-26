# Poisson surface integrator, based on finding least squares solution to matching gradients.
import os
import sys

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
import scipy.sparse.linalg
import matplotlib.pyplot as plt
from adjacency import MeshAdjacencyBuilder


# Convert a normal list to a WxH map.
"""
def normalListToMap(vertices, normals, W=256, H=256):
    indexMap = MeshAdjacencyBuilder.buildIndexMap(vertices, W=W, H=H)
    normalMap = np.stack([
            np.zeros_like((W,H)),
            np.zeros_like((W,H)),
            np.ones_like((W,H))
        ], axis=2)
    normalMap = vertices[indexMap, :] * (validMask) + normalMap * (1 - validMask)
    return normalMap
"""

def normalsFromField(field):
    # Compute normals from a given height field.
    # TODO: Finish this.
    # How to handle edges? They'll be invalid along the edges because of the 0s outside the field.
    # Maybe set that area to NaN?
    # Could just ignore it.
    # Let's ignore it for now.

    W,H = field.shape

    diffY = (field[2:,:] - field[:-2,:])[:,1:-1] * (W/4)
    diffX = (field[:,2:] - field[:,:-2])[1:-1,:] * (H/4)

    diffX = np.pad(diffX, ((1,1),(1,1)), mode="reflect")
    diffY = np.pad(diffY, ((1,1),(1,1)), mode="reflect")
    diffZ = np.ones_like(field)

    denormals = np.stack([diffX, diffY, diffZ], axis=2)
    #plt.imshow(denormals)
    #plt.show()

    normals = denormals / np.linalg.norm(denormals, keepdims=True, axis=2)

    return normals

def integrate(normals, mean=0.0, zcutoff=0.05, zflipped=True, edge_protect=0, mask=None):
    # Get W, H coordinates
    dims = normals.shape
    W = dims[0]
    H = dims[1]

    # Sub in mask if not available.
    if mask is None:
        mask = np.ones((W,H))

    # Reshape top WxHx3
    normals = normals.reshape((W, H, 3))

    if zflipped:
        badnormals = (normals[:, :, 2] > -zcutoff)
        normals[:, :, 2] = badnormals * -zcutoff + (1 - badnormals) * normals[:, :, 2]
    else:
        badnormals = (normals[:, :, 2] < zcutoff)
        normals[:, :, 2] = badnormals * zcutoff + (1 - badnormals) * normals[:, :, 2]

    # Find differential
    zx = np.nan_to_num(np.divide(normals[:, :, 0], normals[:, :, 2]))
    zy = np.nan_to_num(np.divide(normals[:, :, 1], normals[:, :, 2]))
    zones = np.ones_like(zx)

    #plt.imshow(np.stack([zx, zy, zones], axis=2))
    #plt.show()

    # Solve a poisson least squares setup that attempts to find a solution to Zs that best fit the
    # given gradients.
    vcount = zx.shape[0] * zx.shape[1]

    ccount = ((W-1) * (H-1))*2 + 1 + 4 + 4

    if edge_protect != 0:
        zx[:edge_protect, :]  = 0
        zx[:, :edge_protect]  = 0
        zx[:, -edge_protect:] = 0
        zx[-edge_protect:, :] = 0

        zy[:edge_protect, :]  = 0
        zy[:, :edge_protect]  = 0
        zy[:, -edge_protect:] = 0
        zy[-edge_protect:, :] = 0

    _zx = zx[0:W-1, 0:H-1]
    _zy = zy[0:W-1, 0:H-1]

    C = lil_matrix((ccount, vcount))
    wts = lil_matrix((ccount, ccount))
    Y = np.concatenate([_zx.reshape([(W-1)*(H-1)])/W, _zy.reshape([(W-1)*(H-1)])/H, np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])], axis=0)

    # TODO: Investigate later.
    #mask = mask[::-1,:]

    #plt.imshow(normals)
    #plt.show()
    #plt.imshow(mask * 1.0)
    #plt.show()

    constraint = 0
    # Construct Gx.
    for y in range(0, H-1):
        for x in range(0, W-1):
            wts[constraint, constraint] = np.abs(normals[y, x, 2]) * (mask[y, x])
            C[constraint, (x) + y*W] = (-1/2.0) * wts[constraint, constraint]
            C[constraint, (x+1) + y*W] = (+1/2.0) * wts[constraint, constraint]
            Y[constraint] = ((zx[y, x] + zx[y, x+1]) / 2) / W
            #if(Y[constraint] > 5) :
            #    print("Y: ", Y[constraint])

            constraint += 1
            #if constraint % 100 == 0:
            #    print("\rAdding constraints: " + format(constraint) + "/" + format(ccount) + "\r")

    # Construct Gy.
    for y in range(0, H-1):
        for x in range(0, W-1):
            wts[constraint, constraint] = np.abs(normals[y, x, 2]) * (mask[y, x])
            C[constraint, x + (y)*W] = (-1/2.0) * wts[constraint, constraint]
            C[constraint, x + (y+1)*W] = (+1/2.0) * wts[constraint, constraint]
            Y[constraint] = ((zy[y, x] + zy[y+1, x]) / 2) / H

            constraint += 1
            #if constraint % 100 == 0:
            #    print("\rAdding constraints: " + format(constraint) + "/" + format(ccount) + "\r")

    # Fix average.
    C[constraint, :] = np.ones((vcount)) / vcount

    # Fix end-points.
    constraint += 1
    C[constraint, 0] = +0
    C[constraint, 1] = -0
    Y[constraint] = 0
    wts[constraint, constraint] = 1

    constraint += 1
    C[constraint, H-1] = +0
    C[constraint, H-2] = -0
    Y[constraint] = 0
    wts[constraint, constraint] = 1

    constraint += 1
    C[constraint, (H-1)*W + H-1] = +0
    C[constraint, (H-1)*W + H-2] = -0
    Y[constraint] = 0
    wts[constraint, constraint] = 1

    constraint += 1
    C[constraint, (H-1)*W + 0] = +0
    C[constraint, (H-1)*W + 1] = -0
    Y[constraint] = 0
    wts[constraint, constraint] = 1

    checkerboardWt = 0
    # Add checkerboard constraints.
    constraint += 1
    for y in range(0, H, 2):
        for x in range(0, W, 2):
            wts[constraint, constraint] = checkerboardWt
            C[constraint, x + y*W] = checkerboardWt
            C[constraint, x + (y+1)*W] = -checkerboardWt
    Y[constraint] = 0

    constraint += 1
    for y in range(1, H, 2):
        for x in range(0, W, 2):
            wts[constraint, constraint] = checkerboardWt
            C[constraint, x + y*W] = checkerboardWt
            C[constraint, (x+1) + y*W] = -checkerboardWt
    Y[constraint] = 0

    constraint += 1
    for y in range(0, H, 2):
        for x in range(1, W, 2):
            wts[constraint, constraint] = checkerboardWt
            C[constraint, x + y*W] = checkerboardWt
            C[constraint, (x-1) + y*W] = -checkerboardWt
    Y[constraint] = 0

    constraint += 1
    for y in range(1, H, 2):
        for x in range(1, W, 2):
            wts[constraint, constraint] = checkerboardWt
            C[constraint, x + y*W] = checkerboardWt
            C[constraint, x + (y-1)*W] = -checkerboardWt
    Y[constraint] = 0

    C = C.tocsr()
    Y = wts * Y

    Z, lstop, itn, r1norm, r2norm, anorm, acound, arnorm, xnorm, cvar = scipy.sparse.linalg.lsqr(C, Y)
    Z = Z.reshape([W,H])

    return Z