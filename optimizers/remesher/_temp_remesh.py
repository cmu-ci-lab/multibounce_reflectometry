import pstereo
import integrate
import z2mesh
import plyutils
import math
import sys

from matplotlib import pyplot as plt
import numpy as np

def remesh(meshfile, omeshfile, keep_normals=False):
    # Load pre-defined normals from a CSV (exported from MATLAB).
    # Use psereo.get_normals(images, lights) to obtain normals using photometric
    # stereo.
    (vertices, normals, indices) = plyutils.readPLY(meshfile)
    old_normals = np.array(normals)
    print("OLD: ", old_normals[0, :])

    print(normals.shape)
    w = int(math.sqrt(normals.shape[0]))
    h = w
    normals = normals.reshape((w,h,3))
    #normals[:, :, 2] = -normals[:, :, 2]

    # Integrate normals into a heightfield.
    zfield = integrate.integrate(normals, 0.0)

    # Some stats.
    print("zmax: ", np.max(zfield))
    print("zmin: ", np.min(zfield))

    # Create a mesh from the heightfield.
    mesh = z2mesh.z2mesh(zfield, -1.0, 1.0, -1.0, 1.0)

    # Expand mesh components.
    new_vertices, new_normals, new_indices = mesh
    new_vertices[:, 2] = -new_vertices[:, 2]

    if keep_normals:
        #old_normals[:, :, 2] = -old_normals[:, :, 2]
        #old_normals[:, 2] = -old_normals[:, 2]
        new_normals = old_normals

    print("NEW: ", new_normals[0, :])
    # Write the mesh to mesh file.
    plyutils.writePLY(omeshfile, new_vertices, new_normals, new_indices)