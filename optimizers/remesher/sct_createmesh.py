import pstereo
import integrate
import z2mesh
import plyutils

from matplotlib import pyplot as plt
import numpy as np
import sys

from algorithms.poisson.integrator import integrate

print("Do not use this script. it hasn't been updated and may break something.")
assert(False)
# Load pre-defined normals from a CSV (exported from MATLAB).
# Use psereo.get_normals(images, lights) to obtain normals using photometric
# stereo.
normals, W, H = pstereo.load_from_csv(sys.argv[1] + "/normal_field");

normals[:, :, 2] = -normals[:, :, 2]

# Integrate normals into a heightfield.
zfield = integrate(normals, 0.0);

# Some stats.
print("zmax: ", np.max(zfield))
print("zmin: ", np.min(zfield))

# Create a mesh from the heightfield.
mesh = z2mesh.z2mesh(-zfield, -1.0, 1.0, -1.0, 1.0)

new_vertices, new_normals, new_indices = mesh

if "--keep-normals" in sys.argv:
    normals = np.flip(normals, axis=1)
    normals[:, :, 2] = -normals[:, :, 2]
    normals = -normals
    normals = normals.reshape((W*H, 3))
else:
    normals = new_normals

# Write the mesh to mesh file.
plyutils.writePLY(sys.argv[2], new_vertices, normals, new_indices)