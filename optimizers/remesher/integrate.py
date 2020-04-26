import os
import sys

# Local module imports
import plyutils
import frankot

import numpy as np
#nfile = sys.argv[1];

#print("Opening " + nfile + " to read normals");

# Load normalfield.

#vertices, normals, indices = plyutils.readPLY(nfile);

def integrate(normals, mean=0.0, zcutoff=0.05):
    # Get W, H coordinates
    dims = normals.shape;
    W = dims[0];
    H = dims[1];

    # Reshape top WxHx3
    normals = normals.reshape((W, H, 3));

    badnormals = (normals[:, :, 2] > -zcutoff)
    normals[:, :, 2] = badnormals * -zcutoff + (1 - badnormals) * normals[:, :, 2]

    # Find differential
    zx = np.nan_to_num(np.divide(normals[:, :, 0], normals[:, :, 2]));
    zy = np.nan_to_num(np.divide(normals[:, :, 1], normals[:, :, 2]));

    # Mirror the vector to remove periodic constraint.
    mx = np.concatenate( [np.concatenate([zx, -zx[:,-1::-1]], axis=1), 
                    np.concatenate([zx[-1::-1,:],-zx[-1::-1,-1::-1]], axis=1)], axis=0);
    my = np.concatenate( [np.concatenate([zy, zy[:,-1::-1]], axis=1), 
                    np.concatenate([-zy[-1::-1,:],-zy[-1::-1,-1::-1]], axis=1)], axis=0);
    sz = mx.shape

    assert(sz[0] == W * 2 and sz[1] == H * 2);
    z = frankot.project_surface(mx, my);

    # Slice the first quarter
    z = z[:sz[0]/2, :sz[1]/2];
    assert(z.shape[0] == W and z.shape[1] == H);

    # Readjust z values to fit the average.
    newMeanZ = z.mean(axis=0).mean(axis=0);
    oldMeanZ = mean;

    z2 = (oldMeanZ - newMeanZ) + z;

    return z2 / (W/2);

"""# Readjust the points.
vertices[:, 2] = z2.reshape((W*H, 1));

num_samples = np.zeros(normals[:, :, 1]);
# Recompute the normals.
# Loop through indices and update the normals.

# first, put the normals back into W*H form
normals = normals.reshape((W*H, 3));
normalcounts = np.zeros(W*H);

for t in indices:
    n = cross(vertices[t[0]] - vertices[t[1]], vertices[t[2]] - vertices[t[1]]);
    n.normalize();
    nc0 = normalcounts[t[0]];
    nc1 = normalcounts[t[1]];
    nc2 = normalcounts[t[2]];
    normals[t[0]] = (normals[t[0]] * nc0 + n * 1) / (nc0 + 1);
    normals[t[1]] = (normals[t[1]] * nc1 + n * 1) / (nc1 + 1);
    normals[t[2]] = (normals[t[2]] * nc2 + n * 1) / (nc2 + 1);
    normalcounts[t[0]] += 1;
    normalcounts[t[1]] += 1;
    normalcounts[t[2]] += 1;

# Create the new mesh
plyutils.writePLY(normals, vertices, indices);
"""
