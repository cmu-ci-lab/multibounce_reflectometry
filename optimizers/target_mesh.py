
import json
from renderoriginals import renderOriginals
from hdsutils import loadHDSImage
from photometric import photometric
from remesher import remesh
from remesher.algorithms.poisson.integrator import integrate
from mask_remesher import remesh as masked_remesh
import remesher.plyutils
import sys
import os
import numpy as np
import remesher.z2mesh
import matplotlib.pyplot as plt

directory = sys.argv[1]
configfile = directory + "/config.json"

config = json.load(open(configfile, "r"))

if not "normals-file" in config["target"]:
    print("No 'target.normals-file' attribute")

normals = np.load(directory + "/" + config["target"]["normals-file"])
W,H = normals.shape[:2]

if "mask" in config["initial-reconstruction"]:
    mask = np.load(directory + "/" + config["initial-reconstruction"]["mask"])
else:
    mask = np.ones((W,H))

allzs = np.stack([np.zeros((W,H)), np.zeros((W,H)), np.ones((W,H))], axis=2)

mask = mask[:,:,np.newaxis]
normals = normals * mask + (1 - mask) * (allzs)
print(normals.shape)

#normals = np.flipud(normals)
#normals = np.fliplr(normals)
normals[:,:,1] = normals[:,:,1]
normals[:,:,2] = -normals[:,:,2]
normals[:,:,0] = -normals[:,:,0]
#plt.imshow(np.concatenate(images, axis=1))
plt.quiver(normals[::3,::3,0], normals[::3,::3,1])
plt.show()
#print(normals[100:120,100:120,:])

# Reconstruct the mesh.

# Integrate normals into a heightfield.
zfield = integrate(normals, 0.0, zflipped=False, mask=np.squeeze(mask))
#zfield = -zfield

# Some stats.
print("zmax: ", np.max(zfield))
print("zmin: ", np.min(zfield))

# Create a mesh from the heightfield.
zfield = np.flipud(zfield)
mesh = remesher.z2mesh.z2mesh(-zfield, -1.0, 1.0, -1.0, 1.0, flip=True)

new_vertices, new_normals, new_indices = mesh

#normals = np.flip(normals, axis=1)
#normals[:, :, 2] = -normals[:, :, 2]
#normals[:, :, 0] = -normals[:, :, 0]
normals[:,:,1] = -normals[:,:,1]
normals = -normals
normals = normals.reshape((W*H, 3))
#new_normals = -new_normals

targetmesh = directory + "/originals/target.ply"
# Write the mesh to mesh file.
remesher.plyutils.writePLY(targetmesh, new_vertices, new_normals, new_indices)
masked_remesh(targetmesh, targetmesh, W, H, mask=np.squeeze(mask), rescale=False)