# Initializes mesh for the given project.

import json
from renderoriginals import renderOriginals
from hdsutils import loadHDSImage
from photometric import photometric
from nlsphotometric import photometric as nls_photometric
from twinphotometric import photometric as twin_photometric
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
lightsfile = directory + "/" + config["initial-reconstruction"]["lights"]
intensitiesfile = directory + "/" + config["initial-reconstruction"]["light-intensities"]

lights = open(lightsfile, "r").readlines()
lts = [ll.split(" ")[1:4] for ll in lights]
flts = [ [float(k) for k in lt] for lt in lts ]

intensities = open(intensitiesfile, "r").readlines()
its = [float(li) for li in intensities]

irType = config["initial-reconstruction"]["type"]
if irType == "render":
    # Render the original images first.
    numLights = renderOriginals(directory, config, extension="hds", fileFormat="hds", samples=64)

    # Load the HDS images.
    W,H = loadHDSImage(directory + "/originals/" + format(0).zfill(4) + ".hds").squeeze().shape
    print("W,H:", W,H)

    images = np.zeros((numLights,W,H))

    for i in range(numLights):
        images[i,:,:] = loadHDSImage(directory + "/originals/" + format(i).zfill(4) + ".hds").squeeze()

elif irType == "file":
    # Load images straight from file.
    images = np.load(directory + "/" + config["initial-reconstruction"]["file"])
    W,H = images.shape[:2]
    images = images.transpose([2,0,1])
else:
    print("Invalid reconstruction type: ", irType)
    sys.exit(1)

if "mask" in config["initial-reconstruction"]:
    mask = np.load(directory + "/" + config["initial-reconstruction"]["mask"])
else:
    mask = np.ones((W,H))

pmfunction = None
if ("recalibrate-lights" in config["initial-reconstruction"]) and config["initial-reconstruction"]["recalibrate-lights"]:
    # Recalibrate light intensities using non-linear least squares.
    pmfunction = twin_photometric
else:
    pmfunction = photometric


lights = np.array(flts) * np.array(its)[:, np.newaxis]

olights = np.array(lights)

# TODO: WARN: Change back.
lights[:,0] = olights[:,0]
lights[:,1] = olights[:,1]
lights[:,2] = olights[:,2]

# NxWxH
normals, intensities = pmfunction(images, lights)
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

normals = np.flip(normals, axis=0)
#normals[:, :, 2] = -normals[:, :, 2]
#normals[:, :, 0] = -normals[:, :, 0]
normals[:,:,1] = -normals[:,:,1]
normals = normals
normals = normals.reshape((W*H, 3))
#new_normals = -new_normals

targetmesh = directory + "/originals/photometric-unmasked.ply"
maskedmesh = directory + "/originals/photometric.ply"
# Write the mesh to mesh file.
remesher.plyutils.writePLY(targetmesh, new_vertices, normals, new_indices)
masked_remesh(targetmesh, maskedmesh, W, H, mask=np.squeeze(mask), rescale=False)