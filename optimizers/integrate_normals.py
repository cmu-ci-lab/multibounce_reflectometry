import optparse
from remesher.algorithms.poisson.integrator import integrate
from mask_remesher import remesh as masked_remesh
import remesher.z2mesh
import numpy as np

parser = optparse.OptionParser()
parser.add_option("-o", "--output-file", dest="outputFile", type="str", default=None)
parser.add_option("-m", "--mask-file", dest="maskFile", type="str", default=None)
parser.add_option("-p", "--post-transforms", dest="postTransforms", action="store_true", default=False)
(options, args) = parser.parse_args()

normalsfile = args[0]
normals = np.load(normalsfile)
mask = np.load(options.maskFile)

if options.postTransforms:
    normals[:,:,0] = -normals[:,:,0]

# Integrate normals into a heightfield.
zfield = integrate(normals, 0.0, zflipped=False, mask=np.squeeze(mask))
#zfield = -zfield

# Some stats.
print("zmax: ", np.max(zfield))
print("zmin: ", np.min(zfield))

# Create a mesh from the heightfield.
zfield = np.flipud(zfield)
mesh = remesher.z2mesh.z2mesh(zfield, -1.0, 1.0, -1.0, 1.0, flip=False)

new_vertices, new_normals, new_indices = mesh
W,H,_ = normals.shape

normals = np.flip(normals, axis=0)
#normals[:, :, 2] = -normals[:, :, 2]
normals[:, :, 0] = -normals[:, :, 0]
#normals[:,:,1] = -normals[:,:,1]
normals = normals
normals = normals.reshape((normals.shape[0]*normals.shape[1], 3))
#new_normals = -new_normals

targetmesh = "/tmp/temporary.ply"
maskedmesh = options.outputFile

# Write the mesh to mesh file.
remesher.plyutils.writePLY(targetmesh, new_vertices, normals, new_indices)
masked_remesh(targetmesh, maskedmesh, W, H, mask=np.squeeze(mask), rescale=False)