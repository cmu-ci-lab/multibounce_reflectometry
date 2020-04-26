import sys
import os
import matplotlib.pyplot as plt
from matplotlib import pylab
import load_normals
import json
import cv2
import numpy as np

directory = sys.argv[1]
parameters = json.load(open(directory + "/inputs/config.json", "r"))

def plotNormals(vertices, normals, color='k'):
    # Select a slice of vertices.
    shortlist = []
    for vertex, normal in zip(vertices, normals):
        if vertex[1] > -0.002 and vertex[1] < +0.002:
            shortlist.append((vertex, normal))

    for vertex, normal in shortlist[::2]:
        plt.plot([vertex[0], vertex[0] + normal[0]*0.04], [vertex[2], vertex[2] + normal[2]*0.04], color=color, linestyle='-', linewidth=0.1)


def _polarizeData(data):
        positives = np.clip(data, 0, 255)
        negatives = np.clip(data, -255, 0)
        return np.concatenate((positives[..., np.newaxis], np.zeros(positives.shape)[..., np.newaxis], -negatives[..., np.newaxis]), axis=2).astype('uint8')


# Render normals map and mesh image.
def renderMeshAndMap(normalfilename, meshviewfilename, meshfile):
    dirpath = os.path.dirname(__file__)
    print("Rendering mesh ", meshfile)
    os.system("mitsuba " + dirpath + "/xml/normals.xml -o \"" + normalfilename + "\" -Dmesh=\"" + meshfile + "\" -Dwidth=512 -Dheight=512 -DsampleCount=4 > /dev/null")
    os.system("mitsuba " + dirpath + "/xml/meshview.xml -o \"" + meshviewfilename + "\" -Dmesh=\"" + meshfile + "\" -Dwidth=512 -Dheight=512 -DsampleCount=128 > /dev/null")


def plot(filenames):
    referenceNormals = None
    referenceVertices = None
    for idx, filename in enumerate(filenames):
        # TODO: Likely performance issues with this statement.
        vertices = load_normals.load_vertices(filename)
        normals = load_normals.load_normals(filename)

        if idx == 0:
            referenceVertices = vertices
            referenceNormals = normals

        shortlist = []
        for vertex, normal in zip(vertices, normals):
            if vertex[1] > -0.002 and vertex[1] < +0.002:
                shortlist.append((vertex, normal))

        xs = []
        ys = []

        for vertex, normal in shortlist:
            xs.append(vertex[0])
            ys.append(vertex[2])

        if not idx == 0:
            plotNormals(vertices, normals, color='C' + format(idx))
            plotNormals(vertices, referenceNormals, color='C0')

        plt.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False,
        right=False,
        labelleft=False,
        labelbottom=False) # labels along the bottom edge are off

        plt.scatter(xs, ys, s=2.0, linewidths=2.0)

targetsuperiter = int(sys.argv[2])

plot([directory + "/inputs/meshes/target.ply", directory + "/inputs/meshes/photometric.ply"])
plt.savefig("initialization.png")
plt.clf()

if os.path.exists(directory + "/targets/npy/target-image-00.npy"):
    targetdata = np.load(directory + "/targets/npy/target-image-00.npy")
else:
    targetdata = np.load(directory + "/inputs/target.npy")[:,:,0]

cv2.imwrite("image-target.png", np.clip((targetdata * 1 * 256), 0, 255).astype('uint8'))

plot([directory + "/inputs/meshes/target.ply", directory + "/meshes/normals/" + format(targetsuperiter) + "/0000.ply"])
plt.savefig("mesh.png")
plt.clf()

targetdata = np.load(directory + "/images/current/npy/00/0000-img-00.npy")
cv2.imwrite("image-photometric.png", np.clip((targetdata * 1 * 256), 0, 255).astype('uint8'))

targetdata = np.load(directory + "/images/current/npy/" + format(targetsuperiter) + "/0000-img-00.npy")
cv2.imwrite("image-final.png", np.clip((targetdata * 1 * 256), 0, 255).astype('uint8'))

renderMeshAndMap("normals-initial.png", "mesh-initial.png", directory + "/inputs/meshes/photometric.ply")
renderMeshAndMap("normals-final.png", "mesh-final.png", directory + "/meshes/normals/" + format(targetsuperiter) + "/0000.ply")
renderMeshAndMap("normals-target.png", "mesh-target.png", directory + "/inputs/meshes/target.ply")

posImg = directory + "/renders/gradients/00/0000-img00.p.npy"
negImg = directory + "/renders/gradients/00/0000-img00.n.npy"

gradientData = (np.load(posImg) - np.load(negImg)) * 200 * 255

cv2.imwrite("gradients-x.png", _polarizeData(gradientData[:,:,0]).astype(np.uint8))
cv2.imwrite("gradients-y.png", _polarizeData(gradientData[:,:,1]).astype(np.uint8))
cv2.imwrite("gradients-z.png", _polarizeData(gradientData[:,:,2]).astype(np.uint8))

# Estimate BSDFs.
