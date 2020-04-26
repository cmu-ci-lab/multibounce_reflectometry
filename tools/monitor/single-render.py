import sys
import os
import matplotlib.pyplot as plt
from matplotlib import pylab
import load_normals
import json

directory = sys.argv[1]
parameters = json.load(open(directory + "/inputs/config.json", "r"))

w = 3
h = 3
if len(sys.argv) > 2:
    w = int(sys.argv[2])
    h = int(sys.argv[3])

def plotNormals(vertices, normals, color='k'):
    # Select a slice of vertices.
    shortlist = []
    for vertex, normal in zip(vertices, normals):
        if vertex[1] > -0.002 and vertex[1] < +0.002:
            shortlist.append((vertex, normal))

    for vertex, normal in shortlist[::2]:
        plt.plot([vertex[0], vertex[0] + normal[0]*0.04], [vertex[2], vertex[2] + normal[2]*0.04], color=color, linestyle='-', linewidth=0.1)

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
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False,
        right=False,
        labelleft=False,
        labelbottom=False) # labels along the bottom edge are off

        plt.scatter(xs, ys, s=0.1, linewidths=0.1)

plt.subplot(w, h, 1)
plt.tick_params(axis='both', which='major', labelsize=3)
plt.tick_params(axis='both', which='minor', labelsize=3)
plot([directory + "/inputs/meshes/target.ply", directory + "/meshes/normals/00/0000.ply"])

for index in range(parameters["remesher"]["iterations"]):
    print("Found instance index " + format(index))

    plt.subplot(w, h, index + 2)
    plt.tick_params(axis='both', which='major', labelsize=3)
    plt.tick_params(axis='both', which='minor', labelsize=3)

    #params = json.load(open(directory + "/" + instance + "/inputs/config.json", "r"))
    #plt.text(
    #    0, 0.2,
    #    "D: " + format(params["estimator"]["depth"]) + "\nS: " + format(params["estimator"]["samples"]["samples"][0]) + "\nI: " + format(params["estimator"]["iterations"]) + "\nO: " + params["estimator"]["optimizer"]["type"],
    #    size=3, horizontalalignment='center')
    if os.path.exists(directory + "/meshes/remeshed/" + format(index).zfill(2) + ".ply"):
        plot([directory + "/inputs/meshes/target.ply", directory + "/meshes/remeshed/" + format(index).zfill(2) + ".ply"])

plt.savefig("meshes.png", dpi=2000)