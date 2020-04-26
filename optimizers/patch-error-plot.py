# Compares finite-differences with monte-carlo gradients

from dictionary_embedded import embedDictionary
import json
import numpy as np
import sys
import os
from shutil import copyfile
import hdsutils
import optparse

import matplotlib.pyplot as plt

from dataset_reader import Dataset, Testset, toMap, mergeMaps

import load_normals

import splitpolarity
import rendernormals

parser = optparse.OptionParser()
parser.add_option("-l", "--linear", dest="linear", action="store_true", default=False)
parser.add_option("-c", "--distribution", dest="distribution", default=None, type="string")
parser.add_option("-s", "--samples", dest="samples", default=4, type="int")
parser.add_option("-r", "--resolution", dest="resolution", default=10, type="int")
parser.add_option("-x", "--nx", dest="nx", default=0.0, type="float")
parser.add_option("-y", "--ny", dest="ny", default=0.0, type="float")
parser.add_option("-z", "--nz", dest="nz", default=1.0, type="float")

(options, args) = parser.parse_args()

directory = args[0]

testset = Testset(directory)

paramlist = testset.parameterList()
adaptiveList = testset.bsdfAdaptiveSamplingParameterList
bsdf = toMap(paramlist, testset.targetBSDF())
adaptiveDistr = toMap(adaptiveList, testset.targetBSDF())
bsdf = mergeMaps(bsdf, adaptiveDistr)

numVertices = load_normals.load_normals(testset.initialMeshPath()).shape[0]


targets = []
for k in range(testset.numLights()):
    print "Light ", k, "/", testset.numLights(), "\r",
    sys.stdout.flush()

    testset.gradientRenderables[k].setEmbeddedParameter("sampleCount", options.samples)
    testset.gradientRenderables[k].setParameter("blockSize", 8)
    testset.renderables[k].setEmbeddedParameter("sampleCount", options.samples)
    testset.renderables[k].setParameter("blockSize", 8)
    copyfile(testset.initialMeshPath(),
                        "/tmp/mts_mesh_gradient_slot_" + format(k) + ".ply")
    copyfile(testset.initialMeshPath(),
                        "/tmp/mts_mesh_intensity_slot_" + format(k) + ".ply")
    
    nml = np.array((options.nx, options.ny, options.nz))
    nml = nml / np.linalg.norm(nml)
    normals = [(nml[0], nml[1], nml[2])] * numVertices

    load_normals.emplace_normals_as_colors(
                "/tmp/mts_mesh_intensity_slot_" + format(k) + ".ply",
                "/tmp/mts_mesh_intensity_slot_" + format(k) + ".emplaced.ply",
                np.array(normals),
                asfloat=True,
                asnormals=True
            )

    copyfile("/tmp/mts_mesh_intensity_slot_" + format(k) + ".emplaced.ply", "/tmp/mts_mesh_intensity_slot_" + format(k) + ".ply")

    testset.gradientRenderables[k].setEmbeddedParameter("depth", -1)
    testset.gradientRenderables[k].setEmbeddedParameter("meshSlot", k)
    testset.renderables[k].setEmbeddedParameter("depth", 2)
    testset.renderables[k].setEmbeddedParameter("meshSlot", k)
    width = 8
    height = 8
    de = np.ones((width, height))
    width = de.shape[0]
    height = de.shape[1]
    testset.gradientRenderables[k].setEmbeddedParameter("width", width)
    testset.gradientRenderables[k].setEmbeddedParameter("height", height)
    testset.renderables[k].setEmbeddedParameter("width", width)
    testset.renderables[k].setEmbeddedParameter("height", height)

    targets.append(testset.renderables[k].renderReadback(
                readmode="hds",
                distribution=options.distribution,
                output="/tmp/patch-error-plot-" + format(k) + ".hds",
                localThreads=2,
                embedded=bsdf,
                quiet=True))

    hdsutils.writeHDSImage("/tmp/reductor-" + format(k) + ".hds", de.shape[0], de.shape[1], 1, de[:,:,np.newaxis])

errorPlot = np.zeros((options.resolution,options.resolution))
gradientPlot = np.zeros((options.resolution,options.resolution,3))
maskPlot = np.zeros((options.resolution,options.resolution))

for x in range(options.resolution):
    for y in range(options.resolution):
        
        fx = (float(x)/options.resolution)
        fy = (float(y)/options.resolution)
        nx = (-1.0) * (1 - fx) + fx * 1.0
        ny = (-1.0) * (1 - fy) + fy * 1.0

        if (nx**2 + ny**2) > 1:
            continue

        nz = np.sqrt(1 - (nx**2 + ny**2))
        maskPlot[x,y] = 1

        print "Position ", x, "/", options.resolution, " ", y, "/", options.resolution, 
        print "Angle: " , ('%.3f' % nx), ",", ('%.3f' % ny), ",", ('%.3f' % nz), "\r",
        sys.stdout.flush()

        normals = [(nx, ny, nz)] * numVertices

        loss = 0
        gradient = np.zeros((3,))
        for k in range(testset.numLights()):

            load_normals.emplace_normals_as_colors(
                "/tmp/mts_mesh_gradient_slot_" + format(k) + ".ply",
                "/tmp/mts_mesh_gradient_slot_" + format(k) + ".emplaced.ply",
                np.array(normals),
                asfloat=True,
                asnormals=True
            )

            copyfile("/tmp/mts_mesh_gradient_slot_" + format(k) + ".emplaced.ply", "/tmp/mts_mesh_gradient_slot_" + format(k) + ".ply")

            load_normals.emplace_normals_as_colors(
                "/tmp/mts_mesh_intensity_slot_" + format(k) + ".ply",
                "/tmp/mts_mesh_intensity_slot_" + format(k) + ".emplaced.ply",
                np.array(normals),
                asfloat=True,
                asnormals=True
            )

            copyfile("/tmp/mts_mesh_intensity_slot_" + format(k) + ".emplaced.ply", "/tmp/mts_mesh_intensity_slot_" + format(k) + ".ply")

            normalGradients, bsdfGradients = testset.gradientRenderables[k].renderReadback(
                readmode="shds",
                distribution=options.distribution,
                output="/tmp/single-bounce-gradients-" + format(k) + ".shds",
                embedded=bsdf,
                localThreads=6,
                quiet=True)

            gradient += normalGradients[normalGradients.shape[0]//2, :]

            intensity = testset.renderables[k].renderReadback(
                readmode="hds",
                distribution=options.distribution,
                output="/tmp/single-bounce-gradients-" + format(k) + ".hds",
                localThreads=6,
                embedded=bsdf,
                quiet=True)

            loss += np.sum((intensity - targets[k]) ** 2)
            #plt.imshow(np.concatenate([intensity[:,:,0],targets[k][:,:,0]], axis=1))
            #plt.show()

        gradientPlot[x,y,:] = gradient
        errorPlot[x,y] = loss

        np.save(directory + "/errors-patch-plot.npy", errorPlot)
        np.save(directory + "/gradients-patch-plot.npy", gradientPlot)