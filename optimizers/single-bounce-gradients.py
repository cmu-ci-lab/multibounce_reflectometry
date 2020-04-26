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

import merl_io

parser = optparse.OptionParser()
parser.add_option("-l", "--linear", dest="linear", action="store_true", default=False)
parser.add_option("-c", "--distribution", dest="distribution", default=None, type="string")
parser.add_option("-r", "--samples", dest="samples", default=128, type="int")
parser.add_option("-i", "--iteration", dest="iteration", default=None, type="int")
parser.add_option("-s", "--super-iteration", dest="superiteration", default=0, type="int")

(options, args) = parser.parse_args()

directory = args[0]

dataset = Dataset(directory)
testset = dataset.testSet()

gdirpath = directory + "/meshes/single-bounce-gradients"
idirpath = directory + "/images/single-bounce-currents"
gidirpath = directory + "/renders/single-bounce-gradients"
if not os.path.exists(gdirpath):
    os.mkdir(gdirpath)
if not os.path.exists(idirpath):
    os.mkdir(idirpath)
if not os.path.exists(gidirpath):
    os.mkdir(gidirpath)

print(dataset.testSet().numIterations() + dataset.testSet().numBSDFIterations())
print("")
print("")

for i in range(dataset.testSet().numIterations() + dataset.testSet().numBSDFIterations()):
    gsubdir = gdirpath + "/" + format(options.superiteration).zfill(2)
    isubdir = idirpath + "/" + format(options.superiteration).zfill(2)
    gisubdir = gidirpath + "/" + format(options.superiteration).zfill(2)

    if not os.path.exists(gsubdir):
        os.mkdir(gsubdir)
    if not os.path.exists(isubdir):
        os.mkdir(isubdir)
    if not os.path.exists(gisubdir):
        os.mkdir(gisubdir)

    if options.iteration is not None and i != options.iteration:
        continue

    if dataset.testSet().numIterations() > i:
        ii = i
        bi = -1
    else:
        ii = 0
        bi = i - dataset.testSet().numIterations()

    if dataset.meshfileAt(iteration=ii, superiteration=options.superiteration) is None and ii != 0:
        print("Couldn't find ", options.superiteration, "-", ii)
        break

    if ii != 0:
        meshfile = dataset.meshfileAt(iteration=ii, superiteration=options.superiteration)
    else:
        meshfile = dataset.testSet().initialMeshPath()

    paramlist = testset.parameterList()
    bsdf = toMap(paramlist, dataset.BSDFAt(iteration=bi, superiteration=options.superiteration))

    for k in range(dataset.testSet().numLights()):
        print "Iteration ", i, "/", testset.numIterations(), ", Light ", k, "/", dataset.testSet().numLights(), "",
        sys.stdout.flush()

        testset.gradientRenderables[k].setEmbeddedParameter("sampleCount", options.samples)
        testset.gradientRenderables[k].setParameter("blockSize", 8)
        testset.renderables[k].setEmbeddedParameter("sampleCount", options.samples)
        testset.renderables[k].setParameter("blockSize", 8)
        copyfile(meshfile,
                            "/tmp/mts_mesh_gradient_slot_" + format(k) + ".ply")
        copyfile(meshfile,
                            "/tmp/mts_mesh_intensity_slot_" + format(k) + ".ply")
        testset.gradientRenderables[k].setEmbeddedParameter("depth", 2)
        testset.gradientRenderables[k].setEmbeddedParameter("meshSlot", k)
        testset.renderables[k].setEmbeddedParameter("depth", 2)
        testset.renderables[k].setEmbeddedParameter("meshSlot", k)

        os.system("rm /tmp/tabular-bsdf-" + format(k) + ".binary")
        merl_io.merl_write("/tmp/tabular-bsdf-" + format(k) + ".binary", dataset.tabularBSDFAt(iteration=bi, superiteration=options.superiteration))

        de = dataset.errorAtN(lindex=k, iteration=i, superiteration=options.superiteration)
        width = de.shape[0]
        height = de.shape[1]
        testset.gradientRenderables[k].setEmbeddedParameter("width", width)
        testset.gradientRenderables[k].setEmbeddedParameter("height", height)

        #plt.imshow(de)
        #plt.show()
        hdsutils.writeHDSImage("/tmp/reductor-" + format(k) + ".hds", de.shape[0], de.shape[1], 1, de[:,:,np.newaxis])

        normalGradients, bsdfGradients = testset.gradientRenderables[k].renderReadback(
            readmode="shds",
            distribution=options.distribution,
            output="/tmp/single-bounce-gradients-" + format(k) + ".shds",
            embedded=bsdf,
            localThreads=2,
            quiet=False)

        intensity = testset.renderables[k].renderReadback(
            readmode="hds",
            distribution=options.distribution,
            output="/tmp/single-bounce-gradients-" + format(k) + ".hds",
            localThreads=2,
            embedded=bsdf,
            quiet=True)

        outplyfile = gsubdir + "/" + format(i).zfill(4) + "-" + format(k).zfill(2) + ".ply"
        load_normals.emplace_normals_as_colors(
            meshfile,
            outplyfile,
            normalGradients,
            asfloat=True
        )

        splitpolarity.splitPolarity(outplyfile, "/tmp/neg.ply", "/tmp/pos.ply")
        rendernormals.renderMesh("/tmp/neg.ply", testset.sceneColors(), "/tmp/neg.hds", "/tmp/neg.npy", W=256, H=256)
        rendernormals.renderMesh("/tmp/pos.ply", testset.sceneColors(), "/tmp/pos.hds", "/tmp/pos.npy", W=256, H=256)
        negvals = np.load("/tmp/neg.npy")
        posvals = np.load("/tmp/pos.npy")

        final = posvals - negvals

        np.save(isubdir + "/" + format(i).zfill(4) + "-" + format(k).zfill(2) + ".npy", intensity)
        np.save(gisubdir + "/" + format(i).zfill(4) + "-" + format(k).zfill(2) + ".npy", final)