# Compares finite-differences with monte-carlo gradients

from dictionary_embedded import embedDictionary
import json
import numpy as np
import sys
import os
from shutil import copyfile
import hdsutils
import optparse
import time

import matplotlib.pyplot as plt

from dataset_reader import Dataset, Testset, toMap, mergeMaps

import load_normals

import splitpolarity
import rendernormals

import merl_io

import xml.etree.ElementTree as ET


parser = optparse.OptionParser()
parser.add_option("-l", "--linear", dest="linear", action="store_true", default=False)
parser.add_option("-c", "--distribution", dest="distribution", default=None, type="string")
parser.add_option("-r", "--samples", dest="samples", default=128, type="int")
parser.add_option("-d", "--direct-samples", dest="directSamples", default=4, type="int")
parser.add_option("-i", "--iteration", dest="iteration", default=None, type="int")
parser.add_option("-s", "--super-iteration", dest="superiteration", default=0, type="int")
parser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False)
parser.add_option("--direct-only", dest="directOnly", action="store_true", default=False)

(options, args) = parser.parse_args()

directory = args[0]

dataset = Dataset(directory)
testset = dataset.testSet()

if not options.directOnly:
    idirpath = directory + "/renders/indirect"
else:
    idirpath = directory + "/renders/direct"

if not os.path.exists(idirpath):
    os.mkdir(idirpath)

for i in range(dataset.testSet().numIterations() + dataset.testSet().numBSDFIterations()):
    if i >= dataset.testSet().numIterations():
        ii = 0
        bi = i - dataset.testSet().numIterations()
    else:
        ii = i
        bi = -1

    isubdir = idirpath + "/" + format(options.superiteration).zfill(2)

    if not os.path.exists(isubdir):
        os.mkdir(isubdir)

    if options.iteration is not None and i != options.iteration:
        continue

    if dataset.meshfileAt(iteration=ii, superiteration=options.superiteration) is None and dataset.testSet().numIterations() != 0:
        print("Couldn't find ", options.superiteration, "-", i)
        break


    if dataset.testSet().numIterations() == 0:
        meshfile = dataset.testSet().initialMeshPath()
    else:
        meshfile = dataset.meshfileAt(iteration=ii, superiteration=options.superiteration)

    paramlist = testset.parameterList()
    bsdf = toMap(paramlist, dataset.BSDFAt(iteration=bi, superiteration=options.superiteration))

    for k in range(dataset.testSet().numLights()):
        print "Iteration ", i, "/", testset.numIterations(), ", Light ", k, "/", dataset.testSet().numLights(), "\r",
        sys.stdout.flush()

        testset.renderables[k].setEmbeddedParameter("sampleCount", options.directSamples)
        testset.renderables[k].setParameter("blockSize", 8)
        copyfile(meshfile,
                            "/tmp/mts_mesh_intensity_slot_" + format(k) + ".ply")
        merl_io.merl_write("/tmp/tabular-bsdf-" + format(k) + ".binary", dataset.tabularBSDFAt(iteration=bi, superiteration=options.superiteration))

        testset.renderables[k].setEmbeddedParameter("depth", 2)
        testset.renderables[k].setEmbeddedParameter("meshSlot", k)
        de = dataset.errorAtN(lindex=k, iteration=i, superiteration=options.superiteration)

        #testset.renderables[k].setEmbeddedParameter("width", de.shape[0])
        #testset.renderables[k].setEmbeddedParameter("height", de.shape[1])

        directIntensity = testset.renderables[k].renderReadback(
            readmode="hds",
            distribution=options.distribution,
            output="/tmp/indirects-" + format(k) + ".hds",
            localThreads=2,
            embedded=bsdf,
            quiet=not options.verbose)

        if not options.directOnly:
            testset.renderables[k].setEmbeddedParameter("depth", -1)
            testset.renderables[k].setEmbeddedParameter("sampleCount", options.samples)

            intensity = testset.renderables[k].renderReadback(
                readmode="hds",
                distribution=options.distribution,
                output="/tmp/indirects-" + format(k) + ".hds",
                localThreads=2,
                embedded=bsdf,
                quiet=not options.verbose)

        if not options.directOnly:
            np.save(isubdir + "/" + format(i).zfill(4) + "-" + format(k).zfill(2) + ".npy", intensity - directIntensity)
        else:
            np.save(isubdir + "/" + format(i).zfill(4) + "-" + format(k).zfill(2) + ".npy", directIntensity)