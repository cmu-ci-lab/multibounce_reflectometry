# Plots error surface (Uses new dataset_reader library)

from dictionary_embedded import embedDictionary
import json
import numpy as np
import sys
import os
from shutil import copyfile
import hdsutils
import optparse

import matplotlib.pyplot as plt

#from dataset_reader import Dataset
execfile(os.path.dirname(__file__) + "/../tools/dataset_reader.py")

def toMap(lst, vals):
    m = {}
    for l, v in zip(lst, vals):
        m[l] = v
    return m

def mergeMaps(m1, m2):
    m3 = dict(m1)
    for k in m2:
        m3[k] = m2[k]
    return m3


parser = optparse.OptionParser()
parser.add_option("-l", "--linear", dest="linear", action="store_true", default=False)
parser.add_option("-r", "--resolution", dest="resolution", default=128, type="int")
parser.add_option("-s", "--samples", dest="samples", default=2048, type="int")
parser.add_option("-c", "--distribution", dest="distribution", default=None)
parser.add_option("-m", "--start", dest="startIdx", default=0, type="int")
parser.add_option("-n", "--end", dest="endIdx", default=-1, type="int")

(options, args) = parser.parse_args()

if options.endIdx == -1:
    options.endIdx = options.resolution - 1

if options.startIdx >= options.resolution:
    print("Start IDX cannot be higher than the resolution")
    sys.exit(1)

if options.endIdx < 0:
    print("End IDx cannot be less than 0")
    sys.exit(1)

directory = args[0]

if not os.path.exists("error-surface"):
    print("Creating error-surface")
    os.mkdir("error-surface")

dataset = Dataset(directory)

copyfile(dataset.testSet().targetMeshPath, "/tmp/mts_mesh_intensity_slot_0.ply")

pstring = ""

resolution = (options.resolution,)

surface = np.zeros(resolution)

if options.linear:
    renderable = dataset.testSet().renderable(1)
    pt0 = np.array(dataset.lastAvailableBSDF())
    pt1 = np.array(dataset.testSet().targetBSDF())

    paramList = dataset.testSet().parameterList()
    #print paramList
    #print pt0
    #print toMap(paramList, pt1)
    if dataset.testSet().bsdfAdaptiveSampled:
        adaptiveParamList = dataset.bsdfAdaptiveSamplingParameterList
        print adaptiveParamList
        adaptiveParamMap = toMap(adaptiveParamList, len(adaptiveParamList) * [1.0])
    else:
        adaptiveParamMap = {}

    renderable.setEmbeddedParameter("sampleCount", options.samples)
    renderable.setParameter("blockSize", 8)

    targets = []
    for k in range(dataset.testSet().numLights()):
        target = renderable.renderReadback(
                distribution=options.distribution,
                output="/tmp/es2-target.hds",
                embedded=mergeMaps(adaptiveParamMap, toMap(paramList, pt1)),
                localThreads=2,
                quiet=True
            )

        np.save("error-surface/target-" + format(k).zfill(2) + ".npy", target)
        targets.append(
            target
        )

    progress = 0
    for i in range(options.resolution):
        factor = (float(i) / options.resolution)
        print(factor)
        pt = pt0 * factor + pt1 * (1 - factor)
        currents = []

        if not (i <= options.endIdx) or (not (i >= options.startIdx)):
            continue

        for k in range(dataset.testSet().numLights()):
            current = renderable.renderReadback(
                    distribution=options.distribution,
                    output="/tmp/es2-current.hds",
                    embedded=mergeMaps(adaptiveParamMap, toMap(paramList, pt)),
                    localThreads=2,
                    quiet=True
                )
            np.save("error-surface/current-" + format(i).zfill(4) + "-" + format(k).zfill(2) + ".npy", current)
            currents.append(
                current
            )

        sumSquaredError = np.sum((np.stack(currents, axis=0) - np.stack(targets, axis=0))**2)
        surface[i] = sumSquaredError

        progress += 1
        print "\r" + ("%.1f" % ((float(progress * 100)/(resolution[0]))) + "\r")

np.save("error-surface/surface.npy", surface)