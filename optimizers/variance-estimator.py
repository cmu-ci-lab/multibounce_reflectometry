# Computes variance in gradient

from dictionary_embedded import embedDictionary
import json
import numpy as np
import sys
import os
from shutil import copyfile
import hdsutils
import optparse

import matplotlib.pyplot as plt

from dataset_reader import Dataset, Testset
import scipy.ndimage
#execfile(os.path.dirname(__file__) + "/../tools/dataset_reader.py")

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

def directSpatialTexture(de, scount):
    # Take absolute value and blur the textures a little.
    ade = np.abs(de)
    bade = np.zeros_like(ade)
    for i in range(ade.shape[2]):
        bade[:,:,i] = scipy.ndimage.filters.gaussian_filter(ade[:,:,i], 4.0)

    # 'normalize' the sampling texture with the L1 metric. (Set mean to 1.)
    nbade = bade / np.mean(bade)

    # Add a minimum value of 4 samples to every pixel.
    snbade = nbade + 4.0/scount

    return snbade

parser = optparse.OptionParser()
parser.add_option("-n", "--repeat-count", dest="repeatCount", default=32, type="int")
parser.add_option("-r", "--render-samples", dest="renderSamples", default=16, type="int")
parser.add_option("-s", "--sample-file", dest="sampleFile", default=None)
parser.add_option("-c", "--distribution", dest="distribution", default=None)
parser.add_option("-u", "--use-actual-reductor", action="store_true", dest="useActualReductor", default=False)
parser.add_option("-p", "--use-plain-reductor", action="store_true", dest="usePlainReductor", default=False)
(options, args) = parser.parse_args()

directory = args[0]

dataset = Dataset(directory)

if options.sampleFile is None:
    print("Using default sample profile filename")
    options.sampleFile = directory + "/spatial_sample_profiles.json"

sampleProfiles = json.load(open(options.sampleFile, "r"))

pt = np.array(dataset.lastAvailableBSDF())

if not dataset.testSet().bsdfAdaptiveSampled:
    print("BSDF Adaptive Sampling should be enabled for variance testing")
    sys.exit(1)

adaptiveParamList = dataset.bsdfAdaptiveSamplingParameterList
paramList = dataset.testSet().parameterList()
testset = dataset.testSet()

# override
for k in range(dataset.testSet().numLights()):
    testset.gradientRenderables[k].setEmbeddedParameter("sampleCount", options.renderSamples)

testset.gradientRenderables[k].setParameter("blockSize", 8)


reductors = None
if options.useActualReductor:
    # Compute the actual reductor texture
    copyfile(testset.targetMeshPath, "/tmp/mts_mesh_intensity_slot_0.ply")
    reductors = []
    currents = []

    for k in range(testset.numLights()):
        #plt.imshow(np.ones((testset.targetWidth, testset.targetHeight)))
        #plt.show()
        hdsutils.writeHDSImage("/tmp/sampler-" + format(k) + ".hds", testset.targetWidth, testset.targetHeight, 1, np.ones((testset.targetWidth, testset.targetHeight, 1)))
        cimg = testset.renderables[k].renderReadback(
                readmode="hds",
                distribution=options.distribution,
                output="/tmp/variance-testing.hds",
                embedded=mergeMaps(
                        {"meshSlot":k, "sampleCount": 512},
                        mergeMaps(toMap(adaptiveParamList, pt), toMap(paramList, pt))
                    ),
                localThreads=2,
                quiet=False
            )
        pt_target = testset.targetBSDF()
        timg = testset.renderables[k].renderReadback(
                readmode="hds",
                distribution=options.distribution,
                output="/tmp/variance-testing.hds",
                embedded=mergeMaps(
                        {"meshSlot":k, "sampleCount": 512},
                        mergeMaps(toMap(adaptiveParamList, pt_target), toMap(paramList, pt_target))
                    ),
                localThreads=2,
                quiet=False
            )
        reductor = 2 * (cimg - timg)
        reductors.append(reductor)
        currents.append(cimg)
        hdsutils.writeHDSImage("/tmp/reductor-" + format(k) + ".hds", reductor.shape[0], reductor.shape[1], 1, reductor[:,:,np.newaxis])

elif options.usePlainReductor:
    # Write a sample HDS file.
    for k in range(testset.numLights()):
        hdsutils.writeHDSImage("/tmp/reductor-" + format(k) + ".hds", testset.targetWidth, testset.targetHeight, 1, np.ones((testset.targetWidth, testset.targetHeight, 1)))
else:
    # Don't write a reductor texture.
    pass

# Set the target mesh as the true mehs.
copyfile(testset.targetMeshPath, "/tmp/mts_mesh_gradient_slot_0.ply")

for profile in sampleProfiles:
    if "type" not in profile:
        profile["type"] = "bsdf-adaptive"

data = []
for profile in sampleProfiles:
    #if profile["name"] != "bsdfWeight":
    #    continue

    allgradients = []

    if profile["type"] == "bsdf-adaptive":
        adaptiveParamMap = toMap(adaptiveParamList, profile["sampleWeights"])
    else:
        adaptiveParamMap = toMap(adaptiveParamList, pt)
    

    for i in range(options.repeatCount):
        gradients = []
        for k in range(dataset.testSet().numLights()):
            print("TYPE: ", profile["type"])
            if profile["type"] == "spatial-custom":
                sampler = np.load(profile["spatial-sampler"])[k, :, :]
                #plt.imshow(sampler)
                #plt.show()
                hdsutils.writeHDSImage("/tmp/sampler-" + format(k) + ".hds", sampler.shape[0], sampler.shape[1], 1, sampler[:,:,np.newaxis])
            elif profile["type"] == "spatial-auto":
                if reductors is None:
                    print("Spatial-Auto mode requires pre-computed reductors")
                    sys.exit(1)
                sampler = directSpatialTexture(currents[k], options.renderSamples)
                hdsutils.writeHDSImage("/tmp/sampler-" + format(k) + ".hds", sampler.shape[0], sampler.shape[1], 1, sampler[:,:,np.newaxis])

            normaldata, bsdfdata = testset.gradientRenderables[k].renderReadback(
                readmode="shds",
                distribution=options.distribution,
                output="/tmp/variance-testing.shds",
                embedded=mergeMaps({"meshSlot":k},mergeMaps(adaptiveParamMap, toMap(paramList, pt))),
                localThreads=2,
                quiet=True
            )
            gradients.append(bsdfdata)
            print profile["name"], ": Repetition ", i, "/", options.repeatCount, ", Light ", k, "/", dataset.testSet().numLights(), "\r",
            sys.stdout.flush()

        gradients = np.sum(np.stack(gradients, axis=1), axis=1)
        allgradients.append(gradients)

    gradvars = np.var(np.stack(allgradients, axis=1), axis=1)
    data.append({"name": profile["name"], "gradvars": gradvars, "gradients": allgradients})
    print("")
    print(profile["name"])
    print(gradvars)

np.save(directory + "/variance-data.npy", data)