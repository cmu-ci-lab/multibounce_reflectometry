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

from dataset_reader import Dataset
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


parser = optparse.OptionParser()
parser.add_option("-l", "--linear", dest="linear", action="store_true", default=False)
parser.add_option("-c", "--distribution", dest="distribution", default=None, type="string")
parser.add_option("-r", "--samples", dest="samples", default=128, type="int")
parser.add_option("-f", "--fd-samples", dest="fdSamples", default=1024, type="int")
parser.add_option("-s", "--jump-size", dest="jumpSize", default=None, type="string")
parser.add_option("-k", "--resolution", dest="resolution", default=256, type="int")

(options, args) = parser.parse_args()

directory = args[0]

dataset = Dataset(directory)

pt = np.array(dataset.lastAvailableBSDF())

if not dataset.testSet().bsdfAdaptiveSampled:
    print("BSDF Adaptive Sampling should be enabled for variance testing")
    sys.exit(1)

adaptiveParamList = dataset.bsdfAdaptiveSamplingParameterList
paramList = dataset.testSet().parameterList()
testset = dataset.testSet()

###
# Monte Carlo gradient
###

# override
for k in range(dataset.testSet().numLights()):
    testset.gradientRenderables[k].setEmbeddedParameter("sampleCount", options.samples)

testset.gradientRenderables[k].setParameter("blockSize", 8)

# Compute the actual reductor texture
copyfile(testset.targetMeshPath, "/tmp/mts_mesh_intensity_slot_0.ply")

currents = []
targets = []
reductors = []
print("Rendering reductors...")
for k in range(testset.numLights()):
    cimg = testset.renderables[k].renderReadback(
            readmode="hds",
            distribution=options.distribution,
            output="/tmp/first-order-gradient-testing.hds",
            embedded=mergeMaps(
                    {
                        "sampleCount": options.samples,
                        "width": options.resolution,
                        "height": options.resolution
                    },
                    mergeMaps(toMap(adaptiveParamList, pt), toMap(paramList, pt))
                ),
            localThreads=2,
            quiet=True
        )
    pt_target = testset.targetBSDF()
    timg = testset.renderables[k].renderReadback(
            readmode="hds",
            distribution=options.distribution,
            output="/tmp/first-order-gradient-testing.hds",
            embedded=mergeMaps(
                    {
                        "sampleCount": options.samples,
                        "width": options.resolution,
                        "height": options.resolution
                    },
                    mergeMaps(toMap(adaptiveParamList, pt_target), toMap(paramList, pt_target))
                ),
            localThreads=2,
            quiet=True
        )
    reductor = 2 * (cimg - timg)
    
    print "Light ", k, "/", dataset.testSet().numLights(), "\r",

    currents.append(cimg)
    targets.append(timg)
    reductors.append(reductor)

    hdsutils.writeHDSImage("/tmp/reductor-" + format(k) + ".hds", reductor.shape[0], reductor.shape[1], 1, reductor[:,:,np.newaxis])

# Set the target mesh as the true mehs.
for k in range(dataset.testSet().numLights()):
    copyfile(testset.targetMeshPath, "/tmp/mts_mesh_gradient_slot_" + format(k) + ".ply")

allgradients = []
print("Rendering Monte-Carlo gradients...")
for k in range(dataset.testSet().numLights()):
    #os.system("rm /tmp/first-order-gradient-testing-" + format(k) + ".shds")
    normaldata, bsdfdata = testset.gradientRenderables[k].renderReadback(
        readmode="shds",
        distribution=options.distribution,
        output="/tmp/first-order-gradient-testing-" + format(k) + ".shds",
        embedded=mergeMaps(
            {
                "meshSlot":k,
                "width": options.resolution,
                "height": options.resolution,
                "sampleCount": options.samples
            },
            mergeMaps(
                toMap(adaptiveParamList, pt),
                toMap(paramList, pt))),
        localThreads=2,
        quiet=True
    )

    print "Light ", k, "/", dataset.testSet().numLights(), "\r",

    allgradients.append(bsdfdata)

# Obtained monte-carlo gradient.
mcgrad = np.sum(np.stack(allgradients, axis=1), axis=1)

# --------------------------------------------------------------

###
# Finite difference gradient.
###

def verify(vals):
    if not np.all(vals < 1.0):
        return False
    if not np.all(vals > 0.0):
        return False
    return True


# To conserve the properties of the mixture-bsdf model.
# the perturb function operates by perturbing the 0 reflectance
# unit in the reverse direction.
# Assumes that the last BSDF is 0 reflectance
def perturb(pt, pt_target, k):
    mask = np.zeros_like(pt)
    mask[k] = 1

    pt_perturbed = np.array(pt)
    pt_perturbed[k] += 1e-3
    pt_perturbed[-1] -= 1e-3

    #print "Perturbing ", k, pt_perturbed.shape
    #print pt_perturbed

    #pt_perturbed = pt + (pt_target - pt) * mask * 0.05
    #delta = np.linalg.norm(pt_perturbed - pt)
    delta = 1e-3

    if verify(pt_perturbed):
        return pt_perturbed, delta
    else:
        return pt_perturbed, None

def verifyGrad(pt, pt2):
    allmatch = True
    for i, (u,v) in enumerate(zip(pt, pt2)):
        print i, " MC: ", u, " FD: ", v,
        if u*v > 0:
            print "same-sign"
        else:
            print "mismatch"
            allmatch = False

    if allmatch:
        print "All signs match"


# For every available BSDF parameter.
print("Computing finite differences")
print(pt.shape)
fdgrad = []

for i in range(pt.shape[0]):
    newpt, delta = perturb(pt, pt_target, i)
    if delta is None:
        print("Perturbed vector invalid. Skipping this dimension")
        print(newpt)
        fdgrad.append(0.0)
        continue

    gradient = 0.0
    for k in range(testset.numLights()):
        #os.system("rm /tmp/first-order-gradient-testing-" + format(i) + "-" + format(k) + ".hds")
        timg = testset.renderables[k].renderReadback(
            readmode="hds",
            distribution=options.distribution,
            output="/tmp/first-order-gradient-testing-" + format(i) + "-" + format(k) + ".hds",
            embedded=mergeMaps(
                    {
                        "sampleCount": options.fdSamples,
                        "width": options.resolution,
                        "height": options.resolution
                    },
                    mergeMaps(toMap(adaptiveParamList, newpt), toMap(paramList, newpt))
                ),
            localThreads=2,
            quiet=True
        )
        
        #plt.imshow(timg[:,:,0] - targets[k][:,:,0])
        #plt.imshow(currents[k][:,:,0] - targets[k][:,:,0])
        #plt.show()
        gradient += (np.sum((timg - targets[k])**2) - np.sum((currents[k] - targets[k])**2)) / delta

    fdgrad.append(-gradient)

    delement = testset.bsdfDictionary["elements"][i]

    if delement["type"] == "roughconductor":
        estring = "a/" + format(delement["alpha"]) + "/e/" + format(delement["eta"])
    elif delement["type"] == "diffuse":
        estring = "d/" + format(delement["reflectance"])

    print i, " ", estring, " MC: ", mcgrad[i], " FD: ", fdgrad[i],
    if mcgrad[i]*fdgrad[i] > 0:
        print " same-sign ",
    else:
        print " mismatch ",
    print " Current: ", pt[i], " Target: ", pt_target[i]


fdgrad = np.array(fdgrad)

if not os.path.exists(directory + "/first-order-testing"):
    os.mkdir(directory + "/first-order-testing")

np.save(directory + "/first-order-testing/fdgrad.npy", fdgrad)
np.save(directory + "/first-order-testing/mcgrad.npy", mcgrad)
json.dump(
    {
        "samples": options.samples,
        "jump-size": options.jumpSize,
        "fd-samples": options.fdSamples
    }, open(directory + "/first-order-testing/options.json", "w"))

verifyGrad(gradient, mcgrad)