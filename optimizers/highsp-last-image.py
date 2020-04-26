import optparse
import os
import json
import numpy as np
from shutil import copyfile

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

from dataset_reader import Dataset, Testset

# Plots high sample count last image.
parser = optparse.OptionParser()
parser.add_option("-s", "--samples", dest="samples", default=2048, type="int")
parser.add_option("-c", "--distribution", dest="distribution", default=None)

(options, args) = parser.parse_args()

directory = args[0]
dataset = Dataset(directory)

# TODO: Temporary
copyfile(dataset.testSet().targetMeshPath, "/tmp/mts_mesh_intensity_slot_0.ply")

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

if not os.path.exists(directory + "/images/current-highsp"):
    os.mkdir(directory + "/images/current-highsp")

if not os.path.exists(directory + "/images/current-highsp/hds"):
    os.mkdir(directory + "/images/current-highsp/hds")

if not os.path.exists(directory + "/images/current-highsp/npy"):
    os.mkdir(directory + "/images/current-highsp/npy")

pt0 = np.array(dataset.lastAvailableBSDF())
pt_target = np.array(dataset.testSet().targetBSDF)
for k in range(dataset.testSet().numLights()):
    data = renderable.renderReadback(
                distribution=options.distribution,
                output=directory + "/images/current-highsp/hds/final-" + format(k).zfill(2) + ".hds",
                embedded=mergeMaps(adaptiveParamMap, toMap(paramList, pt0)),
                localThreads=2,
                quiet=False
            )

    np.save(directory + "/images/current-highsp/npy/final-" + format(k).zfill(2) + ".npy", data)
