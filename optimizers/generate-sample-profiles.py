# Generates sample profiles.

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

def invAlpha(dictionary):
    epsilon = 1e-3
    wts = []
    for element in dictionary["elements"]:
        if element["type"] == "roughconductor":
            wts.append(1/(element["alpha"] + epsilon))
        elif element["type"] == "diffuse":
            wts.append(1.0)
        else:
            wts.append(1.0)
    return np.array(wts)

# from dataset_reader import Dataset (Alternative temporary fix)
#execfile(os.path.dirname(__file__) + "/../tools/dataset_reader.py")


parser = optparse.OptionParser()

(options, args) = parser.parse_args()

directory = args[0]

dataset = Dataset(directory)

profiles = [
    {
        "name": "invAlpha",
        "sampleWeights": invAlpha(dataset.testSet().bsdfDictionary).tolist(),
        "type": "bsdf-adaptive"
    },
    {
        "name": "uniform",
        "sampleWeights": [1.0] * len(dataset.testSet().bsdfDictionary["elements"]),
        "type": "bsdf-adaptive"
    },
    {
        "name": "bsdfWeight",
        "sampleWeights": dataset.lastAvailableBSDF(),
        "type": "bsdf-adaptive"
    }
]

uniform = np.stack([np.ones((dataset.testSet().targetWidth, dataset.testSet().targetHeight))] * dataset.testSet().numLights(), axis=0)
np.save(directory + "/uniform.npy", uniform)
spatialProfiles = [
    {
        "name": "spatialUniform",
        "type": "spatial-custom",
        "spatial-sampler": directory + "/uniform.npy"
    },
    {
        "name": "spatialAdaptive",
        "type": "spatial-auto"
    }
]

json.dump(profiles, open(directory + "/sample_profiles.json", "w"))
json.dump(spatialProfiles, open(directory + "/spatial_sample_profiles.json", "w"))