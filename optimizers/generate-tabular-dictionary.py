import numpy as np
import json
import os
import sys
import merl_io
import optparse
import matplotlib.pyplot as plt
import bivariate_proj as bivariate_proj

parser = optparse.OptionParser()

(options, args) = parser.parse_args()

merlBinary = args[1]
directory = args[0]


dparams = []
subdifferentiables = []
params = {"type": "tabular","elements": dparams, "id": "white-rough", "subdifferentiables":subdifferentiables, "undifferentiables":0}

json.dump(params, open(directory + "/bsdf-dictionary.json", "w"))

# Save empty arrays for the tabular dictionary
np.save(directory + "/weights/initialization.npy", np.array([]))
np.save(directory + "/weights/target.npy", np.array([]))

bsdf = merl_io.merl_read(merlBinary)

targetTabularBSDF = np.sum(bsdf * (bsdf >= 0), axis=2, keepdims=True) / np.sum((bsdf >= 0), axis=2, keepdims=True)
initializationTabularBSDF = np.ones(targetTabularBSDF.shape)
initializationTabularBSDF = bivariate_proj.bivariate_proj(initializationTabularBSDF)

if not os.path.exists(directory + "/tabular-bsdf"):
    os.mkdir(directory + "/tabular-bsdf")

np.save(directory + "/tabular-bsdf/initialization.npy", initializationTabularBSDF)
np.save(directory + "/tabular-bsdf/target.npy", targetTabularBSDF)