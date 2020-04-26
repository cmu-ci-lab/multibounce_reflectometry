import json
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

def toBsdfFilename(directory, idx):
    return directory + "/errors/bsdf-errors-" + format(idx).zfill(2) + ".json"

directory = sys.argv[1]

# Find config file.
configfile = directory + "/inputs/config.json"
if not os.path.exists(configfile):
    print("Couldn't open ", configfile)
    sys.exit(1)

config = json.load(open(configfile, "r"))

lastIdx = -1
# Load the parameters at the last complete BSDF iteration.
for i in range(config["remesher"]["iterations"]):
    bsdffile = toBsdfFilename(directory, i)
    if os.path.exists(bsdffile):
        lastIdx = i

# Check if last index is complete.
if (lastIdx >= 2) and len(json.load(open(toBsdfFilename(directory, lastIdx), "r"))["bvals"]) != len(json.load(open(toBsdfFilename(directory, lastIdx-1), "r"))["bvals"]):
    lastIdx -= 1 # Choose the previous superiteration if the current entry has not converged.

if len(sys.argv) > 2:
    overrideIndex = int(sys.argv[2])
else:
    overrideIndex = None

if overrideIndex is not None:
    if overrideIndex > lastIdx:
        print("Index requested is too large. Max complete iterations: ", lastIdx)
        sys.exit(1)
    lastIdx = overrideIndex

print ("Using index ", lastIdx)
# Load weights.
weights = json.load(open(toBsdfFilename(directory, lastIdx), "r"))["bvals"][-1]

# Find dictionary file.
if ("bsdf-preprocess" not in config) or not config["bsdf-preprocess"]["enabled"]:
    print("BSDF preprocessing not enabled. No bsdf data to plot")
    print("Config file: ", open(configfile,"r").read())
    sys.exit(1)

dictionaryfile = config["bsdf-preprocess"]["file"]
dictionary = json.load(open(directory + "/inputs/" + dictionaryfile, "r"))

alphas = {}
etas = {}
alphaetamap = {}

# Go through the atoms to find the elements
for idx, atom in enumerate(dictionary["elements"]):
    if atom["type"] == "roughconductor":
        alpha = atom["alpha"]
        eta = atom["eta"]
        alphas[alpha] = 1
        etas[eta] = 1
        alphaetamap[(alpha,eta)] = weights[idx]

grid = np.zeros((len(alphas), len(etas)))
_alphas = alphas.keys()
_alphas.sort()
_etas = etas.keys()
_etas.sort()

for x, alpha in enumerate(_alphas):
    for y, eta in enumerate(_etas):
        grid[x,y] = alphaetamap[(alpha, eta)]

plt.imshow(grid, cmap='hot', interpolation="bicubic")
plt.xticks(np.arange(len(_etas)), ["%.2f" % eta for eta in _etas])
plt.yticks(np.arange(len(_alphas)), ["%.2f" % alpha for alpha in _alphas])
plt.ylabel("alpha")
plt.xlabel("eta")
plt.show()
