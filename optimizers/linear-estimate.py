import json
import sys
import os

from initialization.linear_estimator.linear import renderDictionary, estimateWeights, loadLinearParameters
from parameters import prepareParameters

# Script to perform linear estimation.

def nsub(d, x, v):
    if x not in d:
        return v
    else:
        return d[x]

configdir = sys.argv[1]

if len(sys.argv) < 2:
    print("Usage: python linear-estimate.py <config-file-path>")

config = json.load(open(configdir + "/config.json", "r"))

if "weight-estimation" not in config:
    print ("weight-estimation parameters not in config file.")
    sys.exit(1)

wparams = config["weight-estimation"]

if not wparams["enabled"]:
    print ("weight-estimation.enabled is False. Skipping this config")
    sys.exit(0)

dictfile = wparams["dfile"]
dcache = None
if "cache" in wparams:
    dcache = wparams["cache"]

# Get the source rendering.
sparams = prepareParameters(config, directory=configdir)
sourceimgs = sparams["target"]["data"]

lparams = loadLinearParameters(configdir + "/" + dictfile)

# Get the dictionary rendering. could take a while.
dictionary = renderDictionary(
                {"scene":wparams["scene"],
                "depth":nsub(wparams,"depth",-1),
                "samples":nsub(wparams, "samples", 64),
                "mesh":config["initialization"]["file"],
                "lights":sparams["lights"]["data"]},
                lparams,
                directory=configdir,
                cache=True)

# Estimate sparse weights
W = estimateWeights(dictionary, sourceimgs, type=nsub(wparams, "regressor", "sgd"))

# Find peaks
wps = zip(W,lparams)
wps = sorted(wps, lambda a,b: a[0] < b[0])

# Print out the best parameter matches.
print(wps[:5])