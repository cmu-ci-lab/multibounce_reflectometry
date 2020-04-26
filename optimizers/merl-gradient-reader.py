import merl_io
import optparse
import os
import sys
import bivariate_proj

import matplotlib.pyplot as plt
import json
import numpy as np

parser = optparse.OptionParser()
parser.add_option("-s", "--super-iteration", dest="superIteration", type="int", default=0)
parser.add_option("-i", "--iteration", dest="iteration", type="int", default=0)
parser.add_option("-g", "--gradient", dest="gradient", action="store_true", default=False)
parser.add_option("-t", "--target", dest="target", action="store_true", default=False)
parser.add_option("-u", "--unprojected", dest="unprojected", action="store_true", default=False)
parser.add_option("-p", "--projection-difference", dest="projectionDifference", action="store_true", default=False)
parser.add_option("-d", "--delta", dest="delta", action="store_true", default=False)
parser.add_option("-k", "--initial", dest="initial", action="store_true", default=False)
parser.add_option("-c", "--compare", dest="compare", action="store_true", default=False)
parser.add_option("-l", "--log", dest="log", action="store_true", default=False)
parser.add_option("-n", "--initialization", dest="initialization", action="store_true", default=False)
parser.add_option("-r", "--threshold", dest="threshold", default=None, type="float")

(options, args) = parser.parse_args()

directory = args[0]

errors = json.load(open(directory + "/errors/bsdf-errors-" + format(options.superIteration).zfill(2) + ".json", "r"))

if options.gradient:
    currentTBval = np.array(errors["tbgrads"][options.iteration])
elif options.target:
    config = json.load(open(directory + "/inputs/config.json"))
    currentTBval = np.load(directory + "/inputs/" + config["target"]["tabular-bsdf"])
elif options.unprojected:
    currentTBval = np.array(errors["tbuvals"][options.iteration])
elif options.projectionDifference:
    currentTBval = np.array(errors["tbvals"][options.iteration]) - np.array(errors["tbuvals"][options.iteration])
elif options.delta:
    if options.iteration == 0:
        print("Can't compute delta at iteration==0. Use a non-zero iteration number")
        sys.exit(1)
    currentTBval = np.array(errors["tbvals"][options.iteration])
    currentTBval = np.concatenate([np.array(errors["tbvals"][options.iteration - 1]), currentTBval], axis=1)
    currentTBval = np.concatenate([np.array(errors["tbvals"][options.iteration - 1]) - np.array(errors["tbvals"][options.iteration]), currentTBval], axis=1)
    #currentTBval = np.array(errors["tbvals"][options.iteration - 1]) - np.array(errors["tbvals"][options.iteration])
    #currentTBval[currentTBval > 10000] = 10000
elif options.initial:
    config = json.load(open(directory + "/inputs/config.json"))
    initialization = np.load(directory + "/inputs/" + config["bsdf-estimator"]["tabular-bsdf"]["initialization"])
    #projected = bivariate_proj.bivariate_proj(initialization.transpose((1,0,2))).transpose((1,0,2))
    projected = bivariate_proj.bivariate_proj(initialization)
    currentTBval = np.concatenate([initialization - projected, initialization, projected], axis=1)
    #currentTBval[currentTBval > 10000] = 10000

elif options.compare:
    config = json.load(open(directory + "/inputs/config.json"))
    currentTBval = np.load(directory + "/inputs/" + config["target"]["tabular-bsdf"])
    #currentTBval = np.array(errors["tbvals"][options.iteration]) - currentTBval
    currentTBval = np.concatenate([np.array(errors["tbvals"][options.iteration]),currentTBval, np.array(errors["tbvals"][options.iteration]) - currentTBval], axis=1)

else:
    currentTBval = np.array(errors["tbvals"][options.iteration])

if options.iteration is not None and "lregs" in errors:
    print(errors["lregs"][options.iteration])

if options.threshold is not None:
    currentTBval[currentTBval > options.threshold] = options.threshold
    currentTBval[currentTBval < -options.threshold] = -options.threshold

if options.log:
    currentTBval = np.log(np.abs(currentTBval))

print(currentTBval.shape)
plt.imshow(currentTBval[:,:,0])
plt.show()