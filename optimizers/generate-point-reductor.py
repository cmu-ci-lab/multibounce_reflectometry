import numpy as np
import json
import os
import sys
import merl_io
import optparse
import matplotlib.pyplot as plt
import bivariate_proj as bivariate_proj
import hdsutils



parser = optparse.OptionParser()
parser.add_option("-o", "--output", dest="output", default="/tmp/reductor-0.hds")
parser.add_option("-x", "--x", dest="x", default=128, type="int")
parser.add_option("-y", "--y", dest="y", default=128, type="int")
parser.add_option("--width", dest="width", default=256, type="int")
parser.add_option("--height", dest="height", default=256, type="int")
parser.add_option("--scale", dest="scale", default=1.0, type="float")

(options, args) = parser.parse_args()

reductor = np.zeros((options.width, options.height, 1))
reductor[options.y, options.x, 0] = options.scale

hdsutils.writeHDSImage(options.output, reductor.shape[0], reductor.shape[1], reductor.shape[2], reductor)