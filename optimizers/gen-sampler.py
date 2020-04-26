# Generate Sampler textures.
import optparse
import numpy as np
import os
import hdsutils
import scipy.ndimage
import matplotlib.pyplot as plt

parser = optparse.OptionParser()
parser.add_option("-w", "--width", dest="width", default=256, type="int")
parser.add_option("-t", "--height", dest="height", default=256, type="int")
parser.add_option("-m", "--mode", dest="mode", default=None)
(options, args) = parser.parse_args()

outputfile = args[0]

if options.mode == "uniform":
    uniform = np.ones((options.width, options.height, 1))
    plt.imshow(uniform[:,:,0])
    plt.show()
    hdsutils.writeHDSImage(outputfile, options.width, options.height, 1, uniform)
elif options.mode == "gaussian":
    fn = np.zeros((options.width, options.height))
    fn[128, 128] = 20
    gaussian = scipy.ndimage.filters.gaussian_filter(fn, (options.width)/5)
    gaussian = gaussian / np.mean(gaussian)
    plt.imshow(gaussian)
    plt.show()
    hdsutils.writeHDSImage(outputfile, options.width, options.height, 1, gaussian[:,:,np.newaxis])