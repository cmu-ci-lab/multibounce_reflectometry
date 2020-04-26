import struct
import sys
import numpy as np
import histutils
import optparse
import matplotlib.pyplot as plt

parser = optparse.OptionParser()

parser.add_option("-n", "--by-hn", action="store_true", dest="byHN", default=False)
parser.add_option("-v", "--by-hv", action="store_true", dest="byHV", default=False)
parser.add_option("-p", "--by-pathlength", action="store_true", dest="byPathLength", default=False)
parser.add_option("-d", "--direct-only", action="store_true", dest="directOnly", default=False)
parser.add_option("-i", "--indirect-only", action="store_true", dest="indirectOnly", default=False)
parser.add_option("-w", "--weighted", action="store_true", dest="weightedPaths", default=False)

(options, args) = parser.parse_args()

histfile = args[0]

histogram = histutils.loadHistogramFile(histfile)
print histogram.shape

if options.weightedPaths:
        histogram = histogram[:,:,:,:,0]
else:
        histogram = histogram[:,:,:,:,1]

if options.directOnly:
        # reset s+t > 3
        for m,n in [(i, d-i) for d in range(4,14) for i in range(d+1) if i < 7 and d-i < 7]:
                histogram[m,n,:,:] = 0
elif options.indirectOnly:
        # reset s+t == 3
        for m,n in [(i, d-i) for d in range(0,4) for i in range(d+1) if i < 7 and d-i < 7]:
                histogram[m,n,:,:] = 0
        #histogram[2,1,:,:] = 0
        #histogram[:2,:2,:,:] = 0

print histogram.shape
if options.byHV:
        plt.bar(
                np.array(range(histogram.shape[2]))/float(histogram.shape[2]),
                np.sum(histogram, axis=(0,1,3)), width=0.8/histogram.shape[2]
        )
elif options.byHN:
        plt.bar(
                np.array(range(histogram.shape[3]))/float(histogram.shape[3]),
                np.sum(histogram, axis=(0,1,2)), width=0.8/histogram.shape[3]
        )
elif options.byPathLength:
        bypath = np.zeros((14,))
        for i in range(bypath.shape[0]):
                for k in range(i+1):
                        if k >= 7 or i-k >= 7:
                                continue
                        bypath[i] += np.sum(histogram[k, i-k, :, :])
        bypath = bypath[2:]
        plt.bar(range(bypath.shape[0]), bypath)
else:
        print("Pick a dimension")

plt.show()