import hdsutils
import numpy as np
import matplotlib.pyplot as plt
import optparse

parser = optparse.OptionParser()
parser.add_option("-l", "--log", dest="log", action="store_true", default=False)

(options, args) = parser.parse_args()

filename = args[0]

npdata = hdsutils.loadHDSImage(filename)

if options.log:
    npdata = np.log(np.abs(npdata[:,:,0]))
else:
    npdata = npdata[:,:,0]

plt.imshow(npdata)
plt.show()