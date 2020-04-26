import numpy as np
import matplotlib.pyplot as plt
import optparse

parser = optparse.OptionParser()
parser.add_option("-e", "--exposure", dest="exposure", default=1.0, type="float")
parser.add_option("-r", "--threshold", dest="threshold", default=1000.0, type="float")
(options, args) = parser.parse_args()

fname = args[0]


img = np.load(fname)
if len(img.shape) == 3 and img.shape[2] == 1:
    img = img[:,:,0]

img[img > options.threshold] = options.threshold

plt.imshow(img)
plt.show()