import shdsutils
import numpy as np
import matplotlib.pyplot as plt
import optparse

parser = optparse.OptionParser()
parser.add_option("-w", "--weights", dest="weights", default=0, type="int")
parser.add_option("-t", "--tabular-weights", dest="tabularWeights", default="(90,90,1)", type="string")
parser.add_option("-s", "--super-iteration", dest="superiteration", default=0, type="int")
parser.add_option("-l", "--log", dest="log", action="store_true", default=False)

(options, args) = parser.parse_args()

weights = options.weights
tabularWeights = eval(options.tabularWeights)
filename = args[0]

npdata, wtdata, tabdata = shdsutils.loadSHDS(filename, numWeights=weights, tabularSize=tabularWeights)

if options.log:
    tabdata = np.log(np.abs(tabdata[:,:,0]))
else:
    tabdata = tabdata[:,:,0]

plt.imshow(tabdata)
plt.show()