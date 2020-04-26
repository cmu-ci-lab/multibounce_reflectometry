import merl_io
import optparse
import os
import sys

import matplotlib.pyplot as plt

parser = optparse.OptionParser()
parser.add_option("-r", "--threshold", dest="threshold", type="int")
(options, args) = parser.parse_args()

bfile = args[0]

table = merl_io.merl_read(bfile)

print(merl_io.merl_read_raw(bfile)[15817])

if (table < 0).any():
    print("Found negatives")
print(table[0,0,0])

if options.threshold is not None:
    table[table > options.threshold] = options.threshold

plt.imshow(table[:,:,table.shape[2]//2])
plt.show()