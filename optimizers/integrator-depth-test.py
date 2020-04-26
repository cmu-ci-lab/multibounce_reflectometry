# Unit test to check if the integrator is working properly.

import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import optparse
from remesher.algorithms.poisson.integrator import integrate as poisson_integrator

# Test flat plate without any masked pixels.

def testFlatNoMask(integrate):
    mask = np.ones((256, 256))

    normals = np.stack([
        np.zeros_like(mask),
        np.zeros_like(mask),
        np.ones_like(mask)
    ], axis=2)

    zfield = integrate(normals, 0.0, zflipped=False, mask=np.squeeze(mask))

    return zfield

def testSlopeNoMask(integrate):
    mask = np.ones((256, 256))

    normals = np.stack([
        np.zeros_like(mask),
        np.ones_like(mask),
        np.ones_like(mask)
    ], axis=2)

    zfield = integrate(normals, 0.0, zflipped=False, mask=np.squeeze(mask))

    return zfield

def testSlopeMasked(integrate):
    mask = np.ones((256, 256))

    mask[:50,:] = 0
    mask[:50,:] = 0

tests = [testFlatNoMask, testSlopeNoMask]

parser = optparse.OptionParser()
parser.add_option("-p", "--plot-only", dest="plotOnly", action="store_true", default=False)
(options, args) = parser.parse_args()

for test in tests:
    print "Running test: ", test.__name__
    result = test(poisson_integrator)
    if options.plotOnly:
        plt.imshow(result, cmap='viridis')
        plt.show()
    else:
        print "Test result gilding features unavailable at this time."
        sys.exit(1)