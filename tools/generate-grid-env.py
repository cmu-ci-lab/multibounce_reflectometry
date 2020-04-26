import sys
import os
import numpy as np
import OpenEXR
import array

W = 4096
H = 2048

sqside = 512

grid = np.ones((W, H, 3)) * 0.1

intensity = 4.0

for i in range(W // (sqside)):
    for j in range(H // (sqside)):
        if (i + j) % 2 == 0:
            grid[i*sqside:(i+1)*sqside, j*sqside:(j+1)*sqside, :] = intensity

out = OpenEXR.OutputFile("grid.exr", OpenEXR.Header(W, H))
(Rs, Gs, Bs) = [ array.array('f', chan.reshape((W*H)).tolist()).tostring() for chan in (grid[:,:,0], grid[:,:,1], grid[:,:,2])]
out.writePixels({'R' : Rs, 'G' : Gs, 'B' : Bs})
