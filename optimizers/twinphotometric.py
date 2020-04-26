# Photometric stereo implementation.

import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

def lserror(x, L, I, y, n):
    print y, n
    N, R = unpack(x, y, n)
    print N.shape
    print R.shape
    print L.shape
    print I.shape

    err = np.concatenate([(I - np.matmul(R, np.matmul(L, N))).flatten(), np.array([np.sum(R) - n])], axis=0)
    print err.shape
    return err

def unpack(x, y, n):
    N = (x[:3 * y]).reshape((3,y)) # Normals
    R = np.diag(x[3 * y:]) # Radiances.
    return N, R

def pack(normals, intensities):
    return np.concatenate([normals.flatten(), intensities], axis=0)

def photometric(images, lights, darkstop=1, lightstop=2):
    """
    Inputs 
        images (NxWxH)
        lights (Nx3)
    Output - normals (WxHx3)
           - radiance (N)
    """

    # Simply solve for each pixel.
    print(images.shape)
    print(lights.shape)

    assert(images.shape[0] == lights.shape[0])

    N = images.shape[0]
    W = images.shape[1]
    H = images.shape[2]

    # Flatten last two dims
    # (NxY)
    images = images.reshape([images.shape[0], W*H])

    intensities = np.ones((N,))
    #intensities = np.mean(images, axis=1)
    intensities = intensities / np.mean(intensities)
    normals = None

    maxIterations = 1
    for i in range(maxIterations):
        print intensities
        loutput = np.linalg.lstsq(lights * intensities[:,np.newaxis], images)
        normals = loutput[0]
        print "Iteration " + format(i)
        reconstructed = np.matmul(lights * intensities[:,np.newaxis], normals)
        intensities = np.mean(reconstructed, axis=1)
        intensities = intensities / np.mean(intensities)
        print "Intensities: ", intensities
        print "E: ", np.sum(np.square(reconstructed - images))
        print "Error: ", np.sum(loutput[1])

    normals = normals.transpose().reshape((W,H,3))

    albedos = np.linalg.norm(normals, axis=2, keepdims=True)
    normals = normals / (albedos + 1e-5)

    plt.imshow(albedos.squeeze())
    plt.show()
    plt.imshow((normals + 1) * 0.5)
    plt.show()

    return normals, intensities