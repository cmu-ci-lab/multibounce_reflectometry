# Photometric stereo implementation.

import numpy as np
import scipy.optimize

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
    normals = np.ones((3, W*H))

    print pack(normals, intensities).shape[0]
    x1 = scipy.optimize.leastsq(lserror, x0=pack(normals, intensities), args=(lights, images, W*H, N))
    normals, intensities = unpack(x1)
    
    intensities = np.diag(intensities)
    normals = normals.transpose().reshape((W,H,3))

    return normals / (np.linalg.norm(normals, axis=2, keepdims=True) + 1e-5), intensities