# Photometric stereo implementation.

import numpy as np

def lserror(x, L, I, y, n):
    N = (x[:3 * y]).reshape((3,y)) # Normals
    R = np.diag(x[3 * y:]) # Radiances.
    return np.flatten(I - np.matmul(R, np.matmul(L, N)))


def photometric(images, lights, darkstop=0, lightstop=0):
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

    lnormals = []
    # 3xY
    for i in range(images.shape[1]):
        #if i == 5000:
        #print(lights.shape)
        #print(images[:,i].shape)
        if i % 10000 == 0:
            print i, "/", images.shape[1]

        L = lights
        I = images[:,i]

        idxs = np.argsort(I)
        #print idxs
        idxs = idxs[darkstop:len(idxs)-lightstop-1]
        L = lights[idxs, :]
        I = I[idxs]

        pnormals, residual, rank, s = np.linalg.lstsq(L, I)
        if np.linalg.norm(pnormals) > 1e-6:
            lnormals.append(pnormals)
        else:
            lnormals.append(np.array([0, 0, 1]))
        #lnormals.append(np.array([0, 0, 1]))

    normals = np.stack(lnormals, axis=1)
    #normals, residual, rank, s = np.linalg.lstsq(lights, images)

    normals = normals.transpose().reshape((W,H,3))

    return normals / (np.linalg.norm(normals, axis=2, keepdims=True) + 1e-5), np.ones((N,))