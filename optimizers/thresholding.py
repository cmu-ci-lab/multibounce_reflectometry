import numpy as np

def clipDarkRegions(data, num=2):
    """
    data: WxHxN
    num: Number of images to clip
    Produces a weight mask to clip dark regions of the image.
    """

    weightmask = np.ones_like(data)

    indices = np.argsort(data, axis=2)
    for i in range(num):
        x,y = np.ogrid[0:data.shape[0], 0:data.shape[1]]
        weightmask[x, y, indices[:,:,i]] = 0

    return weightmask

def clipBrightRegions(data, num=2):
    """
    data: WxHxN
    num: Number of images to clip
    Produces a weight mask to clip bright regions of the image.
    """

    weightmask = np.ones_like(data)

    indices = np.argsort(data, axis=2)

    # Invert indices.
    indices = indices[:,:,::-1]

    for i in range(num):
        x,y = np.ogrid[0:data.shape[0], 0:data.shape[1]]
        weightmask[x, y, indices[:,:,i]] = 0

    return weightmask

def intensityWeights(data, gamma=1.0):
    """
    data: WxHxN
    Inverse square root weighting.
    """
    epsilon = 1e-5
    weightmask = np.power(data, gamma) + epsilon
    mask = data > 1e-3

    return (1.0 / weightmask) * mask