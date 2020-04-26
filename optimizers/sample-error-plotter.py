# Sample error plotter.

import matplotlib.pyplot as plt
import numpy as np
import optparse

from dataset_reader import Dataset, Testset, toMap, mergeMaps
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class LossFn:
    def __init__(self, lights, alpha):
        self.lights = np.array(lights)
        self.alpha = alpha

    def L(self, nx, ny, tx, ty):
        normals = np.stack([nx, ny, np.sqrt(1 - (nx**2 + ny**2))], axis=len(nx.shape))
        normals = normals / np.linalg.norm(normals, axis=-1, keepdims=True)
        tnormals = np.stack([tx, ty, np.sqrt(1 - (tx**2 + ty**2))], axis=len(tx.shape))
        tnormals = tnormals / np.linalg.norm(tnormals, axis=-1, keepdims=True)
        print(normals.shape)
        print(self.lights.shape)
        dotproducts = np.tensordot(normals, self.lights, axes=[-1,-1])
        tdotproducts = np.tensordot(tnormals, self.lights, axes=[-1,-1])
        print dotproducts.shape
        print tdotproducts.shape
        dps = self.bsdf(dotproducts)
        print(tdotproducts)
        print(self.bsdf(tdotproducts))
        x = self.bsdf(dotproducts)
        t = self.bsdf(tdotproducts[0,:])
        return self.loss(x * (1-dotproducts)**2,t * (1-dotproducts)**2)

    def loss(self, x, t):
        return np.sum((x - t)**2, axis=-1)
        #return np.sum(t * np.log(x/t), axis=-1)

    def bsdf(self, d):
        return (self.alpha**2) / ((d**2) * ((self.alpha**2) - 1) + 1)**2


parser = optparse.OptionParser()

(options, args) = parser.parse_args()

directory = args[0]

testset = Testset(directory)

loss = LossFn(testset.lightDirections(), 0.5)

res = 400
nx = np.tile(np.linspace(-0.7, 0.7, num=res)[:,np.newaxis], [1,res])
ny = np.tile(np.linspace(-0.7, 0.7, num=res)[np.newaxis,:], [res,1])

#ny = np.linspace(-0.6, 0.6, num=50)
tx = np.array([0.4])
ty = np.array([-0.6])
errors = loss.L(nx, ny, tx, ty)

errors = np.log(errors)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surface = np.array([ [errors[x,y] for x in range(errors.shape[0])] for y in range(errors.shape[1]) ])
x, y = np.meshgrid(range(errors.shape[0]), range(errors.shape[1]))
ax.plot_surface(x, y, surface)
plt.show()

"""
plt.imshow(errors)
plt.show()
"""