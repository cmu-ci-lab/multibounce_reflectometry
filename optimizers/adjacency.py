
import numpy as np
import sys
import matplotlib.pyplot as plt

class MeshAdjacencyBuilder:

    @classmethod
    def buildAdjacency(cls, vertices, radius=2.0, width=256, height=256):
        """
            Returns a weight matrix of size VxV (V is the number of vertices)
            based on the closeness of the vertex.
        """

        numVertices = vertices.shape[0]
        #wtmat = np.zeros((numVertices, numVertices))
        #for idx in range(numVertices):
        #    wtmat[idx] = self._adjacencyAt(vertices, idx, radius=radius, width=width, height=height)

        xvals = vertices[:, 0]
        yvals = vertices[:, 1]
        
        xmatrix = np.tile(xvals[:,np.newaxis], [1, numVertices])
        ymatrix = np.tile(yvals[np.newaxis,:], [numVertices, 1])

        x2matrix = np.tile(xvals[np.newaxis, :], [numVertices, 1])
        y2matrix = np.tile(yvals[:, np.newaxis], [1, numVertices])

        return 2.0 * radius - np.sqrt(((xmatrix - x2matrix) ** 2) + ((ymatrix - y2matrix) ** 2))

    def _weight(vtx0, vtx1, radius):
        return 2.0 * radius - np.sqrt(((vtx0[0] - vtx1[0]) ** 2) + ((vtx0[1] - vtx1[1]) ** 2))


    def _adjacencyAt(vertices, i, radius=1.0, width=256, height=256):
        # Compute floating point guides
        x = vertices[i, 0]
        y = vertices[i, 1]
        fx = 2.0 * (float(x) / float(width)) - 1.0
        fy = -(2.0 * (float(y) / float(height)) - 1.0)
        deltax = ((2.0 * radius) / float(width))
        deltay = ((2.0 * radius) / float(height))

        xslice = np.logical_and(vertices[:,0] >= fx - (deltax/2), vertices[:,0] <= fx + (deltax/2))
        yslice = np.logical_and(vertices[:,1] >= fy - (deltay/2), vertices[:,1] <= fy + (deltay/2))
        xyblock = np.logical_and(xslice, yslice)

        (index,) = np.where(xyblock)

        row = np.zeros((vertices.shape[0],))
        for idx in index:
            row[idx] = MeshAdjacencyBuilder._weight(vertices[i, :], vertices[idx, :], radius)

        return row

    @classmethod
    def buildIndexMap(cls, vertices, radius=1.0, width=256, height=256):
        print("Building index map for ", vertices.shape[0], " vertices")
        idxMap = np.zeros((width, height), dtype=np.int64)
        validMask = np.zeros((width, height), dtype=np.bool)

        for i in range(vertices.shape[0]):
            y, x = MeshAdjacencyBuilder._indexOf(vertices, i, radius=radius, width=width, height=height)
            if x >= 0 and x < width and y >= 0 and y < height:
                if validMask[x, y]:
                    print "Index Builder: Fatal: Overlapping vertices at ", i, x, y, width, height, idxMap[x,y]
                    sys.exit(1)
                idxMap[x,y] = i
                validMask[x,y] = True

        return idxMap, validMask

    @classmethod
    def buildNormalMap(cls, vertices, normals, radius=1.0, width=256, height=256):
        indexMap, validMask = MeshAdjacencyBuilder.buildIndexMap(vertices, radius=radius, width=width, height=height)
        normalMap = np.stack([
            np.zeros((width,height)),
            np.zeros((width,height)),
            np.ones((width,height))
        ], axis=2)
        print(normalMap.shape)
        normalMap = np.where(validMask[:,:,np.newaxis], normals[indexMap, :], normalMap)
        print(normalMap.dtype)
        return normalMap

    @classmethod
    def buildVertexMap(cls, vertices, normals, radius=1.0, width=256, height=256):
        print("Building index map for ", vertices.shape[0], " vertices")
        idxMap = np.zeros((width, height), dtype=np.int64)
        validMask = np.zeros((width, height), dtype=np.bool)

        vmap = []
        for i in range(vertices.shape[0]):
            y, x = MeshAdjacencyBuilder._indexOf(vertices, i, radius=radius, width=width, height=height)
            vmap.append((x,y))
            """
            if x >= 0 and x < width and y >= 0 and y < height:
                if validMask[x, y]:
                    print "Index Builder: Fatal: Overlapping vertices at ", i, x, y, width, height, idxMap[x,y]
                    sys.exit(1)
                idxMap[x,y] = i
                validMask[x,y] = True
            """
        return np.array(vmap)

    @classmethod
    def _indexOf(cls, vertices, i, radius=1.0, width=256, height=256):
        x = vertices[i, 0]
        y = vertices[i, 1]
        fx = ((float(x) + 1) / 2.0) * width
        fy = ((float(y) + 1) / 2.0) * height
        
        px = int(fx)
        py = int(fy)
        return px, py