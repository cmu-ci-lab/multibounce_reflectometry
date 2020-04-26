# z2mesh.py

# Converts a heightfield to a mesh.
import numpy as np

def t2v(t):
    return np.array([t[0], t[1], t[2]]);
# Input: numpy 2D matrix holding height values and x,y bounds.
# Output: A mesh with one vertex per element. format=(vertexlist, indexlist)
def z2mesh(Z, xstart, xend, ystart, yend, flip=False):
    sz = Z.shape

    xdim = sz[0]
    ydim = sz[1]
    assert(xdim > 1 and ydim > 1)

    vertexlist = []
    indexlist = []
    for j in range(ydim):
        for i in range(xdim):
            idx = len(vertexlist)

            # Compute vertex coordinates.
            #xcoord = (float(i) / (xdim - 1)) * (xend - xstart) + xstart
            #ycoord = (float(j) / (ydim - 1)) * (yend - ystart) + ystart

            # Compute bias-corrected vertex coordinates.
            # THIS IS IMPORTANT. MESH WILL BEGIN TO SLOWLY DRIFT OTHERWISE.
            xcoord = ((float(i) / xdim) * (xend - xstart) + xstart) + ((xend - xstart)/(2 * xdim))
            ycoord = ((float(j) / ydim) * (yend - ystart) + ystart) + ((xend - xstart)/(2 * ydim))
            zcoord = Z[j,i]
            vertexlist.append((xcoord, ycoord, zcoord))

            # Compute indices for two triangles.
            if i == xdim - 1 or j == ydim - 1:
                continue;

            a0 = idx
            a1 = idx+xdim+1
            a2 = idx+1

            b0 = idx
            b1 = idx+xdim
            b2 = idx+xdim+1

            if not flip:
                indexlist.append((a1, a0, a2))
                indexlist.append((b1, b0, b2))
            else:
                indexlist.append((a0, a1, a2))
                indexlist.append((b0, b1, b2))

    normals = np.zeros((xdim * ydim, 3));
    normalcounts = np.zeros((xdim * ydim,));
    for t in indexlist:
        n = np.cross(t2v(vertexlist[t[0]]) - t2v(vertexlist[t[1]]), t2v(vertexlist[t[2]]) - t2v(vertexlist[t[1]]));
        n = n / np.linalg.norm(n);

        nc0 = normalcounts[t[0]]
        nc1 = normalcounts[t[1]]
        nc2 = normalcounts[t[2]]
        assert(nc0 <= 7)
        assert(nc1 <= 7)
        assert(nc2 <= 7)
        normals[t[0]] = (normals[t[0]] * nc0 + n * 1) / (nc0 + 1)
        normals[t[1]] = (normals[t[1]] * nc1 + n * 1) / (nc1 + 1)
        normals[t[2]] = (normals[t[2]] * nc2 + n * 1) / (nc2 + 1)
        normalcounts[t[0]] += 1
        normalcounts[t[1]] += 1
        normalcounts[t[2]] += 1

    vertexlist = np.array(vertexlist);
    indexlist = np.array(indexlist);
    return (vertexlist, normals, indexlist);
