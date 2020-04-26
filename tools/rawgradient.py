# Renders raw gradient provided a gradient-scene.xml and a mesh.
# Uses a unit reductor image to extract the unweighted gradient.

import hdsutils
import shdsutils
import monitor.load_normals as normals
from shutil import copyfile
import rendernormals
import np2exr
import sys
import numpy as np
import os
import splitpolarity

print("Do not use this file. It hasn't been updated and it may break something")
assert(False)

width = 256
height = 256
channels = 1

scenename = sys.argv[1]
meshname = sys.argv[2]
colorsscenename = sys.argv[3]

# Setup reductor image
hdsutils.writeHDSImage("/tmp/rwg-reductor.hds", width, height, channels, np.ones([width, height, channels]))

# Setup mesh file
copyfile(meshname, "/tmp/rwg-srcmesh.ply")

shdsname = scenename.replace(".xml", "") + "-" + meshname.split("/")[-1].replace(".ply", ".shds")
os.system("mitsuba " + scenename + " -o " + shdsname + " -Ddepth=-1 -DsampleCount=128 -DlightX=0.30 -DlightY=0.30 -DlightZ=-0.94 -Dreductor=/tmp/rwg-reductor.hds -Dmesh=/tmp/rwg-srcmesh.ply -Dweight1=0.5 -Dweight2=0.5 -Dalpha=0.3 -Dsfilter=0 -Dtfilter=0")

# Load gradients per-vertex.
gradients = shdsutils.loadSHDS(shdsname)

emplacedmesh = meshname.replace(".ply", ".g.ply")

# Create mesh with gradients.
normals.emplace_normals_as_colors(meshname, emplacedmesh, gradients, asfloat=True)

# Setup mesh file
copyfile(emplacedmesh, "/tmp/rwg-srcmesh.xml")

# Render this image as a full precision composite.
ndnegplyfile, ndposplyfile = splitpolarity.makePlyNames(emplacedmesh)
splitpolarity.splitPolarity(emplacedmesh, ndnegplyfile, ndposplyfile)
ndneghdsfile, ndnegnpyfile = rendernormals.makeRenderNames(ndnegplyfile)
ndposhdsfile, ndposnpyfile = rendernormals.makeRenderNames(ndposplyfile)
rendernormals.renderMesh(ndnegplyfile, ndneghdsfile, ndnegnpyfile, colorsscenename, lazy=False)
rendernormals.renderMesh(ndposplyfile, ndposhdsfile, ndposnpyfile, colorsscenename, lazy=False)

# Put this together into a multi-channel EXR.
ndexrfile = emplacedmesh.replace(".ply", ".exr")
np2exr.developCompositeFromFiles(
    ndnegnpyfile,
    ndposnpyfile,
    [
        ndexrfile.replace(".exr", ".x.exr"),
        ndexrfile.replace(".exr", ".y.exr"),
        ndexrfile.replace(".exr", ".z.exr")],
    mode="XYZ")