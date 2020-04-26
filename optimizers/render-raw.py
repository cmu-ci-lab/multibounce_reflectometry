# Compares finite-differences with monte-carlo gradients

from dictionary_embedded import embedDictionary
import json
import numpy as np
import sys
import os
from shutil import copyfile
import hdsutils
import optparse

import matplotlib.pyplot as plt

from dataset_reader import Dataset, Testset, toMap, mergeMaps

import load_normals

import splitpolarity
import rendernormals

import xml.etree.ElementTree as ET
import sys

def getProperty(tag, name, fmt):
    taglist = [ s for s in tag.findall(fmt) if s.get('name') == name ]
    if len(taglist) > 1:
        print "Too many matches for name ", name, " of format ", fmt
    if len(taglist) == 0:
        print "No matches for name ", name, " of format ", fmt
    return taglist[0]

def xmlAddGradientSliceRendering(renderable, slices=0):
    filename = renderable.getFile()
    tree = ET.parse(filename)
    root = tree.getroot()
    film = root.find('sensor').find('film')
    if film.get('type') != "hdrreductorfilm":
        print "Invalid film type ", film.get('type'), " in file ", filename
        sys.exit(1)
    # Remove reductor
    taglist = [ s for s in film.findall('string') if s.get('name') == 'reductorFile' ]
    if len(taglist) != 1:
        print("Couldn't find reductorFile tag ")
        sys.exit(1)
    film.remove(taglist[0])
    integrator = root.find('integrator')

    ignoreIndex = ET.SubElement(film, 'integer')
    ignoreIndex.set('name', 'ignoreIndex')
    ignoreIndex.set('value', '$ignoreIndex')

    ignoreIndex2 = ET.SubElement(integrator, 'integer')
    ignoreIndex2.set('name', 'ignoreIndex')
    ignoreIndex2.set('value', '$ignoreIndex')

    sampler = root.find('sensor').find('sampler')
    if sampler.get('type') == 'varying':
        sampler.set('type', 'ldsampler')
        sampler.remove(getProperty(sampler, 'sampleMultiplier', 'float'))
        sampler.remove(getProperty(sampler, 'samplerFile', 'string'))
        sampleCount = ET.SubElement(sampler, "integer")
        sampleCount.set('name', 'sampleCount')
        sampleCount.set('value', '$sampleCount')
    else:
        print("sampler invalid")
        sys.exit(1)

    newfilename = filename + ".slices-" + format(slices) + ".xml"
    tree.write(filename + ".slices-" + format(slices) + ".xml")
    renderable.setFile(newfilename)
    renderable.setEmbeddedParameter("ignoreIndex", slices)

    return renderable
    # Add in ignoreIndex term.

parser = optparse.OptionParser()
parser.add_option("-d", "--display", dest="display", action="store_true", default=False)
parser.add_option("-c", "--distribution", dest="distribution", default=None, type="string")
parser.add_option("-r", "--samples", dest="samples", default=128, type="int")
parser.add_option("-i", "--iteration", dest="iteration", default=None, type="int")
parser.add_option("-s", "--super-iteration", dest="superiteration", default=0, type="int")
parser.add_option("--layout-columns", dest="layoutColumns", default=5, type="int")
parser.add_option("--layout-rows", dest="layoutRows", default=5, type="int")
parser.add_option("--adaptive-mode", dest="adaptiveMode", default=None, type="string")

(options, args) = parser.parse_args()

directory = args[0]

dataset = Dataset(directory)
testset = dataset.testSet()

gdirpath = directory + "/images/raw-bsdf-gradients"

lr = options.layoutRows
lc = options.layoutColumns

if not os.path.exists(gdirpath):
    os.mkdir(gdirpath)

for k in range(dataset.testSet().numLights()):
    xmlAddGradientSliceRendering(testset.gradientRenderables[k], slices=len(testset.parameterList()))

for i in range(dataset.testSet().numBSDFIterations()):
    if options.iteration is not None and i != options.iteration:
        continue

    gsubdir = gdirpath + "/" + format(options.superiteration).zfill(2)

    if not os.path.exists(gsubdir):
        os.mkdir(gsubdir)

    if options.superiteration >= 1:
        meshfile = dataset.meshfileAt(iteration=0, superiteration=options.superiteration-1)
    else:
        meshfile = dataset.testset.initialMeshPath()

    if meshfile is None:
        print("Couldn't find mesh for ", options.superiteration, "-", i)
        break

    paramlist = testset.parameterList()
    bsdf = toMap(paramlist, dataset.BSDFAt(iteration=i, superiteration=options.superiteration))
    if options.adaptiveMode is None:
        bsdfSamples = toMap(dataset.bsdfAdaptiveSamplingParameterList, dataset.BSDFAt(iteration=i, superiteration=options.superiteration))
    elif options.adaptiveMode == "invAlpha":
        epsilon = 1e-3
        wts = []
        for element in testset.bsdfDictionaryElements:
            if element["type"] == "roughconductor":
                wts.append(1/(element["alpha"] + epsilon))
            elif element["type"] == "diffuse":
                wts.append(1.0)
            else:
                wts.append(1.0)
        return np.array(wts)
        bsdfSamples = toMap(dataset.bsdfAdaptiveSamplingParameterList, wts)


    for k in range(dataset.testSet().numLights()):
        print "Iteration ", i, "/", testset.numBSDFIterations(), ", Light ", k, "/", dataset.testSet().numLights(), "\r",
        sys.stdout.flush()

        testset.gradientRenderables[k].setEmbeddedParameter("sampleCount", options.samples)
        testset.gradientRenderables[k].setParameter("blockSize", 8)

        copyfile(meshfile,
                            "/tmp/mts_mesh_gradient_slot_" + format(k) + ".ply")

        testset.gradientRenderables[k].setEmbeddedParameter("depth", -1)
        testset.gradientRenderables[k].setEmbeddedParameter("meshSlot", k)
        de = dataset.errorAtN(lindex=k, iteration=i, superiteration=options.superiteration)
        width = de.shape[0]
        height = de.shape[1]
        testset.gradientRenderables[k].setEmbeddedParameter("width", width)
        testset.gradientRenderables[k].setEmbeddedParameter("height", height)

        hdsutils.writeHDSImage("/tmp/reductor-" + format(k) + ".hds", de.shape[0], de.shape[1], 1, de[:,:,np.newaxis])
        hdsutils.writeHDSImage("/tmp/sampler-" + format(k) + ".hds", de.shape[0], de.shape[1], 1, np.ones_like(de)[:,:,np.newaxis])

        gradientSlices = testset.gradientRenderables[k].renderReadback(
            readmode="raw",
            distribution=options.distribution,
            output="/tmp/raw-gradients-" + format(k) + ".raw",
            embedded=mergeMaps(bsdf, bsdfSamples),
            localThreads=6,
            quiet=False)

        if options.display:
            fig = plt.figure(figsize=(lr, lc))
            for k in range(gradientSlices.shape[0]):
                if k >= (lr * lc):
                    print "Not enough spots to display all slices. Slices remaining: ", (gradientSlices.shape[0] - k) + 1
                    break
                ax = fig.add_subplot(lr, lc, k+1)
                ax.set_title(testset.bsdfDictionaryElementStrings[k])
                plt.imshow(gradientSlices[k,:,:])
            plt.show()

        np.save(gsubdir + "/" + format(i).zfill(4) + "-" + format(k).zfill(2) + ".npy", gradientSlices)