# Compares finite-differences with monte-carlo gradients

from dictionary_embedded import embedDictionary
import json
import numpy as np
import sys
import os
from shutil import copyfile
import hdsutils
import optparse
import itertools

import matplotlib.pyplot as plt

from dataset_reader import Dataset, Testset, toMap, mergeMaps

import load_normals

import splitpolarity
import rendernormals

parser = optparse.OptionParser()
parser.add_option("-s", "--super-index", dest="superindex", default=0)
parser.add_option("-a", "--all", dest="all", action="store_true", default=False)

(options, args) = parser.parse_args()

if not options.all:
    command = "ffmpeg -framerate 5 -i " + args[0] + "/renders/normals/" + format(options.superindex).zfill(2) + "/%04d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p " + args[0] + "/renders/normals/" + format(options.superindex).zfill(2) + "/normals.mp4"
    print(command)
    os.system(command)
else:
    nfiles = ""
    for i in itertools.count():
        if os.path.exists(args[0] + "/renders/normals/" + format(i).zfill(2)):
            command = "ffmpeg -framerate 5 -y -i " + args[0] + "/renders/normals/" + format(i).zfill(2) + "/%04d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p " + args[0] + "/renders/normals/" + format(i).zfill(2) + "/normals.mp4"
            nfile = args[0] + "/renders/normals/" + format(i).zfill(2) + "/normals.mp4"
            print(command)
            os.system(command)
        else:
            break
        nfiles += "file '" + nfile + "'" + '\n'

    nfiles = nfiles[:-1]

    open("nvideo-list.txt", "w").write(nfiles)

    command = "ffmpeg -y -safe 0 -f concat -i nvideo-list.txt -c copy " + args[0] + "/renders/normals/normals.mp4"
    print(command)
    os.system(command)