import os
import sys
import json
import scipy.io as sio
import numpy as np

def renderOriginals(directory, config, extension="exr", fileFormat="openexr"):

    scenefile  = directory + "/" + config["initial-reconstruction"]["scene"]
    meshfile = directory + "/" + config["initial-reconstruction"]["mesh"]

    scene = directory + "/originals/intensity-scene.xml"

    lightsfile = directory + "/" + config["initial-reconstruction"]["lights"]

    serverstring = " -c "
    if config["distribution"]["enabled"]:
        for server in config["distribution"]["servers"]:
            serverstring += server + ";"

    serverstring = serverstring[:-1]

    # TODO: Temporary.
    serverstring = ""

    lights = open(lightsfile, "r").readlines()
    lts = [ll.split(" ")[1:4] for ll in lights]
    flts = [ [float(k) for k in lt] for lt in lts ]
    print(flts)

    #flts = [ [flt[2], -flt[0], -flt[1]] for flt in flts ]
    flts = [ [flt[0], flt[1], flt[2]] for flt in flts ]

    for i, lt in enumerate(lts):
        lt = [t.strip() for t in lt]
        if not "-lonly" in sys.argv:
            print("mitsuba " + scenefile + " -o " + directory + "/originals/" + format(i).zfill(4) + "." + extension + " -DlightX=" + lt[0] + " -DlightY=" + lt[1] + " -DlightZ=" + lt[2] + " -Ddepth=-1 -DsampleCount=1024 -Dmesh=" + meshfile + " -Dext=" + fileFormat + " " + serverstring)
            os.system("mitsuba " + scenefile + " -o " + directory + "/originals/" + format(i).zfill(4) + "." + extension + " -DlightX=" + lt[0] + " -DlightY=" + lt[1] + " -DlightZ=" + lt[2] + " -Ddepth=-1 -DsampleCount=1024 -Dmesh=" + meshfile + " -Dext=" + fileFormat + " " + serverstring)

    print("Lights ", flts)
    sio.savemat(directory + "/originals/sources.mat", {"S":flts})

if __name__ == "__main__":
    directory = sys.argv[1]

    extension = "exr"
    if extension == "exr":
        fileFormat = "openexr"

    configfile = directory + "/config.json"
    config = json.load(open(configfile, "r"))

    renderOriginals(directory, config, extension, fileFormat)