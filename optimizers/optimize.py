import sys
import os
import subprocess
import time
import normals
from remesher import remesh
import remesher.algorithms.poisson.integrator as poisson_integrator
import remesher.integrate as frankot_integrator
import parameters
from shutil import copyfile, copytree
import load_normals
from dictionary_embedded import embedDictionary
import json
import numpy as np
import restarter
from upscaler import downsampleImage, rescaleMesh
from createmask import renderMask
import mask_remesher
import matplotlib.pyplot as plt
import scipy.ndimage.morphology
import thresholding
import matplotlib.pyplot as plt
import optparse
import datetime
import time
import cv2
from adjacency import MeshAdjacencyBuilder

import imp
awsmanager = imp.load_source('module.name', os.path.dirname(__file__) +  "/../tools/awsmanager.py")


def getParameterMap(params, directory):
    runnerParameterMap = {}

    if type(params["hyper-parameter-list"]) is list:
        runnerParameterList = params["hyper-parameter-list"]
    else:
        runnerParameterList = json.load(open(params["hyper-parameter-list"], "r"))
        params["hyper-parameter-list"] = runnerParameterList

    if "hyper-parameters" in params["original"] and type(params["original"]["hyper-parameters"]) is unicode:
        runnerParameterValues = np.load(directory + "/" + params["original"]["hyper-parameters"])
        params["original"]["hyper-parameters"] = dict(zip(runnerParameterList, runnerParameterValues))

    if "hyper-parameters" in params["estimator"] and type(params["estimator"]["hyper-parameters"]) is unicode:
        runnerParameterValues = np.load(directory + "/" + params["estimator"]["hyper-parameters"])
        params["estimator"]["hyper-parameters"] = dict(zip(runnerParameterList, runnerParameterValues))

    if "hyper-parameters" in params["bsdf-estimator"] and type(params["bsdf-estimator"]["hyper-parameters"]) is unicode:
        runnerParameterValues = np.load(directory + "/" + params["bsdf-estimator"]["hyper-parameters"])
        params["bsdf-estimator"]["hyper-parameters"] = dict(zip(runnerParameterList, runnerParameterValues))

    # If map specified
    runnerParameterMap = None
    if "hyper-parameters" in params["original"] and type(params["original"]["hyper-parameters"]) is dict:
        runnerParameterMap = params["original"]["hyper-parameters"]
    elif "hyper-parameters" in params["estimator"] and type(params["estimator"]["hyper-parameters"]) is dict:
        runnerParameterMap = params["estimator"]["hyper-parameters"]

    # If file specified.
    runnerParameterValues = None
    if "hyper-parameters" in params["original"] and type(params["original"]["hyper-parameters"]) is unicode:
        runnerParameterValues = np.load(directory + "/" + params["original"]["hyper-parameters"])
    elif "hyper-parameters" in params["estimator"] and type(params["estimator"]["hyper-parameters"]) is unicode:
        runnerParameterValues = np.load(directory + "/" + params["estimator"]["hyper-parameters"])

    return runnerParameterMap, runnerParameterList, runnerParameterValues

def loadTabularBSDF(params, directory):
    if "tabular-bsdf" in params["bsdf-estimator"]:
        print("Loading Tabular BSDF:")
        tabular = np.load(directory + "/" + params["bsdf-estimator"]["tabular-bsdf"]["initialization"])
        params["bsdf-estimator"]["tabular-bsdf"]["initialization"] = tabular
        print("Loaded Tabular BSDF with dimensions: ", tabular.shape)

def getAdaptiveSampleMap(params, directory):
    if type(params["weight-samples-parameter-list"]) is list:
        runnerParameterList = params["weight-samples-parameter-list"]
    else:
        runnerParameterList = json.load(open(params["weight-samples-parameter-list"], "r"))
        params["weight-samples-parameter-list"] = runnerParameterList

    # Initialize to all ones
    runnerParameterValues = len(runnerParameterList) * [1.0]

    for k,v in zip(runnerParameterList, runnerParameterValues):
        runnerParameterMap[k] = v

    return runnerParameterMap, runnerParameterList, runnerParameterValues

def multires(params, si):
    if params is None:
        return 1

    if params["type"] == "static-list":
        for i, k in enumerate(params["schedule"]):
            if k > si:
                return params["values"][i]

    return 1

MAX_RESTARTS = 10

print("Parsing arguments")

makeCopy = True

parser = optparse.OptionParser()
parser.add_option("-s", "--distr-server-override", dest="distrServerOverride", default="")
parser.add_option("-c", "--distr-core-override", dest="distrCoreOverride", default="")
parser.add_option("-g", "--remote-git-update", action="store_true", dest="remoteGitUpdate", default=False)
parser.add_option("-e", "--empty-remote", action="store_true", dest="eraseRemote", default=False)
parser.add_option("-n", "--index", dest="index", default=0)
parser.add_option("-r", "--restart", dest="restartIndex", type="int", default=-2)
parser.add_option("-l", "--server-select", dest="serverSelect", type="int", default=-1)
parser.add_option("--restart-source", dest="restartSource", type="string", default="")
(options, args) = parser.parse_args()

if options.restartIndex != -2:
    restart = options.restartIndex
    isRestart = True
    print("Detected restart number: " + format(options.restartIndex))

    if restart != -1 and options.restartSource != "":
        sourceDir = options.restartSource
        print("Detected restart source directory: " + options.restartSource)
    elif restart != -1:
        print("Restart number is non negative. Restart source directory must be present")
        sys.exit(1)

    if not len(os.listdir("./")) == 0 and restart != -1:
        print("Restart number is non negative. Current directory must be empty")
        sys.exit(1)

    if restart != -1:
        restarter.midcopy(sourceDir, "./", restart)
    else:
        print("Restarting from current directory")
        restart = len([k for k in os.listdir("./errors") if k.startswith("errors-") and k.endswith(".json")])-1
        print("Detected last complete superiteration: ", restart)

else:

    print("No restart number. Fresh start")
    restart = None
    isRestart = False

    # Check for directories.
    #if not len(os.listdir("./")) == 0:
    #    print("No restart number detected, but the current directory is populated. Empty current directory first.")
    #    sys.exit(1)

configparam = args[0]
if sys.argv[1] == "local":
    print("Requested local inputs")
    if not os.path.exists("./inputs/config.json"):
        print("Couldn't detect local config. Aborting")
        sys.exit(1)
    configparam = "./inputs/config.json"
    makeCopy = False

# Load and prepare parameters
params, directory = parameters.loadParameters(configparam)

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

print("Making a copy of the inputs")
if makeCopy:
    if not os.path.exists("inputs/"):
        copytree(directory, "inputs/")
    else:
        print("Inputs already exists. Making a duplicate copy.")
        for i in range(MAX_RESTARTS):
            istring = "inputs-" + format(i).zfill(2) + "/"
            if not os.path.exists(istring):
                print("Saving inputs as ", istring)
                copytree(directory, istring)
                break

# #RUNINFO
# Place a timestamp in the run info.
runInfo = {
    "timestamp": time.mktime(datetime.datetime.now().timetuple())
}

json.dump(runInfo, open("run.info", "w"))

if "bsdf-adaptive" not in params["bsdf-estimator"]["samples"]:
    params["bsdf-estimator"]["samples"]["bsdf-adaptive"] = False

bsdfAdaptiveSampling = params["bsdf-estimator"]["samples"]["bsdf-adaptive"]

# Pre-process the XML files.
bsdfDictionary = {}
if "bsdf-preprocess" in params and params["bsdf-preprocess"]["enabled"]:
    ixml = params["scenes"]["intensity"]
    ixmlout = ixml + ".pp.xml"

    gxml = params["scenes"]["gradient"]
    gxmlout = gxml + ".pp.xml"

    print(type(params["hyper-parameter-list"]))

    if not type(params["hyper-parameter-list"]) is unicode:
        print("hyper-parameter-list has been specified while bsdf preprocessing is active. Aborting")
        sys.exit(1)

    paramsout = params["hyper-parameter-list"]
    paddingout = params["zero-padding"]
    dfile = directory + "/" + params["bsdf-preprocess"]["file"]

    if bsdfAdaptiveSampling:
        sampleparamsout = params["weight-samples-parameter-list"]
    else:
        sampleparamsout = None

    bsdfDictionary = json.load(open(dfile, "r"))

    embedDictionary(dfile, directory + "/" + ixml, directory + "/" + ixmlout, paramsout, paddingout, bsdfAdaptiveSampling, sampleparamsout, True)
    embedDictionary(dfile, directory + "/" + gxml, directory + "/" + gxmlout, paramsout, paddingout, bsdfAdaptiveSampling, sampleparamsout, False)

    # Replace the original values with the modified files.
    params["scenes"]["intensity"] = ixmlout
    params["scenes"]["gradient"] = gxmlout
# --

# Set up pipes
print("Initializing FIFO pipes")
print("mkfifo " + params["output-link"]["intensity"])
print("mkfifo " + params["output-link"]["gradient"])

os.system("rm " + params["output-link"]["intensity"])
os.system("rm " + params["output-link"]["gradient"])

os.system("mkfifo " + params["output-link"]["intensity"])
os.system("mkfifo " + params["output-link"]["gradient"])
# --


scriptsDirectory = os.path.dirname(__file__)

runnerParameterMap, runnerParameterList, runnerParameterValues = getParameterMap(params, directory)
if bsdfAdaptiveSampling:
    runnerSamplesParameterMap, runnerSamplesParameterList, runnerSamplesParameterValues = getAdaptiveSampleMap(params, directory)

loadTabularBSDF(params, directory)

runnerParameters = "-DmeshSlot=0 -Ddepth=1 -DsampleCount=1 -Dwidth=256 -Dheight=256 -DlightX=0.0 -DlightY=0.0 -DlightZ=0.0 -Dirradiance=10.0 "

if runnerParameterMap is not None:
    for pkey in runnerParameterList:
        runnerParameters += "-D" + pkey + "=" + format(runnerParameterMap[pkey]) + " "
elif runnerParameterValues is not None:
    for pkey, pval in zip(runnerParameterList, runnerParameterValues):
        runnerParameters += "-D" + pkey + "=" + format(pval) + " "

# Add padding parameters if necessary
# Zero padding.
if "zero-padding" in params and type(params["zero-padding"]) is int:
    zeroPadding = params["zero-padding"]
elif "zero-padding" in params:
    zeroPadding = json.load(open(params["zero-padding"], "r"))[0]
else:
    zeroPadding = 0

for i in range(zeroPadding):
    runnerParameters += "-Dpadding" + format(i).zfill(4) + "=0 "

# Add adaptive sampling parameters to the launch string.
if bsdfAdaptiveSampling:
    for pkey in runnerSamplesParameterList:
        runnerParameters += "-D" + pkey + "=" + format(runnerSamplesParameterMap[pkey]) + " "

if not "type" in params["distribution"]:
    params["distribution"]["type"] = "single"

intensityOut = None
intensityErr = None
gradientErr = None
gradientOut = None

if not os.path.exists("logs"):
    os.mkdir("logs")

# Set up logging.
if "intensity" in params["logging"]:
    if "stdout" in params["logging"]["intensity"]:
        intensityOut = open("logs/" + params["logging"]["intensity"]["stdout"], "w")
    if "stderr" in params["logging"]["intensity"]:
        intensityErr = open("logs/" + params["logging"]["intensity"]["stderr"], "w")

if "gradient" in params["logging"]:
    if "stdout" in params["logging"]["gradient"]:
        gradientOut = open("logs/" + params["logging"]["gradient"]["stdout"], "w")
    if "stderr" in params["logging"]["gradient"]:
        gradientErr = open("logs/" + params["logging"]["gradient"]["stderr"], "w")

runners = []

intensityPorts = []
gradientPorts = []

## Distribution

if options.distrCoreOverride != "":
    # If remoteModeOverride is enabled, the ditribution mode is set to local
    params["distribution"]["core"] = options.distrCoreOverride

if options.distrServerOverride != "":
    # serverOverride option replaces the serverlist with the command line alternative
    params["distribution"]["type"] = "multi"
    params["distribution"]["servers"] = options.distrServerOverride.split(";")
    print(params["distribution"]["servers"])

awsRemoteServer = None
if params["distribution"]["type"] == "auto":
    # Use automatic distribution based on current system.
    serverConfigKey = "MTSTF_SERVER_CONFIG"
    awsConfig = awsmanager.loadRemoteSettings(serverConfigKey)
    if awsConfig is None:
        print("If distribution.type == 'auto', MTSTF_SERVER_CONFIG environment variable must be set")
        sys.exit(1)

    if awsConfig["autoDetect"]:
        # Dynamic detect
        #server = awsmanager.getFreeServer(awsConfig["keyName"], awsConfig["keyFile"])
        if options.serverSelect >= 0:
            server = awsmanager.getAllServers(awsConfig["keyName"])[options.serverSelect]
        else:
            server = awsmanager.getFreeServer(awsConfig["keyName"], awsConfig["keyFile"])

        if server is None:
            print("Error: distribution.type is 'auto', but there are no available servers")
            print("Please free up servers with awsfree.py before running this")
            sys.exit(1)

        lightlines = open(directory + "/" + params["lights"]["file"], "r").readlines()
        lights = np.array([np.array([float(l.split(" ")[1]), float(l.split(" ")[2]), float(l.split(" ")[3])]) for l in lightlines])
        numNodes = lights.shape[0]
        
        if awsConfig["gitUpdate"] or options.remoteGitUpdate:
            # Quick run a git update and recompile
            print("Git update enabled: Updating remote repository and recompiling code")
            print(awsmanager.remoteCommand("cd ~/mitsuba-diff && git pull origin nmap && scons -j 8 && cd tf_ops && make", server, awsConfig["keyFile"]))
            # TODO: Check for success strings.

        print("Starting " + format(numNodes) + " nodes on server " + server)
        command = "".join([k + " " for k in sys.argv])
        nodelist = awsmanager.runServer(server, awsConfig["keyFile"], command=command, nodes=numNodes)
        print("Waiting for nodes to start...")
        time.sleep(3)
        awsRemoteServer = server

        params["distribution"]["servers"] = nodelist
        params["distribution"]["type"] = "multi" # Auto default to 'multi'
    else:
        # Static
        params["distribution"]["servers"] = MTSCFG_NODES
        params["distribution"]["type"] = "multi" # Auto default to 'multi'

if params["distribution"]["core"] == "local":
    # The main script is run locally. This is the default. Do nothing
    print("distribution.core is 'local': TensorFlow script run locally")
elif params["distribution"]["core"] == "remote":
    # The main script is run remotely.
    if awsRemoteServer is None:
        print("Error: distribution.core is 'remote'. Remote distribution requires distribution.type='auto' and MTSCFG_AWS_AUTODETECT set to true")
        sys.exit(1)

    print("distribution.core is 'remote': TensorFlow script launching on " + awsRemoteServer)

    # Transfer the local test folder to the remote node.
    remoteDir = "/home/ubuntu/uploads/"

    print("rm -rf " + remoteDir + os.path.basename(directory))
    awsmanager.remoteCommand("rm -rf " + remoteDir + os.path.basename(directory), awsRemoteServer, awsConfig["keyFile"])
    awsmanager.remoteCommand("mkdir " + remoteDir + os.path.basename(directory), awsRemoteServer, awsConfig["keyFile"])

    print("scp -i '" + awsConfig["keyFile"] + "' -r " + directory + " ubuntu@" + server + ":" + remoteDir)
    os.system("scp -i '" + awsConfig["keyFile"] + "' -r " + directory + " ubuntu@" + server + ":" + remoteDir)

    serverString = ""
    for _server in params["distribution"]["servers"]:
            serverString += _server + ";"
    serverString = serverString[:-1]

    print("Running remote optimizer: python ~/mitsuba-diff/optimizers/optimize.py " + remoteDir + os.path.basename(directory) + " --distr-server-override " + serverString + " --distr-core-override local")
    awsmanager.remoteCommand("mkdir /home/ubuntu/outputs", awsRemoteServer, awsConfig["keyFile"])
    outputDir = "/home/ubuntu/outputs/" + os.path.basename(directory) + "-" + format(options.index)

    awsmanager.remoteCommand("mkdir " + outputDir, awsRemoteServer, awsConfig["keyFile"])
    if outputDir != "" and options.eraseRemote:
        awsmanager.remoteCommand("rm -r " + outputDir + "/*", awsRemoteServer, awsConfig["keyFile"])

    awsmanager.remoteCommand("nohup python /home/ubuntu/mitsuba-diff/optimizers/optimize.py " + remoteDir + os.path.basename(directory) + "/config.json" + " --distr-server-override \"" + serverString + "\" --distr-core-override local > /home/ubuntu/mtsout.log 2> /home/ubuntu/mtserr.log", awsRemoteServer, awsConfig["keyFile"], directory=outputDir)
    print("Launched remote command")
    sys.exit(0)

if params["distribution"]["type"] == "single":
    # Setup mitsuba processes
    intensityCommand = os.path.dirname(scriptsDirectory) + "/build/release/mitsuba/mtstensorflow " + directory + "/" + params["scenes"]["intensity"] + " -o " + params["output-link"]["intensity"] + " -l 7554 " + runnerParameters

    gradientCommand = os.path.dirname(scriptsDirectory) + "/build/release/mitsuba/mtstensorflow " + directory + "/" + params["scenes"]["gradient"] + " -o " + params["output-link"]["gradient"] + " -l 7555 " + runnerParameters

    if "local-cpus" in params["distribution"]:
        intensityCommand += " -p " + format(params["distribution"]["local-cpus"])
        gradientCommand += " -p " + format(params["distribution"]["local-cpus"])

    if params["distribution"]["enabled"]:
        intensityCommand += " -c "
        gradientCommand += " -c "
        for server in params["distribution"]["servers"]:
            intensityCommand += server + ";"
            gradientCommand += server + ";"

        intensityCommand = intensityCommand[:-1]
        gradientCommand = gradientCommand[:-1]

    print("Starting intensity server")
    print(intensityCommand)
    intensityRunner = subprocess.Popen(intensityCommand.split(" "), stdout=intensityOut, stderr=intensityErr)

    print("Starting gradient server")
    print(gradientCommand)
    gradientRunner = subprocess.Popen(gradientCommand.split(" "), stdout=gradientOut, stderr=gradientErr)

    runners.append(intensityRunner)
    runners.append(gradientRunner)

elif params["distribution"]["type"] == "multi":

    #params["distribution"]["intensity-servers"] = ["".join([ server + ";" for server in params["distribution"]["servers"] ])[:-1]]
    params["distribution"]["gradient-servers"] = params["distribution"]["servers"]
    params["distribution"]["intensity-servers"] = params["distribution"]["servers"]

    """if "intensity-count" in params["distribution"]:
        count = params["distribution"]["intensity-count"]
        params["distribution"]["intensity-servers"] = params["distribution"]["servers"][:count]
        params["distribution"]["servers"] = params["distribution"]["servers"][count:]

    if "gradient-count" in params["distribution"]:
        count = params["distribution"]["gradient-count"]
        params["distribution"]["gradient-servers"] = params["distribution"]["servers"][:count]
        params["distribution"]["servers"] = params["distribution"]["servers"][count:]"""

    iservers = params["distribution"]["intensity-servers"]

    if "base-port" not in params:
        params["base-port"] = 7554

    iport = params["base-port"]
    for idx, iserver in enumerate(iservers):
        linkfifo = "/tmp/mtsout-" + format(iport) + ".hds"
        lockfile = "/tmp/mtsintlock-" + format(iport) + ".lock"

        intensityCommand =  os.path.dirname(scriptsDirectory) + "/build/release/mitsuba/mtstensorflow " +\
                            directory + "/" + params["scenes"]["intensity"] +\
                            " -o " + linkfifo +\
                            " -l " + format(iport) +\
                            " -c " + iserver +\
                            " -b 16 " +\
                             runnerParameters
        if "local-cpus" in params["distribution"]:
            intensityCommand += " -p " + format(params["distribution"]["local-cpus"])

        print("mkfifo " + linkfifo)
        os.system("rm " + linkfifo)
        os.system("mkfifo " + linkfifo)

        lfile = open(lockfile, "w")
        lfile.write(" ")
        lfile.close()

        print("Starting intensity server " + format(iport) + ": " + format(iserver))
        print(intensityCommand)
        intensityRunner = subprocess.Popen(intensityCommand.split(" "), stdout=open("logs/out-" + format(iport) + ".log", "w"), stderr=open("logs/err-" + format(iport) + ".log", "w"))

        runners.append(intensityRunner)
        intensityPorts.append(iport)
        iport -= 1

    gservers = params["distribution"]["gradient-servers"]

    gport = params["base-port"] + 1
    for idx, gserver in enumerate(gservers):
        linkfifo = "/tmp/mtsgradout-" + format(gport) + ".shds"
        lockfile = "/tmp/mtsgradlock-" + format(gport) + ".lock"
        gradientCommand =  os.path.dirname(scriptsDirectory) + "/build/release/mitsuba/mtstensorflow " +\
                            directory + "/" + params["scenes"]["gradient"] +\
                            " -o " + linkfifo +\
                            " -l " + format(gport) +\
                            " -c " + gserver +\
                            " -b 8 " +\
                            runnerParameters
        if "local-cpus" in params["distribution"]:
            gradientCommand += " -p " + format(params["distribution"]["local-cpus"])

        print("mkfifo " + linkfifo)
        print("rm " + linkfifo)
        os.system("rm " + linkfifo)
        os.system("mkfifo " + linkfifo)

        lfile = open(lockfile, "w")
        lfile.write(" ")
        lfile.close()

        print("Starting gradient server " + format(gport) + ": " + format(gserver))
        print(gradientCommand)
        gradientRunner = subprocess.Popen(gradientCommand.split(" "), stdout=open("logs/out-" + format(gport) + ".log", "w"), stderr=open("logs/err-" + format(gport) + ".log", "w"))

        runners.append(gradientRunner)

        gradientPorts.append(gport)
        gport += 1

else:
    print("Invalid distribution mode: ", params["distribution"]["type"])

# Give them time to set up
time.sleep(2)

# Setup files and images specified in the parameters
parameters.prepareParameters(params, directory)

# Start iterations
remesherIterations = 0
if params["remesher"]["enabled"]:
    remesherIterations = params["remesher"]["iterations"]
else:
    remesherIterations = 1

integratorFn = None
if params["remesher"]["integrator"] == "poisson":
    integratorFn = poisson_integrator.integrate
elif params["remesher"]["integrator"] == "frankot":
    integratorFn = frankot_integrator.integrate

remesherKeepNormals = ("keep-normals" in params["remesher"]) and (params["remesher"]["keep-normals"])

print("Keeping normals: " + format(remesherKeepNormals))

if "multiresolution" in params and params["multiresolution"]["enabled"]:
    multiresParams = params["multiresolution"]
else:
    multiresParams = None


mkdir("meshes")
mkdir("renders")

mkdir("meshes/remeshed")
mkdir("meshes/normals")
mkdir("meshes/normaldeltas")
mkdir("meshes/gradients")
mkdir("meshes/totalgradients")

mkdir("renders/normals")
mkdir("renders/normaldeltas")
mkdir("renders/gradients")
mkdir("renders/totalgradients")

mkdir("images")
mkdir("images/normalized-absolute-errors")
mkdir("images/normalized-difference-errors")
mkdir("images/difference-errors")
mkdir("images/unweighted-difference-errors")
mkdir("images/current")
mkdir("images/samplers")
mkdir("images/normalized-absolute-errors/png")
mkdir("images/normalized-absolute-errors/npy")
mkdir("images/normalized-difference-errors/npy")
mkdir("images/difference-errors/npy")
mkdir("images/unweighted-difference-errors/npy")
mkdir("images/current/npy")
mkdir("images/samplers/npy")
mkdir("images/samplers/png")

mkdir("errors")

if isRestart:
    remesherStart = restart
else:
    remesherStart = 0

if "mask" in params:
    print("Found mask data.")

    if "orthoHeight" not in params["mask"]:
        params["mask"]["orthoWidth"] = 1

    if "orthoWidth" not in params["mask"]:
        params["mask"]["orthoHeight"] = 1

    if params["mask"]["type"] == "file":
        params["mask"]["data"] = np.squeeze(np.load(directory + "/" + params["mask"]["file"]))
        #params["mask"]["data"] = scipy.ndimage.morphology.binary_erosion(params["mask"]["data"], structure=scipy.ndimage.morphology.generate_binary_structure(2,5)).astype(params["mask"]["data"].dtype)
    elif params["mask"]["type"] == "target":
        print("Rendering MASK from target")
        params["mask"]["data"] = renderMask(directory + "/" + params["target"]["mesh"], W=params["target"]["width"], H=params["target"]["height"], oW=params["mask"]["orthoWidth"], oH=params["mask"]["orthoHeight"])
    elif params["mask"]["type"] == "original":
        print("Rendering MASK from original")
        params["mask"]["data"] = renderMask(directory + "/" + params["initial-reconstruction"]["mesh"], W=params["target"]["width"], H=params["target"]["height"], oW=params["mask"]["orthoWidth"], oH=params["mask"]["orthoHeight"])
        params["mask"]["data"] = params["mask"]["data"][:, ::-1]
else:
    W,H,_ = params["target"]["data"].shape
    params["mask"] = {}
    params["mask"]["data"] = np.ones((W,H),dtype=np.float)

#print "Mask data:", params["mask"]["data"]

bsdf = None
if remesherStart != 0:
    print("Recording restart number ", remesherStart-1)
    os.system("echo \"" + format(remesherStart-1) + "\" >> " + directory + "/restarts.txt")
    print("Restarts so far: ")
    print(open(directory + "/restarts.txt", "r").read())

    print("Copying forward mesh from super-iteration ", remesherStart-1)
    copyfile("meshes/remeshed/" + format(remesherStart-1).zfill(2) + ".ply", "/tmp/mts_srcmesh.ply")
    params["initialization"]["data"] = load_normals.load_normals("/tmp/mts_srcmesh.ply")
    params["initialization"]["vertex-data"] = load_normals.load_vertices("/tmp/mts_srcmesh.ply")
    if os.path.exists("errors/bsdf-errors-" + format(remesherStart-1).zfill(2) + ".json"):
        bsdf = json.load(open("errors/bsdf-errors-" + format(remesherStart-1).zfill(2) + ".json", "r"))["bvals"][-1]

params["target"]["rawdata"] = np.array(params["target"]["data"])

if "mask" in params:
    params["mask"]["rawdata"] = np.array(params["mask"]["data"])

oW = None
oH = None

if remesherIterations > 1:
    for iterindex in range(remesherStart, remesherIterations):
        timestamp = float(time.time())
        print("Running super iteration " + format(iterindex))

        mfactor = multires(multiresParams, iterindex)
        print("Multires-factor " + format(mfactor))
        W,H,_ = params["target"]["rawdata"].shape
        print("Dims: ", W,H)

        nW = W//mfactor
        nH = H//mfactor
        print ("MASK SIZE: ", params["mask"]["data"].shape)
        if "mask" in params:
            params["mask"]["data"] = downsampleImage(params["mask"]["rawdata"], mfactor)
            params["mask"]["data"] = params["mask"]["data"] > 0.95 # Threshold the mask data.

            print("[TIMER] downsampling_mask ", float(time.time()) - timestamp)
            timestamp = float(time.time())
        else: 
            print("Skipping downsample.. ")

        print ("MASK SIZE: ", params["mask"]["data"].shape)
        if oW != nW or oH != nH:
            # Resolution mismatch.
            print("Resolution mismatch.. ", oW, oH, " != ", nW, nH)
            if (mfactor != 1):
                rescaleMesh("/tmp/mts_srcmesh.ply", nW, nH, mask=params["mask"]["data"])
            else:
                print("Skipping rescale..")


        #plt.imshow(params["mask"]["data"] * 1.0)
        #plt.show()

        #plt.imshow(params["mask"]["data"] * 1.0)
        #plt.show()

        if "mask" in params:
            pass
            #mask_remesher.remesh("/tmp/mts_srcmesh.ply", "/tmp/mts_srcmesh.ply", nW, nH, mask=params["mask"]["data"], edge_protect=2, rescale=True)
        else:
            print("Skipping masked remesh")

        params["initialization"]["data"] = load_normals.load_normals("/tmp/mts_srcmesh.ply")
        params["initialization"]["vertex-data"] = load_normals.load_vertices("/tmp/mts_srcmesh.ply")

        print("[TIMER] load_normals ", float(time.time()) - timestamp)
        timestamp = float(time.time())

        targets = np.array(params["target"]["rawdata"])
        numImages = targets.shape[2]
        new_targets = np.zeros([nW,nH,numImages])
        for i in range(numImages):
            if "mask" in params:
                new_targets[:,:,i] = downsampleImage(targets[:,:,i], mfactor) * params["mask"]["data"]
            else:
                new_targets[:,:,i] = downsampleImage(targets[:,:,i], mfactor)

        params["target"]["data"] = new_targets

        # Default weights.
        params["target"]["weights"] = np.ones_like(params["target"]["data"])
        params["target"]["bsdf-weights"] = np.ones_like(params["target"]["data"])


        if "reweighting" in params["target"] and params["target"]["reweighting"]:
            print("Reweighting targets based on intensity")
            params["target"]["weights"] = params["target"]["weights"] * thresholding.intensityWeights(params["target"]["data"])
            params["target"]["bsdf-weights"] = params["target"]["bsdf-weights"] * thresholding.intensityWeights(params["target"]["data"])

        # Autogenerate target weights WxHxN, if thresholding is enabled.
        if "thresholding" in params["target"] and type(params["target"]["thresholding"]) is int:
            params["target"]["thresholding"] = {"type":"ordered","dark":0, "bright":0}

        if "thresholding" in params["target"] and "bright" not in params["target"]["thresholding"]:
            params["target"]["thresholding"]["bright"] = 0
        if "thresholding" in params["target"] and "dark" not in params["target"]["thresholding"]:
            params["target"]["thresholding"]["dark"] = 0

        if "thresholding" in params["target"]:
            print("Clipping dark regions of the target")
            params["target"]["weights"] = params["target"]["weights"] * thresholding.clipDarkRegions(params["target"]["data"], num=params["target"]["thresholding"]["dark"]) * thresholding.clipBrightRegions(params["target"]["data"], num=params["target"]["thresholding"]["bright"])
            params["target"]["bsdf-weights"] = params["target"]["bsdf-weights"] * thresholding.clipDarkRegions(params["target"]["data"], num=params["target"]["thresholding"]["dark"])

        if "mask" in params:
            if "weight-erode" in params["mask"]:
                werode = params["mask"]["weight-erode"]
                params["target"]["weights"] *= cv2.erode(params["mask"]["data"].astype(np.uint8), np.ones([werode, werode], np.uint8))[:,:,np.newaxis]
                params["target"]["bsdf-weights"] *= cv2.erode(params["mask"]["data"].astype(np.uint8), np.ones([werode, werode], np.uint8))[:,:,np.newaxis]
            else:
                params["target"]["weights"] *= params["mask"]["data"][:,:,np.newaxis]
                params["target"]["bsdf-weights"] *= params["mask"]["data"][:,:,np.newaxis]

        if "regularization" in params["estimator"]:
            radius = params["estimator"]["regularization"]["radius"]
        else:
            radius = 2.0

        np.save('weights.npy', params["target"]["weights"])
        np.save('bsdf-weights.npy', params["target"]["bsdf-weights"])

        print("[TIMER] downsampling_target ", float(time.time()) - timestamp)
        timestamp = float(time.time())

        originalfile = directory + "/" + params["target"]["mesh"]
        copyfile(originalfile, "/tmp/targetmesh_copy.ply")
        #copyfile(originalfile, "/tmp/targetmesh_copy.ply")

        #if oW != nW or oH != nH:
        # Resolution mismatch.
        #plt.imshow(params["mask"]["data"] * 1.0)
        #plt.show()

        # TEMP: Disable mesh rescaling.
        #rescaleMesh("/tmp/targetmesh_copy.ply", nW, nH, mask=params["mask"]["data"])

        if "mask" in params:
            mask_remesher.remesh("/tmp/targetmesh_copy.ply", "/tmp/targetmesh_copy.ply", nW, nH, mask=params["mask"]["data"], edge_protect=2, rescale=False)

        params["target"]["normals"] = load_normals.load_normals("/tmp/targetmesh_copy.ply")

        print("[TIMER] load_reference_normals ", float(time.time()) - timestamp)
        timestamp = float(time.time())

        if oW != nW or oH != nH:
            #indexMap, validMask = MeshAdjacencyBuilder.buildIndexMap(
            #    load_normals.load_vertices("/tmp/targetmesh_copy.ply"),
            #    radius=radius,
            #    width=nW,
            #    height=nH)
            indexMap = None
            validMask = None
        else:
            indexMap = None
            validMask = None

        # Optimize normals.
        bsdf, ifactors = normals.optimizeNormals(params, superindex=iterindex, bsdfoverride=bsdf, W=nW, H=nH, iports=intensityPorts, gports=gradientPorts, indexMap=indexMap, indexValidity=validMask)

        if ("recalibrate" in params["lights"]) and params["lights"]["recalibrate"]:
            print("Light recalibration \nOld:", params["lights"]["intensity-data"])
            params["lights"]["intensity-data"] = params["lights"]["intensity-data"] / ifactors
            print("Factors: ", ifactors)
            print("New:", params["lights"]["intensity-data"])

        print("[TIMER] optimizing_normals", float(time.time()) - timestamp)
        timestamp = float(time.time())

        #remesh.remesh("/tmp/mts_mesh_copy_1.ply", "/tmp/mts_remeshed_mesh.ply", keep_normals=remesherKeepNormals, integrator=integratorFn, invert_normals=True, edge_protect=2)
        if "mask" in params:
            mask_remesher.remesh("/tmp/mts_mesh_intensity_slot_0.ply", "/tmp/mts_remeshed_mesh.ply", nW, nH, mask=params["mask"]["data"], edge_protect=2, rescale=True)
        else:
            remesh.remesh("/tmp/mts_mesh_intensity_slot_0.ply", "/tmp/mts_remeshed_mesh.ply", keep_normals=remesherKeepNormals, integrator=integratorFn, invert_normals=True, edge_protect=2)

        copyfile("/tmp/mts_remeshed_mesh.ply", "meshes/remeshed/" + format(iterindex).zfill(2) + ".ply")
        copyfile("/tmp/mts_remeshed_mesh.ply", "/tmp/mts_srcmesh.ply")
        params["initialization"]["data"] = load_normals.load_normals("/tmp/mts_srcmesh.ply")
        params["initialization"]["vertex-data"] = load_normals.load_vertices("/tmp/mts_srcmesh.ply")
        oH = nH
        oW = nW

else:
    print("Running normal optimization only")
    normals.optimizeNormals(params, superindex=0)

print("Killing tensorflow rendering servers")
for runner in runners:
    runner.kill()