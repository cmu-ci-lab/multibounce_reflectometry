import os
import sys
import json
import itertools

from shutil import copytree

configfile = sys.argv[1]
fullconfig = json.load(open(configfile, "r"))
configdir = os.path.dirname(configfile)

if not (fullconfig["version"]["major"] == 1 and fullconfig["version"]["minor"] == 1):
    print("Version mismatch. Must be 1.1")
    assert(0)

if not fullconfig["type"] == "instanced":
    print("Not an instanced config. config.type must be 'instanced'")
else:
    del fullconfig["type"]

def dget(d, field):
    if len(field.split(".")) > 1:
        return dget(d[field.split(".")[0]], "".join(field.split(".")[1:]))
    return d[field]

def dset(d, field, value):
    if len(field.split(".")) > 1:
        dset(d[field.split(".")[0]], "".join(field.split(".")[1:]), value)
    else:
        d[field] = value

# Load instanced attributes

instancedFields = ["remesher.iterations",
                   "target.samples",
                   "target.depth",
                   "estimator.samples",
                   "estimator.iterations",
                   "estimator.optimizer",
                   "estimator.depth"]

dataDirectories = ["scenes", "meshes", "lights"]
instancedFieldValues = []

instances = []
for ifield in instancedFields:
    val = dget(fullconfig, ifield)
    if type(val) is not list:
        print("Instanced parameter " + ifield + " is not a list of possible values.")
        assert(0)

    instancedFieldValues.append(val)

instances = itertools.product(*instancedFieldValues)

irecords = {}
os.mkdir(configdir + "/instances")
for index, instance in enumerate(instances):
    configInstance = dict(fullconfig)
    os.mkdir(configdir + "/instances/" + format(index).zfill(4))

    for fieldname, field in zip(instancedFields, instance):
        dset(configInstance, fieldname, field)

    for directory in dataDirectories:
        copytree(configdir + "/" + directory, configdir + "/instances/" + format(index).zfill(4) + "/" + directory)

    irecords[format(index).zfill(4)] = zip(instancedFields, instance)

    json.dump(configInstance, open(configdir + "/instances/" + format(index).zfill(4) + "/config.json", "w"), indent=4)

json.dump(irecords, open(configdir + "/instances/instances.json", "w"), indent=4)