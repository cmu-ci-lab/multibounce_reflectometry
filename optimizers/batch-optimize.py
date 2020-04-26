# Batch optimizer.
import sys
import os
import json

directory = os.path.abspath(sys.argv[1])
# Load instance list.
instancelist = json.load(open(directory + "/instances/instances.json", "r"))

# Get script directory
scriptsDirectory = os.path.dirname(__file__)

for subdir in instancelist:
    os.mkdir(subdir)

    print("Processing instance " + subdir + "\nParameters: " + format(instancelist[subdir]))
    os.system("cd " + subdir + " && python " + scriptsDirectory + "/optimize.py " + directory + "/instances/" + subdir + "/config.json")