# AWS-specific server operations manager.
# Includes methods to find free servers, launch/kill remote mtssrv processes as well as
# remote mounting options

import boto3
import os
import string
import json
import re
import datetime

def loadRemoteSettings(configKey):
    # Look for settings in the usual areas.
    if configKey not in os.environ:
        return None

    # Run the config file to set global variables
    config = json.load(open(os.environ[configKey], "r"))
    return {
        "keyName": config["MTSCFG_AWS_KEYNAME"],
        "keyFile": config["MTSCFG_AWS_SSHKEY"],
        "autoDetect": config["MTSCFG_AWS_AUTODETECT"],
        "nodes": config["MTSCFG_NODES"],
        "servers": config["MTSCFG_SERVERS"],
        "gitUpdate": config["MTSCFG_AWS_GIT_UPDATE"],
        "dataRepository": config["MTSCFG_DATA_REPOSITORY"]
    }

def _valid(instance):
    return len(instance["NetworkInterfaces"]) > 0

def _dns(instance):
    # Assume that the first network interface represents the primary connection to the public network.
    return instance["NetworkInterfaces"][0]["Association"]["PublicDnsName"]

def remoteCommand(cmd, server, keyFile, user='ubuntu', directory=None, quiet=False):
    if directory is None:
        directory = '/home/' + user

    if not quiet:
        print("Running remote command: " + "ssh -i '" + keyFile + "' " + user + "@" + server + " -t 'cd " + directory + " ; " + cmd + "'")

    if not quiet:
        return os.popen("ssh -i '" + keyFile + "' " + user + "@" + server + " -t 'cd " + directory + " ; " + cmd + "'").read()
    else:
        return os.popen("ssh -i '" + keyFile + "' " + user + "@" + server + " -t 'cd " + directory + " ; " + cmd + "' 2>/dev/null").read()

def getServerStatus(server, keyFile, quiet=False):
    if remoteCommand("test -e /home/ubuntu/.mtslock && echo 1 || echo 0", server, keyFile, quiet=quiet) in ["0\n", "0\r\n"]:
        return "free"
    else:
        return remoteCommand("cat /home/ubuntu/.mtslock", server, keyFile, quiet=quiet)

def getServerDatasets(server, keyFile, quiet=False):
    datasets = []
    for line in remoteCommand("cd /home/ubuntu/outputs && ls -all", server, keyFile, quiet=quiet).split("\n")[3:-1]:
        setname = line.split(" ")[-1].strip()
        dirname = "/home/ubuntu/outputs/" + setname
        try:
            runInfo = json.loads(remoteCommand("cat /home/ubuntu/outputs/" + setname + "/run.info", server, keyFile, quiet=quiet))
            numSuperIters = len(remoteCommand("cd " + dirname + "/images/current/npy && ls -all", server, keyFile, quiet=quiet).split("\r\n")) - 4
            if numSuperIters != 0:
                numIters = len(remoteCommand("cd " + dirname + "/images/current/npy/" + format(numSuperIters-1).zfill(2) + " && ls -all *-img-00.npy", server, keyFile, quiet=quiet).split("\r\n")) - 1
            else:
                numIters = 0

            datasets.append({"name": setname, "datetime": datetime.datetime.fromtimestamp(runInfo["timestamp"]), "last-iters": numIters, "super-iters": numSuperIters})
        except Exception as e:
            datasets.append({"name": setname, "datetime": None})
    return datasets

def getOptimizerStatus(server, keyFile, quiet=False):
    pids = remoteCommand("pgrep -f -a optimizers/optimize.py", server, keyFile, quiet=quiet)
    #print("Optimizers running: ", pids)

    if pids == "":
        return 0

    processes = []
    pidlist = pids.split("\n")[:-1] #TODO: Confirm if this assumption works.
    for pid in pidlist:
        if "pgrep " in pid:
            continue
        directory = pid.split(" ")[3]
        processes.append((pid.split(" ")[0], directory))

    return processes


def getFreeServer(keyName, keyFile, quiet=False):
    servers = getAllServers(keyName)
    for server in servers:
        cmdlineOutput = remoteCommand("test -e /home/ubuntu/.mtslock && echo 1 || echo 0", server, keyFile, quiet=quiet)
        if cmdlineOutput in ["0\n", "0\r\n"]:
            return server
        else:
            print(server, ":", getServerStatus(server, keyFile), ":", cmdlineOutput)
    return None

def getAllServers(keyName, quiet=False):
    ec2 = boto3.client('ec2')
    return [ _dns(instance) for reservation in ec2.describe_instances()["Reservations"] for instance in reservation["Instances"] if instance["KeyName"] == keyName and _valid(instance) ]

def runServer(server, keyFile, command="command-not-specified", nodes=8, quiet=False):
    remoteCommand("echo '" + command + "' > /home/ubuntu/.mtslock", server, keyFile, quiet=quiet)
    moutput = remoteCommand("python /home/ubuntu/mitsuba-diff/optimizers/multisrv.py -n " + format(nodes) + " -a --dry-run", server, keyFile, quiet=quiet)
    remoteCommand("nohup python /home/ubuntu/mitsuba-diff/optimizers/multisrv.py -n " + format(nodes) + " -a 2>nohuperr.log", server, keyFile, quiet=quiet)

    if not quiet:
        print(re.findall(r'\[(.*?)\]', moutput, flags=re.DOTALL))

    matches = [ match for match in re.findall(r'\[(.*?)\]', moutput, flags=re.DOTALL) if "mtssrv" not in match ]

    if len(matches) > 1:
        if not quiet:
            print("ERROR while parsing server output: Too many matches")
        print(matches)

    match = matches[0]

    nodelist = json.loads("[" + match + "]")

    if not quiet:
        print(nodelist)

    return nodelist

def killServer(server, keyFile, quiet=False):
    remoteCommand("rm /home/ubuntu/.mtslock", server, keyFile, quiet=quiet)
    remoteCommand("pkill -9 mtssrv", server, keyFile, quiet=quiet)

# Find and kill optimizers.
def killOptimizer(server, keyFile, force=False, quiet=False):
    optimizers = getOptimizerStatus(server, keyFile, quiet=quiet)

    for optimizer in optimizers:
        if force:
            remoteCommand("kill -9 " + format(optimizer[0]), server, keyFile, quiet=quiet)
        else:
            remoteCommand("kill " + format(optimizer[0]), server, keyFile, quiet=quiet)

    return optimizers

def remoteMount(server, keyFile, user="ubuntu", path="/home/ubuntu", localparent=None, remotedir=None, localdir=None):
    #if remotedir is None:
    #    print("Remote directory not specified")
    #    sys.exit(1)

    if localparent is None:
        localparent = "/home/" + os.environ["USERNAME"] + "/rmounts"

    if localdir is not None:
        localpath = localparent + "/" + localdir
    else:
        cleandir = "".join([ (char if char.isalnum() else "-") for char in user + "@" + server + ":" + path ])
        localpath = localparent + "/" + cleandir

    print("Mounting remote: sshfs " + user + "@" + server + ":" + path + " " + localpath + " -o IdentityFile=" + keyFile)
    os.system("fusermount -u " + localpath)
    os.system("mkdir " + localpath)
    os.system("sshfs " + user + "@" + server + ":" + path + " " + localpath + " -o IdentityFile=" + keyFile)

    return localpath