# Provides AWS Remote manipulation operations

import optparse
import boto3
import awsmanager
import os
import datetime
import termcolor

parser = optparse.OptionParser()
parser.add_option("-l", "--list", action="store_true",
                  dest="list", default=False)
parser.add_option("-k", "--kill", dest="kill", default=-1, type="int")
parser.add_option("-p", "--pull", dest="pull", default=None, type="string")
parser.add_option("-d", "--purge", dest="purge", default=None, type="string")
parser.add_option("-a", "--purge-all", dest="purgeAll",
                  default=None, type="int")
parser.add_option("-u", "--update", dest="update", default=None, type="int")

(options, args) = parser.parse_args()

print("Loading from MTSTF_SERVER_CONFIG")
awsConfig = awsmanager.loadRemoteSettings("MTSTF_SERVER_CONFIG")
print(awsConfig)

if options.list:
    print("all")
    for idx, server in enumerate(awsmanager.getAllServers(awsConfig["keyName"], quiet=True)):
        print idx, "\n", server, "\n\tServers:\n", "\t\t" + awsmanager.getServerStatus(server, awsConfig["keyFile"], quiet=True)
        print "\n\tDatasets:"
        for idx, dataset in enumerate(awsmanager.getServerDatasets(server, awsConfig["keyFile"], quiet=True)):
            if dataset["datetime"] is not None:
                print "\t\t" +\
                    format(idx) + " " + dataset["name"] +\
                    " " + dataset["datetime"].strftime("%I:%M%p %A %d. %B %Y") +\
                    " si/" + format(dataset["super-iters"]) + \
                    " i/" + format(dataset["last-iters"])
            else:
                print "\t\t" + format(idx) + " " + dataset["name"] + " NO TIME SPECIFIED"

        print "\n\tOptimizers:"
        print "\t\t", (awsmanager.getOptimizerStatus(server, awsConfig["keyFile"], quiet=True))
        print "\n"

elif options.kill != -1:
    servers = awsmanager.getAllServers(awsConfig["keyName"], quiet=True)
    server = servers[options.kill]
    print("Killing ", server)
    awsmanager.killServer(server, awsConfig["keyFile"], quiet=False)
    optimizers = awsmanager.killOptimizer(
        server, awsConfig["keyFile"], quiet=False, force=True)
    print("Killing: ", optimizers)

elif options.pull is not None:
    parts = options.pull.split(".")
    if len(parts) != 2:
        print("Specify pull index as <server-index>.<dataset-index>")
    serverIdx = int(parts[0])
    datasetIdx = int(parts[1])

    datasetDest = awsConfig["dataRepository"]
    datasetDest = datasetDest + "/" + datetime.datetime.now().strftime("%b-%d-%Y")
    if not os.path.exists(datasetDest):
        os.mkdir(datasetDest)

    server = awsmanager.getAllServers(
        awsConfig["keyName"], quiet=True)[serverIdx]
    dataset = awsmanager.getServerDatasets(
        server, awsConfig["keyFile"], quiet=True)[datasetIdx]
    print("scp -i '" + awsConfig["keyFile"] + "' -r -v ubuntu@" + server +
          ":/home/ubuntu/outputs/" + dataset["name"] + " " + datasetDest + "/")
    os.system("scp -i '" + awsConfig["keyFile"] + "' -r -v ubuntu@" + server +
              ":/home/ubuntu/outputs/" + dataset["name"] + " " + datasetDest + "/")

elif options.purge is not None:
    parts = options.purge.split(".")
    if len(parts) != 2:
        print("Specify pull index as <server-index>.<dataset-index>")
    serverIdx = int(parts[0])
    datasetIdx = int(parts[1])

    server = awsmanager.getAllServers(
        awsConfig["keyName"], quiet=True)[serverIdx]
    dataset = awsmanager.getServerDatasets(
        server, awsConfig["keyFile"], quiet=True)[datasetIdx]

    print("rm -r '/home/ubuntu/outputs/" + dataset["name"] + "'")
    awsmanager.remoteCommand("rm -r '/home/ubuntu/outputs/" +
                             dataset["name"] + "'", server, awsConfig["keyFile"], quiet=False)

elif options.purgeAll is not None:
    serverIdx = options.purgeAll

    allServers = awsmanager.getAllServers(awsConfig["keyName"], quiet=True)
    if serverIdx >= len(allServers):
        print("Invalid server index ", serverIdx, " there are only ",
              len(allServers), " servers online.")

    server = allServers[serverIdx]
    print("rm -r /home/ubuntu/outputs/*")
    awsmanager.remoteCommand(
        "rm -r /home/ubuntu/outputs/*", server, awsConfig["keyFile"], quiet=False)
elif options.update is not None:
    serverIdx = options.update
    allServers = awsmanager.getAllServers(awsConfig["keyName"], quiet=True)
    server = allServers[serverIdx]

    print("Pulling new commits...")
    awsmanager.remoteCommand(
        "cd ~/mitsuba-diff && git pull origin nmap > /tmp/r-out.log 2> /tmp/r-err.log", server, awsConfig["keyFile"])
    print "STDOUT:\n", termcolor.colored(awsmanager.remoteCommand("cat /tmp/r-out.log", server, awsConfig["keyFile"]), 'green')
    print "STDERR:\n", termcolor.colored(awsmanager.remoteCommand("cat /tmp/r-err.log", server, awsConfig["keyFile"]), 'red')

    print("Compiling mitsuba-diff...")
    awsmanager.remoteCommand(
        "cd ~/mitsuba-diff && scons . -j 8 > /tmp/r-out.log 2> /tmp/r-err.log", server, awsConfig["keyFile"])
    print "STDOUT:\n", termcolor.colored(awsmanager.remoteCommand("cat /tmp/r-out.log", server, awsConfig["keyFile"]), 'green')
    print "STDERR:\n", termcolor.colored(awsmanager.remoteCommand("cat /tmp/r-err.log", server, awsConfig["keyFile"]), 'red')

    print("Compiling mitsuba-tensorflow operators...")
    awsmanager.remoteCommand(
        "cd ~/mitsuba-diff/tf_ops && make > /tmp/r-out.log 2> /tmp/r-err.log", server, awsConfig["keyFile"])
    print "STDOUT:\n", termcolor.colored(awsmanager.remoteCommand("cat /tmp/r-out.log", server, awsConfig["keyFile"]), 'green')
    print "STDERR:\n", termcolor.colored(awsmanager.remoteCommand("cat /tmp/r-err.log", server, awsConfig["keyFile"]), 'red')

    print("Done.")
else:
    print("Specify an action flag (--kill, --list, --pull or --purge)")
