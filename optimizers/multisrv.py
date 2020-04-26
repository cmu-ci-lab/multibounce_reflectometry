import signal
import sys
import subprocess
import json
import multiprocessing
from socket import gethostname
import optparse
import os

MITSUBA_DIFF_ENV = "/home/ubuntu/mitsuba-diff"

parser = optparse.OptionParser()
parser.add_option("-n", "--numnodes", dest="numnodes", type="int")
parser.add_option("-a", "--aws", action="store_true", dest="aws")
parser.add_option("-l", "--local", action="store_true", dest="local")
parser.add_option("-d", "--dry-run", action="store_true", dest="dryRun")
parser.add_option("-b", "--base-port", dest="port", default=7554, type="int")
(options, args) = parser.parse_args()

totalcores = multiprocessing.cpu_count()
print("Detected cores: ", totalcores)

#numnodes = int(sys.argv[1])
numnodes = options.numnodes

if options.aws and options.local:
    print("Error: -a and -l flags cannot be used together")
    sys.exit(1)

if options.aws:
    hostnamep = subprocess.Popen("curl http://169.254.169.254/latest/meta-data/public-hostname".split(" "), stdout=subprocess.PIPE)
    hostname = hostnamep.stdout.read()
elif options.local:
    hostname = "127.0.0.1"
else:
    hostname = gethostname()

print("Detected hostname: ", hostname)

#numnodes = totalcores
coresperunit = (totalcores // numnodes)

runners = []

def interruptHandler(sig, frame):
    print("Killing all servers")
    for runner in runners:
        runner.kill()
    sys.exit(0)

for i in range(numnodes):
    command = MITSUBA_DIFF_ENV + "/build/release/mitsuba/mtssrv -p " + format(coresperunit) + " -i " + hostname + " -l " + format(7554 + i)
    print(command)
    if not options.dryRun:
        os.system("rm mtssrv-" + format(options.port + i) + "-out.log")
        os.system("rm mtssrv-" + format(options.port + i) + "-err.log")
        runner = subprocess.Popen(
                command.split(" "),
                stdout=open("mtssrv-" + format(options.port + i) + "-out.log", "w"),
                stderr=open("mtssrv-" + format(options.port + i) + "-err.log", "w"))
        runners.append(runner)

print ("All units: ")
print (json.dumps([ format(hostname) + ":" + format(options.port + i) for i in range(numnodes)], indent=True))
signal.signal(signal.SIGINT, interruptHandler)
