import signal
import sys
import subprocess
import json
import multiprocessing

totalcores = multiprocessing.cpu_count()
print "Detected cores: ", totalcores

numnodes = int(sys.argv[1])

#hostnamep = subprocess.Popen("curl http://169.254.169.254/latest/meta-data/public-hostname".split(" "), stdout=subprocess.PIPE)
hostname = "127.0.0.1"
print "Detected hostname: ", hostname

#numnodes = totalcores
coresperunit = (totalcores // numnodes)

runners = []

def interruptHandler(sig, frame):
    print("Killing all servers")
    for runner in runners:
        runner.kill()
    sys.exit(0)

for i in range(numnodes):
    command = "/home/ubuntu/mitsuba-diff/build/release/mitsuba/mtssrv -p " + format(coresperunit) + " -i " + hostname + " -l " + format(7554 + i)
    print(command)
    runner = subprocess.Popen(command.split(" "))
    runners.append(runner)

print ("All units: ")
print (json.dumps([ format(hostname) + ":" + format(i + 7554) for i in range(numnodes)], indent=True))
signal.signal(signal.SIGINT, interruptHandler)