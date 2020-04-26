import struct
import sys

filename = sys.argv[1]
f = open(filename, "r");
alldata = f.read();
numvals = struct.unpack('iii',alldata[0:12])

num = numvals[0];
print ("Items: ", numvals[0]);
print ("Dimensions: ", numvals[1], numvals[2]);

alldata = alldata[12:];
for i in range(len(alldata)/(16)):
    vals = struct.unpack('ifff',alldata[i*16:i*16+16]);
    print "Differential @ ", vals[0], ": ", vals[1], ",", vals[2], ",", vals[3];
    #break;
