#!/usr/bin/env python

import re
import sys
filename = None
saveToPdf = False
while len(sys.argv)>1:
    if sys.argv[1]=="-h":
        filename = None; break
    if sys.argv[1]=="-s":
        saveToPdf = True
        sys.argv = sys.argv[1:]
        continue
    filename = sys.argv[1]
    break
if filename is None:
    print("Usage: %s [ -h ] [ -s ] timingfile\n" % sys.argv[0])
    sys.exit(1)

rank_nums = []
imp_times = []
ref_times = []
with open(filename,"r") as runfile:
    run = 0
    for l in runfile:
        l = l.strip()
        split = l.split()
        if re.match("Running",l):
            rank_nums.append( int(split[4]) )
        if re.match("reference",l):
            run = 0
        if re.match("imp",l):
            run = 1
        if re.match("unbalanced",l):
            run = 2
        if re.match("Total duration",l):
            t = float( split[2] )
            if run==1:
                imp_times.append(t)
            elif run==2:
                ref_times.append(t)

import matplotlib
if saveToPdf:
    matplotlib.use('pdf')
import matplotlib.pyplot as plt

lines = []; schemes = []
l, = plt.semilogx(rank_nums,imp_times)
lines.append(l); schemes.append("IMP balancing")
l, = plt.semilogx(rank_nums,ref_times)
lines.append(l); schemes.append("Static distribution")

plt.xlabel("MPI ranks")
plt.ylabel("seconds")
plt.ylim( [0,1.1*ref_times[0]] )

#plt.legend("Balancing vs unbalanced",schemes,loc='upper left')
plt.suptitle('Strong scaling timings')
if saveToPdf:
    plt.savefig("balance.pdf")
else:
    plt.show()
