#!/usr/bin/env python

import re
import sys

import matplotlib.pyplot as plt
import numpy as np

with open(sys.argv[1],"r") as tracefile:
    for line in tracefile:
        if re.match("Partition",line):
            step = re.match("Partition at step ([0-9]*)",line).groups()[0]
            step = int(step)
        if re.match("P=",line):
            pline = re.match("P=([0-9]*) w=([0-9\.]*) s=([0-9]*)-([0-9]*)",line)
            if not pline or len(pline.groups())<4:
                print "Could not parse line:",line
                sys.exit(1)
            proc,work,smin,smax = pline.groups()
            proc = int(proc); work = float(work); smin = int(smin); smax = int(smax)

steps = step+1; procs = proc+1; gsize = smax
workgraph = np.zeros((steps, gsize))

with open(sys.argv[1],"r") as tracefile:
    for line in tracefile:
        if re.match("Partition",line):
            step = re.match("Partition at step ([0-9]*)",line).groups()[0]
            step = int(step)
        if re.match("P=",line):
            pline = re.match("P=([0-9]*) w=([0-9\.]*) s=([0-9]*)-([0-9]*)",line)
            if not pline or len(pline.groups())<4:
                print "Could not parse line:",line
                sys.exit(1)
            proc,work,smin,smax = pline.groups()
            proc = int(proc); work = float(work); smin = int(smin); smax = int(smax)
            for s in range(smin,smax-1):
                workgraph[(step,s)] = work
            workgraph[(step,smax-1)] = 0

plt.imshow(workgraph,
           extent=[0,gsize,0,steps], aspect='auto',
           cmap='hot', interpolation='nearest')
plt.colorbar()
plt.savefig("diffuse.pdf")
#plt.show()
