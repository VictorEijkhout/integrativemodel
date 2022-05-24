#!/usr/bin/env python

import matplotlib.pyplot as plt
import re
import sys

if len(sys.argv)==1:
    print "Usage: %s filename" % sys.argv[0]

def plot_block_time(gr,blocks,times):
    gr.plot(blocks,times)

timefile = sys.argv[1]
nlat = 0
nloc = 0
with open(timefile,"r") as timings:
    for line in timings:
        line = line.strip()
        if re.search("Latency",line):
            nlat += 1
        if re.search("nlocal",line):
            nloc += 1

if nloc%nlat!=0:
    print "Incomplete file?"; sys.exit(1)
nloc = nloc/nlat

fig,axes = plt.subplots(nrows=nlat,ncols=nloc,sharey='row')
igraph = -1; blocks = []
with open(timefile,"r") as timings:
    for line in timings:
        line = line.strip()
        if re.search("Latency",line):
            latency = line.split()[1]
        if re.search("nlocal",line):
            if len(blocks)>0:
                ilat = igraph/nloc ; iloc = igraph%nloc; 
                print "printing %d graph as [%d,%d]" % (igraph,ilat,iloc)
                axes[ilat,iloc].plot(blocks,times)
            blocks = []; times = []; igraph += 1
            print "starting graph",igraph
        if re.search("Blocking",line):
            b = line.split()[2]
            blocks.append(b)
        if re.search("Parallel",line):
            t = line.split()[2]
            t = int(t)/1000
            times.append(t)
ilat = igraph/nloc ; iloc = igraph%nloc; 
print "printing %d graph as [%d,%d]" % (igraph,ilat,iloc)
axes[ilat,iloc].plot(blocks,times)

plt.show()
