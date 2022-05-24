#!/usr/bin/env python

import matplotlib.pyplot as plt
import re
import sys

if len(sys.argv)<4:
    print "Usage: %s latency localsize filename1 filename2 ... " % sys.argv[0]

def plot_block_time(gr,blocks,times):
    gr.plot(blocks,times)

try:
    target_latency = int(sys.argv[1])
    target_nlocal = int(sys.argv[2])
except ValueError:
    print "Problem converting numeric arguments: <<%s>> <<%s>>" % (sys.argv[1],sys.argv[2])
    sys.exit(1)

timefiles = sys.argv[3:]
nfil = len(timefiles)
nloc = 3

print "Graphing lat=%d loc=%d in %d files" % (target_latency,target_nlocal,nfil)
fig,axes = plt.subplots(nrows=1,ncols=nfil,sharex='all',sharey='all')
fig.text(0.2,0.9, "Parallel execution time as function of step blocking\nhorizontal: strong scaling (number of cores)",horizontalalignment="left")

igraph = -1; blocks = []; latency = 0; nlocal = 0
for timefile in timefiles :
    print "================",timefile
    with open(timefile,"r") as timings:
        for line in timings:
            line = line.strip()
            if re.search("cores",line):
                cores = line.split()[1]; cores = int(cores)
            if re.search("Latency",line):
                last_latency = latency
                latency = line.split()[1]; latency = int(latency)
                if latency==target_latency:
                    print "Latency %d: graphing" % latency
                else:
                    print "Latency %d: skipping" % latency
            if re.search("nlocal",line):
                last_local = nlocal
                nlocal = line.split()[2]; nlocal = int(nlocal)
                if len(blocks)>0:
                    ilat = igraph/nloc ; iloc = igraph%nloc; 
                    print "printing graph %d as [%d,%d]" % (igraph,0,igraph)
                    axes[igraph].plot(blocks,times)
                    axes[igraph].text\
                        (0.1,0.1,
                         "cores=%d" % cores,
                         fontsize=10,horizontalalignment="left",
                         transform=axes[igraph].transAxes)
                    # if ilat==nfil-1:
                    #     axes[ilat,iloc].text\
                    #         (0.9,0.6,
                    #          "tasksize=%d" % last_local,
                    #          fontsize=10,horizontalalignment="right",
                    #          transform=axes[ilat].transAxes)
                blocks = []; times = []
                if latency==target_latency and nlocal==target_nlocal:
                    igraph += 1
                print "starting graph",igraph
            if re.search("Blocking",line) and latency==target_latency and nlocal==target_nlocal:
                b = line.split()[2]
                blocks.append(b)
            if re.search("Parallel",line) and latency==target_latency and nlocal==target_nlocal:
                t = line.split()[2]
                t = int(t)/1000
                times.append(t)
if len(blocks)>0:
    last_local = nlocal
    ilat = igraph/nloc ; iloc = igraph%nloc; 
    print "printing %d graph as [%d,%d]" % (igraph,ilat,iloc)
    axes[igraph].plot(blocks,times)
    # axes[igraph].text\
    #     (0.9,0.6,
    #      "tasksize=%d" % last_local,
    #      fontsize=10,horizontalalignment="right",
    #      transform=axes[ilat,iloc].transAxes)

#plt.show()
fig.savefig("strongscale-%d.pdf" % target_latency)
