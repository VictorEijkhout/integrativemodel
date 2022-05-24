#!/bin/env python

import numpy
import re
import sys

if len(sys.argv)<2:
    print("Usage: [ -a ] %s resultsfile" % sys.argv[0])
    sys.exit(1)
sys.argv = sys.argv[1:]
plot_analysis = False
while len(sys.argv)>1:
    if re.match(sys.argv[0],"-a"):
        plot_analysis = True
    else:
        print("Usage: [ -a ] %s resultsfile" % sys.argv[0])
        sys.exit(1)
    sys.argv = sys.argv[1:]
fname = sys.argv[0]

pvalue = 0
processors = []
svalue = 0
sizes = []
with open(fname,"r") as results:
    for l in results:
        l.strip()
        if re.match("Processors:",l):
            p = l.split()[1]
            p = int(p)
            if p>pvalue:
                processors.append(p)
                pvalue = p
        if re.match("Size:",l):
            s = l.split()[1]
            s = int(s)
            if s>svalue:
                sizes.append(s)
                svalue = s
print("Processors:",str(processors))
print("Sizes:",str(sizes))
pindex = {}
for i in range(len(processors)):
    pindex[str(processors[i])] = i
sindex = {}
for i in range(len(sizes)):
    sindex[str(sizes[i])] = i

analysis = numpy.zeros( [len(processors),len(sizes)] )
execution = numpy.zeros( [len(processors),len(sizes)] )
with open(fname,"r") as results:
    for l in results:
        l.strip()
        if re.match("Processors:",l):
            p = l.split()[1]
        if re.match("Size:",l):
            s = l.split()[1]
        if re.match("Analysis time:",l):
            a = l.split()[2]
            a = float(a)
            analysis[pindex[p],sindex[s]] = a
        if re.match("Execution time:",l):
            e = l.split()[2]
            e = float(e)
            execution[pindex[p],sindex[s]] = e
print("Analysis time:"); print(analysis)
print("Execution time:"); print(execution)

import matplotlib.pyplot as plt
fig,axes = plt.subplots(nrows=1,ncols=2)
pplot = axes[0]
splot = axes[1]

if plot_analysis:
    plt.suptitle("Time including analysis")
    execution = execution+analysis
else:
    plt.suptitle("Time for iteration")

for p in range(len(processors)):
    splot.semilogy(sizes,execution[p,:],label="p=%d" % processors[p])
splot.legend()
for s in range(len(sizes)):
    pplot.semilogy(processors,execution[:,s],label="slocal=%d" % sizes[s])
pplot.legend()

plt.savefig("lulesh.pdf")
#plt.show()

