#! /usr/bin/env python

import subprocess
import sys

if len(sys.argv)==1:
    domsizes = [1000,3000,10000]
else:
    machine = sys.argv[1]
    proc = subprocess.Popen(["make", "-f","../Makefile","listsizes","TACC_SYSTEM=%s" % machine],
                            stdout=subprocess.PIPE)
    (domsizes, err) = proc.communicate()
    domsizes = domsizes.strip().split()
    domsizes = [ int(s) for s in domsizes ]

class Table():
    def __init__(self,name=None,table=None):
        global domsizes
        if name is not None:
            table = []
            for size in domsizes:
                row = []
                for thick in [1,10,100]:
                    try :
                        with open("laptime-%s-%d-%d.out" \
                                  % (name,size,thick),"r") as tin:
                            time = tin.readline().strip().split()[1]
                            time = float(time)
                            row.append(time)
                    except:
                        row.append(0.0)
                table.append(row)
            self.table = table
        if table is not None:
            self.table = table
    def div(self,other):
        table = []
        for row1,row2 in zip(self.table,other.table):
            row = []
            for t1,t2 in zip(row1,row2):
                if t2==0.0:
                    row.append(0.0)
                else:
                    row.append(t1/t2)
            table.append(row)
        return Table(table=table)
    def minus(self,other):
        table = []
        for row1,row2 in zip(self.table,other.table):
            row = []
            for t1,t2 in zip(row1,row2):
                row.append(t1-t2)
            table.append(row)
        return Table(table=table)
    def percent(self):
        table = []
        for trow in self.table:
            row = []
            for t in trow:
                row.append( int(100*t) )
            table.append(row)
        return Table(table=table)
    def __str__(self):
        table_string = ""
        for row in self.table:
            table_string += str(row)+"\n"
        return table_string


print "2 - 1 : overhead of naive MPI over embarrassingly parallel, using derived type"
base_time = Table("1")
mpi_time = Table("2")
print mpi_time.minus(base_time).div(base_time).percent()

print "2f - 1 : overhead of naive MPI over embarrassingly parallel, using contiguous data"
base_time = Table("1")
mpi_time = Table("2f")
print mpi_time.minus(base_time).div(base_time).percent()

print "2 - 2f : packing overhead"
mpi_time = Table("2")
fake_time = Table("2f")
print mpi_time.minus(fake_time).div(mpi_time).percent()

print "3 - 1 : mpi overhead, pipelined loop body"
base_time = Table("1")
mpi_time = Table("3")
print mpi_time.minus(base_time).div(base_time).percent()

print "4 - 1 : mpi overhead, fully pipelined algorithm"
base_time = Table("1")
mpi_time = Table("4")
print mpi_time.minus(base_time).div(base_time).percent()

print "4h - 1 : mpi overhead, using helper thread for MPI"
base_time = Table("1")
mpi_time = Table("4h")
print mpi_time.minus(base_time).div(base_time).percent()

print "2 - 4 : improvement of pipelined code over naive"
naive_time = Table("2")
pipe_time = Table("4")
print naive_time.minus(pipe_time).div(naive_time).percent()

print "2 - 4h : improvement of pipelined code over helper-thread"
naive_time = Table("2")
pipe_time = Table("4h")
print naive_time.minus(pipe_time).div(naive_time).percent()
