#!/bin/bash

#SBATCH -J luleshtest
#SBATCH -A A-ccsc
#SBATCH -o luleshtest.o%j
#SBATCH -n 384
#SBATCH -N 16
#SBATCH -p skx-normal
#SBATCH -t 01:00:00
##SBATCH --mail-user=eijkhout@tacc.utexas.edu
##SBATCH --mail-type=ALL

set -x

for p in 16 50 100 180 384 ; do
  for s in 10 30 100 300 ; do
    echo "Processors: $p"
    echo "Size: $s"
    ibrun -np $p ./lulesh -elocal $s
  done
done
