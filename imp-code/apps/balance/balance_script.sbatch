#!/bin/bash
#SBATCH -J balancetest
#SBATCH -o balancetest.o%j    # Output and error file name (%j expands to jobID)
#SBATCH -N 16          # Total number of nodes
#SBATCH -n 256            # Total number of mpi tasks requested
#SBATCH -p skx-normal  # Queue (partition) name -- normal, development, etc.
#SBATCH -t 12:00:00     # Run time (hh:mm:ss) - 1.5 hours
#SBATCH -A A-ccsc
#SBATCH --mail-type=end
#SBATCH --mail-user=eijkhout@tacc.utexas.edu

nglobal=25600
for node in 1 2 4 8 16 ; do
  export ranks=$((node*8))
  nlocal=$(( nglobal / ranks ))
  export steps=100
  # $((ranks*8))
  echo "Running on ${node} nodes, ${ranks} ranks, ${steps} steps"

  echo "reference run"
  ibrun -np ${ranks} ./balance -n ${nlocal} -s ${steps} -1

  echo "imp run"
  ibrun -np ${ranks} ./balance -n ${nlocal} -s ${steps}

  echo "imp timing"
  ibrun -np ${ranks} ./balance -n ${nlocal} -s ${steps} -trace \
     | awk '/Maximum/ { t = t+$4} END {print t}'

  echo "unbalanced run"
  ibrun -np ${ranks} ./balance -n ${nlocal} -s ${steps} -B

  echo "unbalanced timing"
  ibrun -np ${ranks} ./balance -n ${nlocal} -s ${steps} -B -trace \
     | awk '/Maximum/ { t = t+$4} END {print t}'

  echo
# -collective 3 -sparse
done
