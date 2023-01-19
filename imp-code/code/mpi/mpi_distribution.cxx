/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2023
 ****
 **** mpi_distribution.cxx: Implementations of the MPI decompositions classes
 ****
 ****************************************************************/

#include "imp_decomp.h"
#include "mpi_decomp.h"
#include "mpi_distribution.h"

template<int d>
mpi_distribution<d>::mpi_distribution
    ( const mpi_decomposition<d>& procs,const coordinate<index_int,d>& domain )
      : distribution<d>(procs,domain) {
};

template<int d>
indexstructure<index_int,d> mpi_distribution<d>::local_domain() const;

template class mpi_distribution<1>;
template class mpi_distribution<2>;
