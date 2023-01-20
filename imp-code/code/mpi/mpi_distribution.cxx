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

#include "mpi_decomp.h"
#include "mpi_distribution.h"

using std::array;

template<int d>
mpi_distribution<d>::mpi_distribution
    ( const mpi_decomposition<d>& procs,const coordinate<index_int,d>& domain )
      : distribution<d>(procs,domain) {
  using I = index_int;
  coordinate<I,d> first,last;
  auto p = procs.procno();
  for ( int id=0; id<d; id++) {
    first.at(id) = patches.at(id).at(p);
    last.at(id) = patches.at(id).at(p+1);
  }
  local_domain = indexstructure<I,d>( contiguous_indexstruct<I,d>(first,last) );
};

template<int d>
indexstructure<index_int,d> mpi_distribution<d>::local_domain() const {
  
}

template class mpi_distribution<1>;
template class mpi_distribution<2>;
template class mpi_distribution<3>;
