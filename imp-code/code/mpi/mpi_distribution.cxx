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

//! The constructor builds the single local domain
template<int d>
mpi_distribution<d>::mpi_distribution
    ( const mpi_decomposition<d>& procs,const coordinate<index_int,d>& domain )
      : distribution<d>(procs,domain) {
  auto p = procs.procno();
  using I = index_int;
  coordinate<I,d> first,last;
  for ( int id=0; id<d; id++) {
    first.at(id) = this->patches.at(id).at(p).first_index().at(0);
     last.at(id) = this->patches.at(id).at(p). last_index().at(0);
  }
  _local_domain = indexstructure<I,d>
    ( contiguous_indexstruct<I,d>( first,last ) );
};

//! MPI distributions own a local domain on each process
template<int d>
const indexstructure<index_int,d>& mpi_distribution<d>::local_domain() const {
  return _local_domain;
};

template class mpi_distribution<1>;
template class mpi_distribution<2>;
template class mpi_distribution<3>;
