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
using std::string;
using fmt::print,fmt::format;

//! The constructor builds the single local domain
template<int d>
mpi_distribution<d>::mpi_distribution
    ( const coordinate<index_int,d>& domain, const mpi_decomposition<d>& procs,
      distribution_type type )
      : distribution<d>(domain,procs,type) {
  const coordinate<int,d> this_proc = procs.this_proc();
  using I = index_int;
  coordinate<I,d> first,last;
  for ( int id=0; id<d; id++) {
    auto pd = this_proc.at(id);
    first.at(id) = this->patches.at(id).at(pd).first_index().at(0);
    last .at(id) = this->patches.at(id).at(pd). last_index().at(0);
  }
  this->_local_domain = indexstructure<I,d>
    ( contiguous_indexstruct<I,d>( first,last ) );
  //print( "proc {} local_domain {}\n",this_proc.as_string(),this->_local_domain.as_string() );
};

template<int d>
mpi_distribution<d> replicated_scalar_distribution( const mpi_decomposition<d>& dist) {
  return mpi_distribution( coordinate<index_int,d>(1),dist,distribution_type::replicated );
};

/*
 * Instantiations
 */
template class mpi_distribution<1>;
template class mpi_distribution<2>;
template class mpi_distribution<3>;

template mpi_distribution<1> replicated_scalar_distribution<1>(const mpi_decomposition<1>&);
template mpi_distribution<2> replicated_scalar_distribution<2>(const mpi_decomposition<2>&);
template mpi_distribution<3> replicated_scalar_distribution<3>(const mpi_decomposition<3>&);

