/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2023
 ****
 **** omp_distribution.cxx: Implementations of the OMP decompositions classes
 ****
 ****************************************************************/

#include "omp_decomp.h"
#include "omp_distribution.h"

using std::array;

//! The constructor builds the single local domain
template<int d>
omp_distribution<d>::omp_distribution
    ( const coordinate<index_int,d>& domain, const omp_decomposition<d>& procs,
      distribution_type type )
      : distribution<d>(domain,procs,type) {
  using I = index_int;
  coordinate<I,d> first,last = domain-1;
  for ( int id=0; id<d; id++)
    first.at(id) = 0;
  this->_local_domain = indexstructure<I,d>
    ( contiguous_indexstruct<I,d>( first,last ) );
};

//! Function to produce a single scalar, replicated over all processes
template<int d>
omp_distribution<d> replicated_scalar_distribution( const omp_decomposition<d>& dist) {
  return omp_distribution( coordinate<index_int,d>(1),dist,distribution_type::replicated );
};

/*
 * Instantiations
 */
template class omp_distribution<1>;
template class omp_distribution<2>;
template class omp_distribution<3>;

template omp_distribution<1> replicated_scalar_distribution<1>(const omp_decomposition<1>&);
template omp_distribution<2> replicated_scalar_distribution<2>(const omp_decomposition<2>&);
template omp_distribution<3> replicated_scalar_distribution<3>(const omp_decomposition<3>&);

