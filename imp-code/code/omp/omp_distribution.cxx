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
    ( const domain<d>& dom, const decomposition<d>& procs,
      distribution_type type )
      : distribution<d>(dom,procs,type) {
  using I = index_int;
  coordinate<I,d> first = dom.first_index(),last = dom.last_actual_index();
  this->_local_domain = domain<d>( contiguous_indexstruct<I,d>( first,last ) );
  /*
   * Polymorphism
   */
  this->location_of_first_index =
    [=] ( const coordinate<int,d> &pcoord) -> index_int
    { 
      auto enc = d->get_enclosing_structure();
      coordinate<index_int,d>
	first = d->first_index_r(pcoord);
      index_int loc = first.linear_location_in(enc);
      return loc;
    };
};

//! Function to produce a single scalar, replicated over all processes
template<int d>
omp_distribution<d> replicated_scalar_distribution( const omp_decomposition<d>& dist) {
  return omp_distribution
    ( domain<d>( coordinate<index_int,d>(1) ),dist,distribution_type::replicated );
};

/*!
 * New OMP distribution by operating
 * This overrides the base method
 * \todo In fact, does this need the base method? Should that one be virtual?
 */
template<int d>
omp_distribution<d> omp_distribution<d>::operate( const ioperator<index_int,d>& op ) const {
  domain<d> the_domain( this->global_domain() );
  domain<d> new_domain( the_domain.operate(op) );
  return omp_distribution<d>
    ( new_domain,this->my_decomposition,this->my_distribution_type);
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

