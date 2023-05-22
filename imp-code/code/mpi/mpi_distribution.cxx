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

/*! The constructor builds the single local domain.
 * The `patches' have been set by the base constructor
 * to the start points of each local domain
 */
template<int d>
mpi_distribution<d>::mpi_distribution
    ( const domain<d>& dom, const decomposition<d>& procs,
      distribution_type type )
      : distribution<d>(dom,procs,type) {
  using I = index_int;
  /*
   * Polymorphism
   */
  const coordinate<int,d> this_proc = procs.this_proc();
  this->location_of_first_index =
    [=] ( const coordinate<int,d> &p) -> index_int {
      if (p==this_proc)
	return 0;
      else throw( "Can only ask location_of_first_index on local proc" );
    }; 
  this->operate =
    [decomp=this->my_decomposition,typ=this->my_distribution_type,
     dom=distribution<d>::global_domain()]
    ( const ioperator<index_int,d>& op ) -> distribution<d> {
      domain<d> the_domain( dom );
      domain<d> new_domain( the_domain.operate(op) );
      return mpi_distribution<d>( new_domain,decomp,typ);
    };
};

//! Function to produce a single scalar, replicated over all processes
template<int d>
mpi_distribution<d> replicated_scalar_distribution( const mpi_decomposition<d>& dist) {
  return mpi_distribution
    ( domain<d>( coordinate<index_int,d>(1) ),dist,distribution_type::replicated );
};

/*!
 * New MPI distribution by operating
 * This overrides the base method
 * \todo In fact, does this need the base method? Should that one be virtual?
 */
// template<int d>
// mpi_distribution<d> mpi_distribution<d>::operate( const ioperator<index_int,d>& op ) const {
//   domain<d> the_domain( this->global_domain() );
//   domain<d> new_domain( the_domain.operate(op) );
//   return mpi_distribution<d>
//     ( new_domain,this->my_decomposition,this->my_distribution_type);
// };

/*
 * Instantiations
 */
template class mpi_distribution<1>;
template class mpi_distribution<2>;
template class mpi_distribution<3>;

template mpi_distribution<1> replicated_scalar_distribution<1>(const mpi_decomposition<1>&);
template mpi_distribution<2> replicated_scalar_distribution<2>(const mpi_decomposition<2>&);
template mpi_distribution<3> replicated_scalar_distribution<3>(const mpi_decomposition<3>&);

