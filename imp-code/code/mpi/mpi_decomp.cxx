/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2022
 ****
 **** mpi_decomp.cxx: Implementations of the MPI decompositions classes
 ****
 ****************************************************************/

#include "mpi_env.h"
#include "mpi_decomp.h"
using fmt::print;

//! Multi-d decomposition from default grid from environment
template<int d>
mpi_decomposition<d>::mpi_decomposition( const mpi_environment& env )
  : mpi_decomposition<d>( env,coordinate<int,d>(env) ) {
};

//! Multi-d decomposition from explicit processor grid layout
template<int d>
mpi_decomposition<d>::mpi_decomposition
        ( const mpi_environment& env,const coordinate<int,d> &grid)
  : decomposition<d>(grid) {
  int procid = env.procid(); int over = env.get_over_factor();
  for ( int local=0; local<over; local++) {
    // coordinate_from_linear
    push_back( get_domain_layout().location_of_linear(over*procid+local) );
  }
};

/*!
  A factory for making new distributions from this decomposition
*/
template<int d>
void mpi_decomposition<d>::set_decomp_factory() {
  // new_block_distribution = [this] (index_int g) -> shared_ptr<distribution> {
  //   return shared_ptr<distribution>( make_shared<mpi_block_distribution>(*this,g) );
  // };
};

template<int d>
std::string mpi_decomposition<d>::as_string() const {
  return "mpidecomp"; // fmt::format("MPI decomposition <<{}>>",decomposition::as_string());
};

template class mpi_decomposition<1>;
template class mpi_decomposition<2>;
