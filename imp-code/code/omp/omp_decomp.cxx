/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2025
 ****
 **** omp_decomp.cxx: Implementations of the OMP decompositions classes
 ****
 ****************************************************************/

#include "omp_env.h"
#include "omp_decomp.h"

using fmt::print;

//! Multi-d decomposition from default grid from environment
template<int d>
omp_decomposition<d>::omp_decomposition( const omp_environment& env )
  : omp_decomposition<d>( endpoint<int,d>(env.nprocs()) ) {
};

//! Multi-d decomposition from explicit processor grid layout
template<int d>
omp_decomposition<d>::omp_decomposition
    ( const coordinate<int,d> &grid )
      : decomposition<d>(grid) {
  for ( int procid=0; procid<grid.span(); procid++ ) {
    decomposition<d>::_local_procs.push_back
      ( grid.location_of_linear(procid) );
  }
};

/*
 * Stuff related to this OMP process
 */

//! Our process rank. \todo derive this from the coordinate?
template<int d>
int omp_decomposition<d>::procno() const {
  throw( "no procno for omp" );
};

// //! This process as d-dimensional coordinate
// template<int d>
// const coordinate<int,d>& omp_decomposition<d>::this_proc() const {
//   throw( "no proc coord for omp" );
// };

/*!
  A factory for making new distributions from this decomposition
*/
template<int d>
void omp_decomposition<d>::set_decomp_factory() {
  // new_block_distribution = [this] (index_int g) -> shared_ptr<distribution> {
  //   return shared_ptr<distribution>( make_shared<omp_block_distribution>(*this,g) );
  // };
};

template<int d>
std::string omp_decomposition<d>::as_string() const {
  return "ompdecomp"; // fmt::format("OMP decomposition <<{}>>",decomposition::as_string());
};

template class omp_decomposition<1>;
template class omp_decomposition<2>;
template class omp_decomposition<3>;
