/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2023
 ****
 **** mpi_decomp.cxx: Implementations of the MPI decompositions classes
 ****
 ****************************************************************/

#include "mpi_env.h"
#include "mpi_decomp.h"
// from fmt
using fmt::print;
// from optional
#include <optional>
// using std::has_value;

/*!
 * Multi-d decomposition from default grid from environment,
 * delegates to the more explicit one
 */
template<int d>
mpi_decomposition<d>::mpi_decomposition( const mpi_environment& env )
  : mpi_decomposition<d>( endpoint<int,d>(env.nprocs()),env.procid() ) {
};

/*!
 * Multi-d decomposition from explicit processor grid layout
 * \todo unify the this_proc function & recorded process coordinate
 */
template<int d>
mpi_decomposition<d>::mpi_decomposition
    ( const coordinate<int,d> &grid,int procid )
      : decomposition<d>(grid) {
  // record our process number
  _procno = procid;
  // record our process coordinate, twice
  auto my_proc_coord = decomposition<d>::domain_layout().location_of_linear(procid);
  this->push_back( my_proc_coord );
  this->this_proc = [my_proc_coord] () -> coordinate<int,d> {
    return my_proc_coord; };
};

/*
 * Stuff related to this MPI process
 */

//! Our process rank. \todo derive this from the coordinate?
template<int d>
int mpi_decomposition<d>::procno() const {
  return _procno;
};

// //! This process as d-dimensional coordinate
// template<int d>
// const coordinate<int,d>& mpi_decomposition<d>::this_proc() const {
//   if ( not proc_coord.has_value() )
//     proc_coord = this->coordinate_from_linear(_procno);
//   return proc_coord.value();
// };

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
template class mpi_decomposition<3>;
