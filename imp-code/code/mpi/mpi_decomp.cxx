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

//! Multi-d decomposition from explicit processor grid layout
template<int d>
mpi_decomposition::mpi_decomposition( const mpi_environment& env,const coordinate<int,d> &grid)
  : decomposition<d>(grid) {
  int mytid = env.mytid(); int over = env.get_over_factor();
  for ( int local=0; local<over; local++) {
    auto mycoord = this->coordinate_from_linear(over*mytid+local);
    try {
      add_domain(mycoord);
    } catch (...) { fmt::print("trouble adding domain\n"); };
  }
  set_decomp_factory(); 
};

/*!
  A factory for making new distributions from this decomposition
*/
template<int d>
void mpi_decomposition::set_decomp_factory() {
  new_block_distribution = [this] (index_int g) -> shared_ptr<distribution> {
    return shared_ptr<distribution>( make_shared<mpi_block_distribution>(*this,g) );
  };
};

template<int d>
std::string mpi_decomposition::as_string() const {
  return "mpidecomp"; // fmt::format("MPI decomposition <<{}>>",decomposition::as_string());
};

template class mpi_decomposition<1>;
template class mpi_decomposition<2>;
