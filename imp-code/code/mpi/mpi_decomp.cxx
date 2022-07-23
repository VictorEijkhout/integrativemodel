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

#include "mpi_decomp.h"

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

