/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2023
 ****
 **** mpi_object.cxx: Implementations of the MPI object classes
 ****
 ****************************************************************/

#include "mpi_object.h"

using std::array;
using std::string;
using fmt::print,fmt::format;

template<int d>
mpi_object<d>::mpi_object( const mpi_distribution<d>& dist )
  : object<d>::object(dist) {
};

template class mpi_object<1>;
template class mpi_object<2>;
template class mpi_object<3>;
