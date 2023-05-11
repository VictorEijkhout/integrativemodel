/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2023
 ****
 **** mpi_kernel.cxx: Implementations of the MPI kernel classes
 ****
 ****************************************************************/

#include "mpi_kernel.h"

using std::array;
using std::string;
using fmt::print,fmt::format;
using std::shared_ptr;

template<int d>
mpi_kernel<d>::mpi_kernel( shared_ptr<object<d>> obj )
  : kernel<d>::kernel(obj) {
};

template class mpi_kernel<1>;
template class mpi_kernel<2>;
template class mpi_kernel<3>;
