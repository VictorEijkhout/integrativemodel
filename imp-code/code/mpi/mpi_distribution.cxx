#include "imp_decomp.h"
#include "mpi_decomp.h"

template<int d>
mpi_distribution<d>::mpi_distribution
    ( const mpi_decomposition<d>& d,const coordinate<index_int,d> c )
      : distribution(d,c) {
};
