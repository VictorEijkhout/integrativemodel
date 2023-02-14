/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2023
 ****
 **** omp_object.cxx: Implementations of the OMP object classes
 ****
 ****************************************************************/

#include "omp_object.h"

using std::array;
using std::string;
using fmt::print,fmt::format;

template<int d>
omp_object<d>::omp_object( const omp_distribution<d>& dist )
  : object<d>::object(dist) {
};

template class omp_object<1>;
template class omp_object<2>;
template class omp_object<3>;
