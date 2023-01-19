/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2022
 ****
 **** imp_decomp.cxx: Implementations of the decomposition base classes
 ****
 ****************************************************************/

#include "imp_distribution.h"
using std::vector;

template<int d>
distribution<d>::distribution
    ( const decomposition<d>& procs, const coordinate<index_int,d>& c ) {
  for (int id=0; id<d; id++) {
    int length = procs.size_of_dimension(id);
    vector<index_int> starts = split_points(length);
    vector< indexstructure<index_int,1> > segments(length);
    patches.at(id) = segments;
  }
};

template class distribution<1>;
template class distribution<2>;

