/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2023
 ****
 **** imp_decomp.cxx: Implementations of the decomposition base classes
 ****
 ****************************************************************/

#include "imp_distribution.h"
using std::vector;

template<int d>
distribution<d>::distribution
    ( const decomposition<d>& procs, const coordinate<index_int,d>& c ) {
  using I = index_int;
  for (int id=0; id<d; id++) {
    I domain_d = c.at(id);
    int procs_d = procs.size_of_dimension(id);
    vector<I> starts = split_points(domain_d,procs_d);
    vector< indexstructure<I,1> > segments(procs_d);
    for (int ip=0; ip<procs_d; ip++) {
      coordinate<I,1> first( starts.at(ip) ), last( starts.at(ip+1) );
      segments.at(ip) = indexstructure<I,1>( contiguous_indexstruct<I,1>(first,last) );
    }
    patches.at(id) = segments;
  }
};

template class distribution<1>;
template class distribution<2>;
template class distribution<3>;
