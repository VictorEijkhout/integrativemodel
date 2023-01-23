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
#include <cassert>
using std::vector;

/*!
 * d-dimensional distribution as orthogonal product of 1-d block distributions
 */
template<int d>
distribution<d>::distribution
    ( const coordinate<index_int,d>& c,const decomposition<d>& procs ) {
  using I = index_int;
  I domain_size_reconstruct{1}; int procs_reconstruct{1};
  for (int id=0; id<d; id++) {
    vector<I> starts = procs.split_points_d(c,id);
    domain_size_reconstruct *= starts.back();
    // allocate vector for domain sides;
    vector< indexstructure<I,1> > segments;
    int procs_d = procs.size_of_dimension(id); procs_reconstruct *= procs_d;
    for (int ip=0; ip<procs_d; ip++) {
      coordinate<I,1> first( starts.at(ip) ), last( starts.at(ip+1) );
      segments.push_back( indexstructure<I,1>( contiguous_indexstruct<I,1>(first,last) ) );
    }
    patches.at(id) = segments;
  }
  assert( domain_size_reconstruct==c.span() );
  assert( procs_reconstruct==procs.global_volume() );
};

template class distribution<1>;
template class distribution<2>;
template class distribution<3>;
