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
using std::vector, std::array;

/*!
 * d-dimensional distribution as orthogonal product of 1-d block distributions
 */
template<int d>
distribution<d>::distribution
    ( const coordinate<index_int,d>& c,const decomposition<d>& procs ) {
  using I = index_int;                              I domain_size_reconstruct{1};
  /* */                                             int procs_reconstruct{1};
  array< vector<I>,d > starts; // domain start points in each dimension
  array< int,d > procs_d;      // number of processes in each dimension
  for (int id=0; id<d; id++) {
    starts.at(id)  = procs.split_points_d(c,id);    domain_size_reconstruct *= starts.at(id).back();
    procs_d.at(id) = procs.size_of_dimension(id);   procs_reconstruct *= procs_d.at(id);
  }
  assert                                          ( domain_size_reconstruct==c.span() );
  assert                                          ( procs_reconstruct==procs.global_volume() );

  for (int id=0; id<d; id++) {
    // allocate vector for domain side segments;
    patches.at(id) = 
      [] (int pd,vector<I> starts) {
	vector< indexstructure<I,1> > segments;
	for (int ip=0; ip<pd; ip++) {
	  coordinate<I,1>
	    first( starts.at(ip) ),
	    last ( starts.at(ip+1) );
	  segments.push_back( indexstructure<I,1>( contiguous_indexstruct<I,1>(first,last-1) ) );
	}
	return segments;
      } ( procs_d.at(id),starts.at(id) );
  }
};

/*! Distributions own a local domain on each process
  For MPI that is strictly local, for OpenMP the whole domain
*/
template<int d>
const indexstructure<index_int,d>& distribution<d>::local_domain() const {
  return _local_domain;
};

template class distribution<1>;
template class distribution<2>;
template class distribution<3>;
