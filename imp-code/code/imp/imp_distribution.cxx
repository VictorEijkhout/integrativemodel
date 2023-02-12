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
    ( const coordinate<index_int,d>& dom,const decomposition<d>& procs,
      distribution_type type ) {
  // give me a unique distribution number
  my_distribution_number = distribution_number++;

  // distribution construction
  using I = index_int;                              I domain_size_reconstruct{1};
  /* */                                             int procs_reconstruct{1};
  array< vector<I>,d > starts,ends; // domain start points in each dimension
  array< int,d > procs_d;           // number of processes in each dimension
  for (int id=0; id<d; id++) {
    auto pd = procs.size_of_dimension(id);
    if (type==distribution_type::orthogonal) {
      starts.at(id)  = split_points(dom.at(id),pd);   
      ends  .at(id)  = vector<I>( starts.at(id).begin()+1,starts.at(id).end() );
    } else if (type==distribution_type::replicated) {
    }                                               domain_size_reconstruct *= ends.at(id).back();
    procs_d.at(id) = pd;   procs_reconstruct *= procs_d.at(id);
  }
  assert                                          ( domain_size_reconstruct==dom.span() );
  assert                                          ( procs_reconstruct==procs.global_volume() );

  for (int id=0; id<d; id++) {
    // allocate vector for domain side segments;
    patches.at(id) = 
      [] (int pd,const vector<I>& starts,const vector<I>& ends) {
	vector< indexstructure<I,1> > segments;
	for (int ip=0; ip<pd; ip++) {
	  coordinate<I,1>
	    first( starts.at(ip) ),
	    last ( ends  .at(ip) );
	  segments.push_back( indexstructure<I,1>( contiguous_indexstruct<I,1>(first,last-1) ) );
	}
	return segments;
      } ( procs_d.at(id),starts.at(id),ends.at(id) );
  }
};

template<int d>
bool distribution<d>::compatible_with( const distribution<d>& other ) const {
  return my_distribution_number==other.my_distribution_number;
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
