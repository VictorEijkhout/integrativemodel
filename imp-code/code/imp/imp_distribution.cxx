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
using fmt::format;

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
  using I = index_int;                            I domain_size_reconstruct{1};
  /* */                                           int procs_reconstruct{1};
  array< vector<I>,d > starts,ends; // domain start points in each dimension
  array< int,d > procs_d;           // number of processes in each dimension
  if (type==distribution_type::orthogonal) {
    // orthogonal : adjacent blocks in both directions
    for (int id=0; id<d; id++) {
      auto pd = procs.size_of_dimension(id);
      starts.at(id) = split_points(dom.at(id),pd);   
      ends  .at(id) = vector<I>( starts.at(id).begin()+1,starts.at(id).end() );
      /* */                                       domain_size_reconstruct *= ends.at(id).back();
      procs_d.at(id) = pd;                        procs_reconstruct *= procs_d.at(id);
    }
    assert                                      ( domain_size_reconstruct==dom.span() );
    assert                                      ( procs_reconstruct==procs.global_volume() );
  } else if (type==distribution_type::replicated) {
    // replicated: a single block for each proc (actually this is also orthogonal)
    for (int id=0; id<d; id++) {
      auto pd = procs.size_of_dimension(id);
      starts.at(id) = vector<I>(pd,0);          // for each proc, starts at 0
      ends  .at(id) = vector<I>(pd,dom.at(id)); // for each proc, end at end of domain
      procs_d.at(id) = pd;
    }
  } else throw("unrecognized distribution type");

  for (int id=0; id<d; id++) {
    // create vector for domain side segments;
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

template<int d>
void distribution<d>::throw_incompatible_with( const distribution<d>& other ) const {
  if ( not compatible_with(other) )
    throw( format("Can not combine objects on different distributions: {} / {}",
		  my_distribution_number,other.my_distribution_number ) );
};

/*! Distributions own a local domain on each process
  For MPI that is strictly local, for OpenMP the whole domain
*/
template<int d>
const indexstructure<index_int,d>& distribution<d>::local_domain() const {
  return _local_domain;
};

/*
 * Instantiations
 */
template class distribution<1>;
template class distribution<2>;
template class distribution<3>;
