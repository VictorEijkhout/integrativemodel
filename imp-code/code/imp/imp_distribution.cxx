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
using std::string;
using fmt::format;

/*!
 * Distribution constructor from
 * d-dimensional distribution as orthogonal product of 1-d block distributions.
 *
 * Arguments:
 * - dom : the global domain
 * - procs : processor decomposition
 * - type : distribution type
 *     orthogonal: split points in all dimensions, then make patches
 *     replicated : each process gets the same zero/end point, then make patches
 *
 */
template<int d>
distribution<d>::distribution
    ( const domain<d>& dom,const decomposition<d>& procs,
      distribution_type type )
      : _global_domain( dom )
      , my_decomposition(procs)
      , my_distribution_type(type) {
  // store global domain
  // give me a unique distribution number
  my_distribution_number = distribution_number++;

  // distribution construction
  using I = index_int;                            I domain_size_reconstruct{1};
  /* */                                           int procs_reconstruct{1};
  array< vector<I>,d > starts,ends; // domain start points in each dimension
  array< int,d > procs_d;           // number of processes in each dimension
  if (type==distribution_type::orthogonal) {
    // orthogonal : adjacent blocks in both directions
    const auto& first = dom.first_index();
    const auto& last = dom.last_index();
    for (int id=0; id<d; id++) {
      auto pd = procs.size_of_dimension(id);
      starts.at(id) = split_points(first.at(id),last.at(id),pd);   
      ends  .at(id) = vector<I>( starts.at(id).begin()+1,starts.at(id).end() );
      /* */                                       domain_size_reconstruct *= ends.at(id).back();
      procs_d.at(id) = pd;                        procs_reconstruct *= procs_d.at(id);
    }
  } else if (type==distribution_type::replicated) {
    // replicated: a single block for each proc (actually this is also orthogonal)
    const auto& first = dom.first_index();
    const auto& last = dom.last_index();
    for (int id=0; id<d; id++) {
      auto pd = procs.size_of_dimension(id);
      // all domains start/end at the domain boundaries
      starts.at(id) = vector<I>(pd,first.at(id));
      ends  .at(id) = vector<I>(pd,last .at(id));
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
  for (int p=0; p<procs.global_volume(); p++) {
    const coordinate<int,d> this_proc = procs.coordinate_from_linear(p);
    coordinate<I,d> first,last;
    for ( int id=0; id<d; id++) {
      auto pd = this_proc.at(id);
      first.at(id) = this->patches.at(id).at(pd).first_index().at(0);
      last .at(id) = this->patches.at(id).at(pd). last_index().at(0) - 1;
    }
    this->_local_domains.push_back( domain<d>( contiguous_indexstruct<I,d>( first,last ) ) );
  }
};

/*!
 * Test identity of two distributions
 * by looking at their unique ID number
 * \todo this should really be by process grid & domain
 */
template<int d>
bool distribution<d>::compatible_with( const distribution<d>& other ) const {
  return my_distribution_number==other.my_distribution_number;
};

//! Force compatibility by throwing an exception if not
template<int d>
void distribution<d>::assert_compatible_with( const distribution<d>& other ) const {
  if ( not compatible_with(other) )
    throw( format("Can not combine objects on different distributions: {} / {}",
		  my_distribution_number,other.my_distribution_number ) );
};

//! Test a distribution being replicated
template<int d>
void distribution<d>::assert_replicated() const {
  if ( my_distribution_type!=distribution_type::replicated )
    throw("distribution needs to be replicated");
};

/*!
 * The local domain as function of process coordinate.
 */
template<int d>
const domain<d>& distribution<d>::local_domain( const coordinate<int,d>& p ) const {
  int p_num = my_decomposition.linearize(p);
  return _local_domains.at(p_num);
};
/*!
 * If there is a `this' proc, return its local domain.
 * This will only work for MPI; it will throw for OpenMP
 */
template<int d>
const domain<d>& distribution<d>::local_domain() const {
  const auto& local_p = my_decomposition.this_proc();
  return local_domain(local_p);
};

/*!
 * The set of all local domains.
 * For MPI that's just one, for OpenMP all.
 */
template<int d>
const vector<domain<d>>& distribution<d>::local_domains() const {
  return _local_domains;
};

/*! Distributions are built on a global domain,
 * this is set in the base constructor.
 * \todo should we inherit from the `domain' class?
*/
template<int d>
const domain<d>& distribution<d>::global_domain() const {
  return _global_domain;
};

/*
 * Instantiations
 */
template class domain<1>;
template class domain<2>;
template class domain<3>;

template class distribution<1>;
template class distribution<2>;
template class distribution<3>;
