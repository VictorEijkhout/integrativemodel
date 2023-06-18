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

#include "imp_decomp.h"
#include <cassert>

#include <iostream>
using std::cout;
#include <numeric>
using std::accumulate;
#include <functional>
using std::multiplies;

using std::array,std::vector,std::string;
using std::shared_ptr,std::make_shared;
using fmt::format,fmt::print;

/****
 **** Construction
 ****/

/*!
 * Make a d-dimensional process grid based on the environment
 * This delegates to the constructor from a coordinate
 * by computing the far-point of the environment.
 */
template<int d>
decomposition<d>::decomposition( const environment& env )
  : decomposition<d>( endpoint<int,d>(env.nprocs()) ) {
};

/*!
 * Make a d-dimensional process grid from the far-point coordinate.
 * This only copies to internal data.
 */
template<int d>
decomposition<d>::decomposition( const coordinate<int,d> grid )
  : _process_grid(grid) {
};

//! How many processes in dimension `d'?
template<int d>
int decomposition<d>::size_of_dimension(int nd) const {
  return _process_grid.at(nd);
};

//! Start points in dimension `d' of a domain, represented by a coordinate<d>
template<int d>
vector<index_int> decomposition<d>::split_points_d
    ( const coordinate<index_int,d>& c,int id ) const {
  index_int domain_d = c.at(id);
  int procs_d = size_of_dimension(id);
  return split_points(domain_d,procs_d);
};

//! Number of local domains: 1 for MPI, all for OpenMP
template<int d>
int decomposition<d>::local_volume() const {
  return local_procs().size();
};

//! Number of global domains
template<int d>
int decomposition<d>::global_volume() const { return _process_grid.span(); };

//! Get the local number where this domain is stored. Domains are multi-dimensionally numbered.
template<int d>
int decomposition<d>::domain_local_number( const coordinate<int,d> &dcoord ) const {
  // for ( int i=0; i<mdomains.size(); i++) {
  //   if (mdomains.at(i)==d) return i;
  // }
  throw(fmt::format("Domain has no localization"));
};

/*
 * Coordinate conversion stuff
 */
template<int d>
int decomposition<d>::linearize( const coordinate<int,d> &p ) const {
  return _process_grid.linear_location_of(p);
};

/*!
  \todo check this calculation
  Reference: http://stackoverflow.com/questions/10903149/how-do-i-compute-the-linear-index-of-a-3d-coordinate-and-vice-versa
*/
template<int d>
coordinate<int,d> decomposition<d>::coordinate_from_linear(int p) const {
  coordinate<int,d> pp;
  for (int id=d-1; id>=0; id--) {
    int dsize = _process_grid.at(id);
    if (dsize==0)
      throw(format("weird layout <<{}>>",_process_grid.as_string()));
    pp.at(id) = p%dsize; p = p/dsize;
  };
  return pp;
};

template<int d>
int decomposition<d>::linear_location_of( const coordinate<int,d>& c ) const {
  return _process_grid.linear_location_of(c);
};

/*
 * Ranging
 */
template<int d>
decomposition<d>::iter decomposition<d>::begin() {
  return decomposition<d>::iter(*this,0);
};

template<int d>
decomposition<d>::iter decomposition<d>::end() {
  return decomposition<d>::iter(*this,this->global_volume());
};

template<int d>
bool decomposition<d>::iter::operator!=( const decomposition<d>::iter& cmp ) const {
  return count!=cmp.count;
};

template<int d>
bool decomposition<d>::iter::operator==( const decomposition<d>::iter& cmp ) const {
  return count==cmp.count;
};

template<int d>
void decomposition<d>::iter::operator++() {
  count++;
};

template<int d>
coordinate<int,d> decomposition<d>::iter::operator*() const {
  auto c = decomp.coordinate_from_linear(count);
  return c;
};

/*
 * Util
 */
template<int d>
string decomposition<d>::as_string() const {
  return "decomp";
};

template class decomposition<1>;
template class decomposition<2>;
template class decomposition<3>;
