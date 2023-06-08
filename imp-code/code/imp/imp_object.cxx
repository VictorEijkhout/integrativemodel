/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2023
 ****
 **** imp_object.cxx: Implementations of the object base classes
 ****
 ****************************************************************/

#include "imp_object.h"
#include <cassert>
#include <cmath>

using std::vector, std::array;
using fmt::format;

#include <ranges>
// to<> not yet in gcc12
//#include <range/v3/all.hpp>
#ifdef RANGES_V3_ALL_HPP
namespace rng = ranges;
#else
namespace rng = std::ranges;
#endif

/*!
 * D-dimensional object.
 * Allocate data with size of local_domain
 */
template<int d>
object<d>::object( const distribution<d>& dist )
  : distribution<d>(dist)
  , _data( vector<double>( dist.global_domain().volume() ) ) {
};

/*
 * Data manipulation
 */

//! Return a star-pointer to the numerical data, mutable
template<int d>
double* object<d>::raw_data() {
  return _data.data();
};

//! Return a star-pointer to the numerical data, constant
template<int d>
double const * object<d>::raw_data() const {
  return _data.data();
};

//! Return access to the numerical data, mutable
template<int d>
vector<double>& object<d>::data() {
  return _data;
};

//! Return access to the numerical data, constant
template<int d>
const vector<double>& object<d>::data() const {
  return _data;
};

/*!
 * Construct an operated distribution.
 * \todo the data copy is wrong if the op is anything but a shift
 */
template<int d>
object<d> object<d>::operate( const ioperator<index_int,d>& op ) const {
  auto operated = object<d>( get_distribution().operate(op) );
  rng::copy( _data.begin(),_data.end(),
	     operated._data.begin() );
  return operated;
};

/*
 * Operations
 */

/*! Set the whole vector to a constant
 * This has no error checking on the constant being
 * constant over processes or so.
 */
template<int d>
void object<d>::set_constant( double x ) {
  for ( auto& e : _data )
    e = x;  
};

/*!
 * Add another object into this one.
 * The compatibility test is a little meagre.
 */
template<int d>
object<d>& object<d>::operator+=( const object<d>& other ) {
  this->assert_compatible_with(other);
  double *xdata = raw_data(); double const * const ydata = other.raw_data();
  for ( size_t i=0; i<this->local_domain().volume(); i++ )
    xdata[i] += ydata[i];
  return *this;
};

/*!
 * Compute the norm on the local domain.
 * The `compute_norm' function does an allreduce over this,
 * which does different things in MPI vs OpenMP.
 */
template<int d>
double object<d>::local_norm() const {
  double norm_value;
  double const * xdata = raw_data();
  for ( size_t i=0; i<this->local_domain().volume(); i++ )
    norm_value += pow( xdata[i],2 );
  return sqrt( norm_value );
};

/*! Compute norm.
 * This depends on the local norm which is overridden for OpenMP
 * Also: the reduction is a no-op in OpenMP.
 * \todo Test if the inputs are on the same process set.
 */
template<int d>
void norm( object<d>& scalar,const object<d>& thing,const environment& env ) {
  scalar.assert_replicated();
  double norm_value;
  norm_value = thing.local_norm();
  norm_value = env.allreduce_d( pow(norm_value,2.) );
  scalar.set_constant( norm_value );
};

/*!
 * Compute the inner_product on the local domain.
 * The `compute_inner_product' function does an allreduce over this,
 * which does different things in MPI vs OpenMP.
 */
template<int d>
double object<d>::local_inner_product( const object<d>& other ) const {
  double inner_product_value;
  double const * xdata = raw_data();
  double const * ydata = other.raw_data();
  for ( size_t i=0; i<this->local_domain().volume(); i++ )
    inner_product_value += xdata[i] * ydata[i];
  return inner_product_value;
};

/*! Compute inner_product.
 * This depends on the local inner_product which is overridden for OpenMP
 * Also: the reduction is a no-op in OpenMP.
 * \todo Test if the inputs are on the same process set.
 */
template<int d>
void inner_product
    ( object<d>& scalar,const object<d>& thing,const object<d>& other,const environment& env ) {
  scalar.assert_replicated();
  thing.assert_compatible_with(other);
  double inner_product_value;
  inner_product_value = thing.local_inner_product(other);
  inner_product_value = env.allreduce_d( inner_product_value );
  scalar.set_constant( inner_product_value );
};

template class object<1>;
template class object<2>;
template class object<3>;

template void norm<1>( object<1>&,const object<1>&,const environment& env );
template void norm<2>( object<2>&,const object<2>&,const environment& env );
template void norm<3>( object<3>&,const object<3>&,const environment& env );

template void inner_product<1>( object<1>&,const object<1>&,const object<1>&,const environment& env );
template void inner_product<2>( object<2>&,const object<2>&,const object<2>&,const environment& env );
template void inner_product<3>( object<3>&,const object<3>&,const object<3>&,const environment& env );
