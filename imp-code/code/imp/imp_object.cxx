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

/*!
 * D-dimensional object.
 * Allocate data with size of local_domain
 */
template<int d>
object<d>::object( const distribution<d>& dist )
  : distribution<d>(dist)
  , _data( vector<double>( dist.local_domain().volume() ) ) {
};

//! Return a star-pointer to the numerical data, mutable
template<int d>
double* object<d>::data() {
  return _data.data();
};

//! Return a star-pointer to the numerical data, constant
template<int d>
double const * object<d>::data() const {
  return _data.data();
};

/*! Set the whole vector to a constant
 * This has no error checking on the constant being
 * constant over processes or so.
 */
template<int d>
void object<d>::set_constant( double x ) {
  for ( auto& e : _data )
    e = x;  
};

template<int d>
object<d>& object<d>::operator+=( const object<d>& other ) {
  this->throw_incompatible_with(other);
  double *xdata = data(); double const * const ydata = other.data();
  for ( size_t i=0; i<this->local_domain().volume(); i++ )
    xdata[i] += ydata[i];
  return *this;
};

template<int d>
double object<d>::local_norm() const {
  double norm_value;
  double const * xdata = data();
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
void compute_norm( object<d>& scalar,const object<d>& thing,const environment& env ) {
  scalar.assert_replicated();
  double norm_value;
  norm_value = thing.local_norm();
  norm_value = env.allreduce_d( pow(norm_value,2.) );
  scalar.set_constant( norm_value );
};

template class object<1>;
template class object<2>;
template class object<3>;

template void compute_norm<1>( object<1>&,const object<1>&,const environment& env );
template void compute_norm<2>( object<2>&,const object<2>&,const environment& env );
template void compute_norm<3>( object<3>&,const object<3>&,const environment& env );
