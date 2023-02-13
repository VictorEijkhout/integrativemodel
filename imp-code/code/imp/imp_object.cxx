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


template class object<1>;
template class object<2>;
template class object<3>;
