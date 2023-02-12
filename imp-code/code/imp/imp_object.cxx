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

/*!
 * d-dimensional object as orthogonal product of 1-d block objects
 */
template<int d>
object<d>::object( const distribution<d>& dist )
  : data( vector<double>( dist.local_domain().volume() ) ) {
};

template class object<1>;
template class object<2>;
template class object<3>;
