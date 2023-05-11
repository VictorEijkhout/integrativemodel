/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2023
 ****
 **** imp_kernel.cxx: Implementations of the kernel base classes
 ****
 ****************************************************************/

#include "imp_kernel.h"
#include "imp_functions.h"

#include <cassert>
#include <cmath>

using std::vector;
using std::shared_ptr;
using fmt::format;

#include <functional>
using std::function;

template<int d>
kernel<d>::kernel( shared_ptr<object<d>> out )
  : output(out) {;
};

template<int d>
void kernel<d>::set_localexecutefn( std::function< kernel_function_proto(d) > f ) {
  localexecutefn = f;
};

  // //! Set the task-local context; in the product mode this is overriden.
  // virtual void set_localexecutectx( void *ctx ) {
  //   localexecutectx = ctx; };

template<int d>
void kernel<d> ::setconstant( double v ) {
  set_localexecutefn
    ( function< kernel_function_proto(d) >{
      [v] ( kernel_function_args(d) ) -> void {
	vecsetconstant( kernel_function_call(d),v ); } } );
};

template class kernel<1>;
template class kernel<2>;
template class kernel<3>;

