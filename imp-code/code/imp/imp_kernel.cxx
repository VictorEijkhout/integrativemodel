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

using std::pair;
using std::vector;
using std::shared_ptr;
using std::function;
using fmt::format;

/*
 * Dependencies
 */
template<int d>
void dependency<d>::analyze() {
  auto operated = obj.operate( op );
  beta = object<d>( operated );
};

/*
 * Kernels
 */
template<int d>
kernel<d>::kernel( shared_ptr<object<d>> out )
  : output(out) {;
};

template<int d>
void kernel<d>::add_dependency( std::shared_ptr<object<d>> input,ioperator<index_int,d> op ) {
  inputs.push_back( dependency<d>(input,op) );
};

template<int d>
void kernel<d>::analyze_dependencies() {
  for ( auto& dep : inputs )
    dep.analyze();
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

