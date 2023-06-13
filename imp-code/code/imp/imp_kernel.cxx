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
  const auto& input_dist = input_object->get_distribution();
  beta = input_dist.operate( op );
  for ( const auto& p : input_dist.local_procs() ) {
    const auto& local_beta_domain = beta.local_domain(p);
    for ( const auto& q : input_dist.global_procs() ) {
      // calculate pq intersection
      const auto& q_domain = input_object.local_domain(q);
      auto pq_intersection  = p_domain.intersect(q_domain);
      if ( not pq_intersection.is_empty() ) {
	print( "Dependency: {} <- {}\n",p,q );
      }
    }
  }
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
void kernel<d>::analyze() {
  for ( auto& dep : inputs )
    dep.analyze();
};

template<int d>
void kernel<d>::set_localexecutefn( std::function< kernel_function_proto(d) > f ) {
  localexecutefn = f;
};

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

