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
using fmt::format,fmt::print;

/*
 * Dependencies
 */
template<int d>
void dependency<d>::analyze()  {
  if (depends.has_value()) return;
  auto o_num = input_object->object_id();
  depends = vector<task_dependency<d>>();
  const auto& input_dist = input_object->get_distribution();
  beta = input_dist.operate( op );
  //for ( const auto& p_domain : beta.value().local_domains() ) {
  const auto& here = beta.value().local_domains();
  assert( here.size()>0 );
  for ( int p_num=0; p_num<here.size(); p_num++ ) {
    const auto& p_domain = here.at(p_num);
    // I wish we have ranges::enumerate
    const auto& all = input_dist.all_domains();
    assert( all.size()>0 );
    //for ( const auto& q_domain : input_dist.all_domains() ) {
    for ( int q_num=0; q_num<all.size(); q_num++ ) {
      //print("Analyzing on p={} and q={}\n",p_num,q_num);
      const auto& q_domain = all.at(q_num);
      // calculate pq intersection
      auto pq_intersection  = p_domain.intersect(q_domain);
      if ( not pq_intersection.is_empty() ) {
	//print( "Dependency: <- {}\n",q_num );
	depends.value().push_back( {o_num,q_num,pq_intersection} );
      } else {
	//print( "No dependency: <- {}\n",q_num );
      }
    }
  }
};

template<int d>
const vector<task_dependency<d>>& dependency<d>::get_dependencies() const {
  if (not depends.has_value())
    throw( "first analyze" );
  return depends.value();
};

/*
 * Kernels
 */
template<int d>
kernel<d>::kernel( shared_ptr<object<d>> out )
  : output(out) {;
};

/*!
 * Add dependency on some operated object
 */
template<int d>
void kernel<d>::add_dependency( std::shared_ptr<object<d>> input,ioperator<index_int,d> op ) {
  inputs.push_back( dependency<d>(input,op) );
};

/*!
 * Analyzing a kernel means analyzing
 * all its dependencies
 */
template<int d>
void kernel<d>::analyze() {
  for ( auto& dep : inputs )
    dep.analyze();
};

/*!
 * Get the process dependencies of a kernel
 */
template<int d>
const vector<task_dependency<d>>& kernel<d>::get_dependencies(int id) const {
  if ( id>=0 and id<inputs.size() )
    return inputs.at(id).get_dependencies();
  else
    throw( "invalid dependency id" );
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

/*
 * Specializations
 */
template class task_dependency<1>;
template class task_dependency<2>;
template class task_dependency<3>;

template class dependency<1>;
template class dependency<2>;
template class dependency<3>;

template class kernel<1>;
template class kernel<2>;
template class kernel<3>;

