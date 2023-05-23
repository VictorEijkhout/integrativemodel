/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2023
 ****
 **** Unit tests for the OMP product backend of IMP
 **** based on the CATCH framework (https://github.com/philsquared/Catch)
 ****
 **** unit tests for omp-based distributions
 ****
 ****************************************************************/

#include <stdlib.h>
#include <math.h>

#include "catch2/catch_all.hpp"

#include "omp_distribution.h"

using fmt::format, fmt::print;

using std::make_shared, std::shared_ptr;
using std::string;
using std::vector;

auto &the_env = omp_environment::instance();

TEST_CASE( "creation","[omp][distribution][01]" ) {
  {
    INFO( "1D" );
    coordinate<index_int,1> omega( 10*the_env.nprocs() );
    omp_decomposition<1> procs( the_env );
    REQUIRE_NOTHROW( omp_distribution<1>( omega,procs ) );
  }
  {
    INFO( "2D" );
    coordinate<index_int,2> omega( 10*the_env.nprocs() );
    omp_decomposition<2> procs( the_env );
    REQUIRE_NOTHROW( omp_distribution<2>( omega,procs ) );
  }
}

TEST_CASE( "global domains","[omp][distribution][02]" ) {
  SECTION( "1D" ) {
    const index_int elts_per_proc = pow(10,1);
    const index_int elts_global = elts_per_proc*the_env.nprocs();
    coordinate<index_int,1> omega( elts_global );
    omp_decomposition<1> procs( the_env );
    omp_distribution<1> omega_p( omega,procs );
    REQUIRE_NOTHROW( omega_p.global_domain() );
    indexstructure<index_int,1> global_domain = omega_p.global_domain();
    REQUIRE( global_domain.is_known() );
    REQUIRE_NOTHROW( global_domain.volume() );
    REQUIRE( global_domain.volume()==elts_global );
  }
  SECTION( "2D" ) {
    INFO( "2D" );
    const int points_per_proc = pow(10,2);
    index_int total_points = points_per_proc*the_env.nprocs();
    INFO( "points/proc=" << points_per_proc << ", total=" << total_points );
    auto omega_point = endpoint<index_int,2>( points_per_proc*the_env.nprocs() );
    coordinate<index_int,2> omega( omega_point );
    INFO( "omega=" << omega.as_string() );
    omp_decomposition<2> procs( the_env );
    omp_distribution<2> omega_p( omega,procs );
    REQUIRE_NOTHROW( omega_p.global_domain() );
    indexstructure<index_int,2> global_domain = omega_p.global_domain();
    REQUIRE_NOTHROW( global_domain.volume() );
    // MPI type test:
    index_int check_total_points = the_env.allreduce_ii( global_domain.volume() );
    REQUIRE( check_total_points==total_points );
    // OMP only test
    REQUIRE( global_domain.volume()==total_points );
  }
}

// OpenMP has no single local domain.

TEST_CASE( "local domains","[omp][distribution][03]" ) {
  SECTION( "1D" ) {
    const index_int elts_per_proc = pow(10,1);
    const index_int elts_global = elts_per_proc*the_env.nprocs();
    coordinate<index_int,1> omega( elts_global );
    omp_decomposition<1> procs( the_env );
    omp_distribution<1> omega_p( omega,procs );
    REQUIRE_THROWS( omega_p.local_domain() );
    // indexstructure<index_int,1> local_domain = omega_p.local_domain();
    // REQUIRE( local_domain.is_known() );
    // REQUIRE_NOTHROW( local_domain.volume() );
    // REQUIRE( local_domain.volume()==elts_global );
  }
  SECTION( "2D" ) {
    INFO( "2D" );
    const int points_per_proc = pow(10,2);
    index_int total_points = points_per_proc*the_env.nprocs();
    INFO( "points/proc=" << points_per_proc << ", total=" << total_points );
    auto omega_point = endpoint<index_int,2>( points_per_proc*the_env.nprocs() );
    coordinate<index_int,2> omega( omega_point );
    INFO( "omega=" << omega.as_string() );
    omp_decomposition<2> procs( the_env );
    omp_distribution<2> omega_p( omega,procs );
    REQUIRE_THROWS( omega_p.local_domain() );
    // indexstructure<index_int,2> local_domain = omega_p.local_domain();
    // REQUIRE_NOTHROW( local_domain.volume() );
    // // MPI type test:
    // index_int check_total_points = the_env.allreduce_ii( local_domain.volume() );
    // REQUIRE( check_total_points==total_points );
    // // OMP only test
    // REQUIRE( local_domain.volume()==total_points );
  }
}

TEST_CASE( "replicated distributions","[mpi][distribution][replication][11]" ) {
  {
    INFO( "1D" );
    const int points_per_proc = ipower(10,1);
    index_int total_points = points_per_proc*the_env.nprocs();
    coordinate<index_int,1> omega( total_points );
    omp_decomposition<1> procs( the_env );
    REQUIRE_NOTHROW( omp_distribution<1>( omega,procs,distribution_type::replicated ) );
    omp_distribution<1> repl1( omega,procs,distribution_type::replicated );
    REQUIRE_NOTHROW( repl1.global_domain() );
    indexstructure<index_int,1> global_domain = repl1.global_domain();
    REQUIRE( global_domain.volume()==total_points );
  }
  {
    INFO( "2D" );
    const int points_per_proc = ipower(10,2);
    index_int total_points = points_per_proc*the_env.nprocs();
    coordinate<index_int,2> omega( endpoint<index_int,2>(total_points) );
    omp_decomposition<2> procs( the_env );
    REQUIRE_NOTHROW( omp_distribution<2>( omega,procs,distribution_type::replicated ) );
    omp_distribution<2> repl2( omega,procs,distribution_type::replicated );
    REQUIRE_NOTHROW( repl2.global_domain() );
    indexstructure<index_int,2> global_domain = repl2.global_domain();
    REQUIRE( global_domain.volume()==total_points );
  }
}

TEST_CASE( "replicated scalars","[mpi][distribution][replication][12]" ) {
  {
    INFO( "1D" );
    omp_decomposition<1> procs( the_env );
    REQUIRE_NOTHROW( replicated_scalar_distribution<1>( procs ) );
    auto repl1     = replicated_scalar_distribution<1>( procs );
    REQUIRE_NOTHROW( repl1.global_domain() );
    indexstructure<index_int,1> global_domain = repl1.global_domain();
    REQUIRE( global_domain.volume()==1 );
  }
  {
    INFO( "2D" );
    omp_decomposition<2> procs( the_env );
    REQUIRE_NOTHROW( replicated_scalar_distribution<2>( procs ) );
    auto repl2     = replicated_scalar_distribution<2>( procs );
    REQUIRE_NOTHROW( repl2.global_domain() );
    indexstructure<index_int,2> global_domain = repl2.global_domain();
    REQUIRE( global_domain.volume()==1 );
  }
}

/****
 **** Operations on distribution
 ****/

TEST_CASE( "distribution shifting" ) {
  {
    INFO( "1D" );
    // processors
    omp_decomposition<1> procs( the_env );
    // global domain
    coordinate<index_int,1> omega( procs.domain_layout()*16 );
    index_int total_points = omega.span();
    domain<1> dom(omega);
    // distributed domain
    omp_distribution<1> dist( omega,procs );
    INFO( "original domain: " << dist.global_domain().as_string() );
    REQUIRE( dist.global_domain().first_index()==constant_coordinate<index_int,1>(0) );

    ioperator<index_int,1> right1(">>1");
    REQUIRE_NOTHROW( dom.operate( right1 ) );
    REQUIRE_NOTHROW( dist.operate( right1 ) );
    auto new_dist = dist.operate( right1 );
    REQUIRE_NOTHROW( new_dist.global_domain() );
    REQUIRE( new_dist.global_domain().first_index()==constant_coordinate<index_int,1>(1) );

    REQUIRE_NOTHROW( new_dist.global_domain() );
    auto new_global = new_dist.global_domain();
    INFO( "shifted global : " << new_global.as_string() );
    REQUIRE( new_global.volume()==total_points );

    // INFO( "original local : " << dist.local_domain().as_string() );
    // REQUIRE_NOTHROW( new_dist.local_domain() );
    // auto new_local = new_dist.local_domain();
    // INFO( "shifted local  : " << new_local.as_string() );
    // REQUIRE( new_local.volume()==dist.local_domain().volume() );
  }
  {
    INFO( "2D" );
    // processors
    omp_decomposition<2> procs( the_env );
    // global domain
    coordinate<index_int,2> omega( procs.domain_layout()*16 );
    index_int total_points = omega.span();
    domain<2> dom(omega);
    // distributed domain
    omp_distribution<2> dist( omega,procs );
    INFO( "original domain: " << dist.global_domain().as_string() );
    REQUIRE( dist.global_domain().first_index()==constant_coordinate<index_int,2>(0) );

    ioperator<index_int,2> right1(">>1");
    REQUIRE_NOTHROW( dom.operate( right1 ) );
    REQUIRE_NOTHROW( dist.operate( right1 ) );
    auto new_dist = dist.operate( right1 );
    REQUIRE_NOTHROW( new_dist.global_domain() );
    REQUIRE( new_dist.global_domain().first_index()==constant_coordinate<index_int,2>(1) );

    REQUIRE_NOTHROW( new_dist.global_domain() );
    auto new_global = new_dist.global_domain();
    INFO( "shifted global : " << new_global.as_string() );
    REQUIRE( new_global.volume()==total_points );

    // INFO( "original local : " << dist.local_domain().as_string() );
    // REQUIRE_NOTHROW( new_dist.local_domain() );
    // auto new_local = new_dist.local_domain();
    // INFO( "shifted local  : " << new_local.as_string() );
    // REQUIRE( new_local.volume()==dist.local_domain().volume() );
  }
}

TEST_CASE( "divided distributions","[mpi][distribution][operation][06]" ) {
  {
    INFO( "1D" );
    const int points_per_proc = ipower(10,1);
    index_int total_points = points_per_proc*the_env.nprocs();
    coordinate<index_int,1> omega( total_points );
    omp_decomposition<1> procs( the_env );
    omp_distribution<1> dist( omega,procs );
    INFO( "original domain: " << dist.global_domain().as_string() );

    ioperator<index_int,1> div2("/2");
    REQUIRE_NOTHROW( dist.operate( div2 ) );
    auto new_dist = dist.operate( div2 );
    REQUIRE_NOTHROW( new_dist.global_domain() );

    REQUIRE_NOTHROW( new_dist.global_domain() );
    auto new_global = new_dist.global_domain();
    INFO( "divided global : " << new_global.as_string() );
    REQUIRE( new_global.volume()==total_points/2 );

    // INFO( "original local : " << dist.local_domain().as_string() );
    // REQUIRE_NOTHROW( new_dist.local_domain() );
    // auto new_local = new_dist.local_domain();
    // INFO( "divided local  : " << new_local.as_string() );
    // REQUIRE( new_local.volume()==total_points/2 );
  }
  {
    INFO( "2D" );
    omp_decomposition<2> procs( the_env );
    INFO( "Decomposition: " << the_env.as_string() );

    coordinate<index_int,2> omega( procs.domain_layout()*16 /* total_points */ );
    index_int total_points = omega.span();
    omp_distribution<2> dist( omega,procs );
    INFO( "original domain: " << dist.global_domain().as_string() );

    ioperator<index_int,2> div2("/2");
    REQUIRE_NOTHROW( dist.operate( div2 ) );
    auto new_dist = dist.operate( div2 );
    REQUIRE_NOTHROW( new_dist.global_domain() );

    REQUIRE_NOTHROW( new_dist.global_domain() );
    auto new_global = new_dist.global_domain();
    INFO( "divided global : " << new_global.as_string() );
    REQUIRE( new_global.volume()==total_points/4 );

    // INFO( "original local : " << dist.local_domain().as_string() );
    // REQUIRE_NOTHROW( new_dist.local_domain() );
    // auto new_local = new_dist.local_domain();
    // INFO( "divided local  : " << new_local.as_string() );
    // index_int check_total_points = the_env.allreduce_ii( new_local.volume() );
    // REQUIRE( check_total_points==total_points/4 );
  }
}

TEST_CASE( "NUMA addressing" ) {
  {
    INFO( "1D" );
    const int points_per_proc = ipower(10,1);
    index_int total_points = points_per_proc*the_env.nprocs();
    coordinate<index_int,1> omega( total_points );
    omp_decomposition<1> procs( the_env );
    omp_distribution<1> omega_p( omega,procs );

    for ( int p=0; p<the_env.nprocs(); p++ ) {
      auto pcoord = procs.coordinate_from_linear(p);
      REQUIRE_NOTHROW( omega_p.location_of_first_index(pcoord) );
      REQUIRE( omega_p.location_of_first_index(pcoord)==p*points_per_proc );
    }
  }
}

