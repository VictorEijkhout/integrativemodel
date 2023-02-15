/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2023
 ****
 **** Unit tests for the OpenMP product backend of IMP
 **** based on the CATCH framework (https://github.com/philsquared/Catch)
 ****
 **** unit tests for OpenMP based objects
 ****
 ****************************************************************/

#include <stdlib.h>
#include <math.h>

#include "catch2/catch_all.hpp"

#include "omp_object.h"

using fmt::format, fmt::print;

using std::make_shared, std::shared_ptr;
using std::string;
using std::vector;

auto &the_env = omp_environment::instance();

TEST_CASE( "creation","[omp][object][01]" ) {
  {
    INFO( "1D" );
    // setup as in distribution test
    const int points_per_proc = ipower(5,1);
    index_int total_points = points_per_proc*the_env.nprocs();
    coordinate<index_int,1> omega( total_points );
    omp_decomposition<1> procs( the_env );
    omp_distribution<1> omega_p( omega,procs );

    // create the object
    REQUIRE_NOTHROW( omp_object<1>( omega_p ) );
    omp_object xp( omega_p ), yp( omega_p );
    REQUIRE( xp.compatible_with(yp) );
  }
  {
    INFO( "2D" );
    // setup as in distribution test
    const int points_per_proc = ipower(5,2);
    index_int total_points = points_per_proc*the_env.nprocs();
    coordinate<index_int,2> omega( total_points );
    omp_decomposition<2> procs( the_env );
    omp_distribution<2> omega_p( omega,procs );

    // create the object
    REQUIRE_NOTHROW( omp_object<2>( omega_p ) );
    omp_object xp( omega_p ), yp( omega_p );
    REQUIRE( xp.compatible_with(yp) );
  }
}

TEST_CASE( "local domain","[omp][object][02]" ) {
  {
    INFO( "1D" );
    // setup as in distribution test
    const int points_per_proc = ipower(5,1);
    index_int total_points = points_per_proc*the_env.nprocs();
    coordinate<index_int,1> omega( total_points );
    omp_decomposition<1> procs( the_env );
    omp_distribution<1> omega_p( omega,procs );

    // create the object
    omp_object xp( omega_p ), yp( omega_p );

    // sanity test on local domain
    REQUIRE_NOTHROW( xp.local_domain() );
    auto xp_local = xp.local_domain();
    REQUIRE( xp_local.volume()==total_points );
  }
  {
    INFO( "2D" );
    // setup as in distribution test
    const int points_per_proc = ipower(5,2);
    index_int total_points = points_per_proc*the_env.nprocs();
    coordinate<index_int,2> omega( total_points );
    omp_decomposition<2> procs( the_env );
    omp_distribution omega_p( omega,procs );

    // create the object
    omp_object xp( omega_p ), yp( omega_p );

    // sanity test on local domain, in 2D the distribution need not be equal
    REQUIRE_NOTHROW( xp.local_domain() );
    auto xp_local = xp.local_domain();
    index_int check_total_points;
    REQUIRE_NOTHROW( check_total_points = the_env.allreduce_ii( xp_local.volume() ) );
    REQUIRE( xp_local.volume()==total_points );
  }
}

TEST_CASE( "addition","[omp][object][03]" ) {
  {
    INFO( "1D" );
    // setup as in distribution test
    const int points_per_proc = ipower(5,1);
    index_int total_points = points_per_proc*the_env.nprocs();
    coordinate<index_int,1> omega( total_points );
    omp_decomposition<1> procs( the_env );
    omp_distribution omega_p( omega,procs );
    omp_object xp( omega_p ), yp( omega_p );

    // set and operate
    REQUIRE_NOTHROW( xp.set_constant(2.5) );
    REQUIRE_NOTHROW( yp.set_constant(2.5) );
    REQUIRE_NOTHROW( xp += yp );
    double *xdata;
    REQUIRE_NOTHROW( xdata = xp.data() );
    REQUIRE( xdata[0]==5. );
  }
  {
    INFO( "2D" );
    // setup as in distribution test
    const int points_per_proc = ipower(5,2);
    index_int total_points = points_per_proc*the_env.nprocs();
    coordinate<index_int,2> omega( total_points );
    omp_decomposition<2> procs( the_env );
    omp_distribution omega_p( omega,procs );
    omp_object xp( omega_p ), yp( omega_p );

    // set and operate
    REQUIRE_NOTHROW( xp.set_constant(2.5) );
    REQUIRE_NOTHROW( yp.set_constant(2.5) );
    REQUIRE_NOTHROW( xp += yp );
    double *xdata;
    REQUIRE_NOTHROW( xdata = xp.data() );
    REQUIRE( xdata[0]==5. );
  }
}