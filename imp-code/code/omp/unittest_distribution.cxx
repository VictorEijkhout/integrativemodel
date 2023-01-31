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

#include "omp_env.h"
#include "omp_decomp.h"
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

TEST_CASE( "local domains","[omp][distribution][02]" ) {
  SECTION( "1D" ) {
    const int over = pow(10,1);
    coordinate<index_int,1> omega( over*the_env.nprocs() );
    omp_decomposition<1> procs( the_env );
    omp_distribution<1> omega_p( omega,procs );
    REQUIRE_NOTHROW( omega_p.local_domain() );
    indexstructure<index_int,1> local_domain = omega_p.local_domain();
    REQUIRE( local_domain.is_known() );
    REQUIRE_NOTHROW( local_domain.volume() );
    REQUIRE( local_domain.volume()==over );
  }
  SECTION( "2D" ) {
    const int over = pow(10,2);
    coordinate<index_int,2> omega( over*the_env.nprocs() );
    omp_decomposition<2> procs( the_env );
    omp_distribution<2> omega_p( omega,procs );
    REQUIRE_NOTHROW( omega_p.local_domain() );
    indexstructure<index_int,2> local_domain = omega_p.local_domain();
    REQUIRE( local_domain.volume()==over );
    REQUIRE( local_domain.is_known() );
    REQUIRE_NOTHROW( local_domain.volume() );
  }
}

