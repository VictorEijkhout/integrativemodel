/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2022
 ****
 **** Unit tests for the MPI product backend of IMP
 **** based on the CATCH framework (https://github.com/philsquared/Catch)
 ****
 **** unit tests for MPI decompositions
 ****
 ****************************************************************/

#include <stdlib.h>
#include <math.h>

#include "catch2/catch_all.hpp"

#include "mpi_env.h"
#include "mpi_decomp.h"

using fmt::format, fmt::print;
using std::array, std::vector;

auto &the_env = mpi_environment::instance();

TEST_CASE( "decomposition constructors" ) {
  {
    REQUIRE_NOTHROW( mpi_decomposition<1>( the_env ) );
    mpi_decomposition<1> d( the_env );
    REQUIRE( d.local_volume()==1 );
    REQUIRE( d.global_volume()==the_env.nprocs() );
  }
  {
    REQUIRE_NOTHROW( mpi_decomposition<2>( the_env ) );
    mpi_decomposition<1> d( the_env );
    REQUIRE( d.local_volume()==1 );
    REQUIRE( d.global_volume()==the_env.nprocs() );
  }
}

TEST_CASE( "decompositions","[mpi][decomposition][01]" ) {
  //  auto &the_env = mpi_environment::instance();
  auto mytid = the_env.procid();
  auto ntids = the_env.nprocs();
  INFO( "mytid=" << mytid );
  SECTION( "1D" ) {
    const int d=1;
    INFO( "1D" );
    mpi_decomposition<d> decomp(the_env);
    int count = 0;
    for ( auto dom : decomp ) {
      int lindom;
      INFO( format("domain {} = {}",count,dom.as_string()) );
      REQUIRE_NOTHROW( lindom = decomp.linear_location_of(dom) );
      INFO( "is linearly: " << lindom );
      CHECK( lindom==mytid+count );
      count++;
    }
    REQUIRE( count==1 );
  }
  SECTION( "2D" ) {
    const int d=2;
    INFO( "2D" );
    mpi_decomposition<d> decomp(the_env);
    int count = 0;
    for ( auto dom : decomp ) {
      int lindom = decomp.linear_location_of(dom); // dom.at(0);
      INFO( "domain=" << lindom );
      CHECK( lindom==mytid+count );
      count++;
    }
    REQUIRE( count==1 );
  };
}

