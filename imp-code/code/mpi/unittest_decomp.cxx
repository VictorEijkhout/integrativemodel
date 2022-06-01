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
 **** unit tests for MPI architecture
 ****
 ****************************************************************/

#include <stdlib.h>
#include <math.h>

#include "catch2/catch_all.hpp"

#include "mpi_decomp.h"

using fmt::format;
using fmt::print;

TEST_CASE( "end points","[arch][01]" ) {

  INFO( "p3 d1" );
  auto p3d1 = endpoint<1>(3);
  REQUIRE( p3d1.size()==1 );
  REQUIRE( p3d1.at(0)==3 );

  INFO( "p4 d2" );
  auto p4d2 = endpoint<2>(4);
  REQUIRE( p4d2.size()==2 );
  REQUIRE( p4d2.at(0)==2 );
  REQUIRE( p4d2.at(1)==2 );

  INFO( "p30 d3" );
  auto p30d3 = endpoint<3>(30);
  REQUIRE( p30d3.size()==3 );
  REQUIRE( p30d3.at(0)==5 );
  REQUIRE( p30d3.at(1)==3 );
  REQUIRE( p30d3.at(2)==2 );

}

TEST_CASE( "coordinates" ) {
  auto ci1 = coordinate<int,1>();
  REQUIRE( ci1.size()==1 );
}
