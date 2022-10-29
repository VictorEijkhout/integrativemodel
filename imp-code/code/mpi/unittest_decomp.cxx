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
  REQUIRE_NOTHROW( mpi_decomposition<1>( the_env,coordinate<int,1>( array<int,1>({5}) ) ) );
  REQUIRE_NOTHROW( mpi_decomposition<2>( the_env,coordinate<int,2>( {5,6} ) ) );
}

TEST_CASE( "decompositions","[mpi][decomposition][01]" ) {
  //  auto &the_env = mpi_environment::instance();
  auto mytid = the_env.procid();
  INFO( "mytid=" << mytid );
  int over;
  REQUIRE_NOTHROW( over = the_env.get_over_factor() );
  const int d=0;
  mpi_decomposition<d> decomp(the_env);
  vector<coordinate<int,d>> domains;
  REQUIRE_NOTHROW( domains = decomp.get_domains() );
  int count = 0;
  for ( auto dom : domains ) {
    int lindom = dom.at(0);
    INFO( "domain=" << lindom );
    CHECK( lindom==over*mytid+count );
    count++;
  }
}

#if 0
TEST_CASE( "coordinate conversion","[mpi][decomposition][02]" ) {
  decomposition oned;
  REQUIRE_NOTHROW( oned = mpi_decomposition
		   (arch,new coordinate<int,d>( vector<int>{4} ) ) );
  CHECK( oned.get_dimensionality()==1 );
  CHECK( oned.domains_volume()==4 );
  coordinate<int,d> onep;
  REQUIRE_NOTHROW( onep = oned.coordinate_from_linear(1) );
  CHECK( onep==coordinate<int,d>( vector<int>{1} ) );
}

TEST_CASE( "coordinate conversion","[mpi][decomposition][02]" ) {

  decomposition twod;
  REQUIRE_NOTHROW( twod = mpi_decomposition
		   (arch,new coordinate<int,d>( vector<int>{2,4} ) ) );
  CHECK( twod.get_dimensionality()==2 );
  CHECK( twod.domains_volume()==8 );
  coordinate<int,d> twop;
  REQUIRE_NOTHROW( twop = twod.coordinate_from_linear(6) );
  INFO( "6 translates to " << twop.as_string() );
  CHECK( twop==coordinate<int,d>( vector<int>{1,2} ) );

  decomposition threed;
  REQUIRE_NOTHROW( threed = mpi_decomposition
		   (arch,new coordinate<int,d>( vector<int>{6,2,4} ) ) );
  CHECK( threed.get_dimensionality()==3 );
  CHECK( threed.domains_volume()==6*2*4 ); 
  coordinate<int,d> threep; // {2,0,1} -> 2*(2*4) 0*2 + 1 = 17
  REQUIRE_NOTHROW( threep = threed.coordinate_from_linear(17) );
  INFO( "17 translates to " << threep.as_string() );
  CHECK( threep== coordinate<int,d>( vector<int>{2,0,1} ) );

}

#endif
