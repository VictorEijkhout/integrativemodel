/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2023
 ****
 **** Unit tests for the MPI product backend of IMP
 **** based on the CATCH framework (https://github.com/philsquared/Catch)
 ****
 **** unit tests for MPI architecture: coordinate tests
 ****
 ****************************************************************/

#include <stdlib.h>
#include <math.h>

#include "catch2/catch_all.hpp"

#include "mpi_env.h"
#include "imp_coord.h"

using fmt::format, fmt::print;
using std::vector;

TEST_CASE( "end points","[arch][01]" ) {

  {
    INFO( "p3 d1" );
    auto p3d1 = endpoint<int,1>(3);
    REQUIRE( p3d1.at(0)==3 );
  }
  {
    INFO( "p4 d2" );
    auto p4d2 = endpoint<int,2>(4);
    REQUIRE( p4d2.at(0)==2 );
    REQUIRE( p4d2.at(1)==2 );
  }
  {
    INFO( "p30 d3" );
    auto p30d3 = endpoint<int,3>(30);
    REQUIRE( p30d3.at(0)==5 );
    REQUIRE( p30d3.at(1)==3 );
    REQUIRE( p30d3.at(2)==2 );
  }
}

TEST_CASE( "split points" ) {
  vector<int> points;
  REQUIRE_NOTHROW( points = split_points( 100,2 ) );
  REQUIRE( points[0]==0 );
  REQUIRE( points[1]==50 );
  REQUIRE( points[2]==100 );
}

TEST_CASE( "coordinates" ) {
  {
    auto ci1 = coordinate<int,1>(0);
    REQUIRE( ci1.at(0)==0 );
  }
  {
    int top=5;
    auto ci1_5 = coordinate<int,1>(top);
    REQUIRE( ci1_5.span()==top );
    coordinate<int,1> last(top-1);
    REQUIRE( ci1_5.linear_location_of(last)==top-1 );
    coordinate<int,1> again;
    REQUIRE_NOTHROW( again = ci1_5.location_of_linear(top-1) );
    INFO( "reinterpret as coord: " << again );
    REQUIRE( again.span()==top-1 );
    for ( int i=0; i<top; i++)
      REQUIRE_NOTHROW( ci1_5.location_of_linear(i) );
  }
  {
    REQUIRE_NOTHROW( coordinate<int,2>(0) );
    auto ci2_0 = coordinate<int,2>(0);
    REQUIRE( ci2_0.at(0)==0 );
    REQUIRE( ci2_0.at(1)==0 );
  }
  {
    const int n1=7,n2=2, n = n1*n2;
    auto fp = endpoint<int,2>(n);
    INFO( "endpoint 14: " << fp[0] << "," << fp[1] );
    auto ci2_14 = coordinate<int,2>(fp);
    auto top = ci2_14.span();
    REQUIRE( ci2_14.span()==n );
    REQUIRE( ci2_14.at(0)==n1 );
    REQUIRE( ci2_14.at(1)==n2 );
    // 7x2 : 3,0 = 6, 2,1 = 5
    coordinate<int,2> c5,c6;
    REQUIRE_NOTHROW( c5 = ci2_14.location_of_linear(5) );
    INFO( "5 in 14:" << c5 );
    REQUIRE( ci2_14.linear_location_of(c5)==5 );
    REQUIRE_NOTHROW( c6 = ci2_14.location_of_linear(6) );
    INFO( "6 in 14:" << c6 );
    REQUIRE( ci2_14.linear_location_of(c6)==6 );
    for ( int i=0; i<top; i++ )
      REQUIRE_NOTHROW( ci2_14.location_of_linear(i) );
  }
  // {
  //   auto c2_61 = coordinate<int,2>( {6,1} );
  //   REQUIRE( c2_61.linear(ci2_14)==ci2_14.span()-1 );
  // }

  // auto ci2_62 = coordinate<int,2>( {6,2} );
  // REQUIRE( ci2_62.before( ci2_14 ) );
  // auto ci2_63 = coordinate<int,2>( {6,3} );
  // REQUIRE( ci2_63.before(ci2_14) );
}

TEST_CASE( "MPI coordinates" ) {
  auto &mpi_env = mpi_environment::instance();
  int np; MPI_Comm_size(MPI_COMM_WORLD,&np);

  SECTION( "using proc counts" ) {
    auto ep1 = endpoint<int,1>( mpi_env.nprocs() );
    auto mpi_grid1 = coordinate<int,1>(ep1);
    REQUIRE( mpi_grid1.span()==np );

    auto ep2 = endpoint<int,2>( mpi_env.nprocs() );
    auto mpi_grid2 = coordinate<int,2>(ep2);
    REQUIRE( mpi_grid2.span()==np );
  }
  SECTION( "using env itself" ) {
    auto ep1 = endpoint<int,1>( mpi_env.nprocs() );
    auto mpi_grid1 = coordinate<int,1>( ep1 );
    REQUIRE( mpi_grid1.span()==np );

    auto ep2 = endpoint<int,2>( mpi_env.nprocs() );
    auto mpi_grid2 = coordinate<int,2>( ep2 );
    REQUIRE( mpi_grid2.span()==np );
  }
}

