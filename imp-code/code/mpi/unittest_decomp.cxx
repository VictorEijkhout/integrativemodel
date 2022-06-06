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

#include "mpi_env.h"
#include "mpi_decomp.h"

using fmt::format;
using fmt::print;

TEST_CASE( "end points","[arch][01]" ) {

  INFO( "p3 d1" );
  auto p3d1 = endpoint<int,1>(3);
  REQUIRE( p3d1.size()==1 );
  REQUIRE( p3d1.at(0)==3 );

  INFO( "p4 d2" );
  auto p4d2 = endpoint<int,2>(4);
  REQUIRE( p4d2.size()==2 );
  REQUIRE( p4d2.at(0)==2 );
  REQUIRE( p4d2.at(1)==2 );

  INFO( "p30 d3" );
  auto p30d3 = endpoint<int,3>(30);
  REQUIRE( p30d3.size()==3 );
  REQUIRE( p30d3.at(0)==5 );
  REQUIRE( p30d3.at(1)==3 );
  REQUIRE( p30d3.at(2)==2 );

}

TEST_CASE( "coordinates" ) {
  auto ci1 = coordinate<int,1>();
  REQUIRE( ci1.dimensionality()==1 );

  auto ci1_2 = coordinate<int,1>(2);
  REQUIRE( ci1_2.span()==2 );
  auto ci2_14 = coordinate<int,2>(14);
  REQUIRE( ci2_14.span()==14 );
  REQUIRE( ci2_14.at(0)==7 );
  REQUIRE( ci2_14.at(1)==2 );

  auto c2_61 = coordinate<int,2>( {6,1} );
  REQUIRE( c2_61.linear(ci2_14)==ci2_14.span()-1 );

  // auto ci2_62 = coordinate<int,2>( {6,2} );
  // REQUIRE( ci2_62.before( ci2_14 ) );
  // auto ci2_63 = coordinate<int,2>( {6,3} );
  // REQUIRE( ci2_63.before(ci2_14) );
}

TEST_CASE( "MPI coordinates" ) {
  auto &mpi_env = mpi_environment::instance();
  auto mpi_grid = coordinate<int,2>( mpi_env.nprocs() );
  int np; MPI_Comm_size(MPI_COMM_WORLD,&np);
  REQUIRE( mpi_grid.span()==np );
}

TEST_CASE( "parallel structure" ) {
  const int nprocs = 2;
  auto twoprocs   = coordinate<int,1>(nprocs);
  auto fourpoints = coordinate<index_int,1>(2*nprocs);
  auto parallel   = parallel_structure<1>( twoprocs,fourpoints );
  index_int first = 0;
  for (int iproc=0; iproc<nprocs; iproc++) {
    decltype( parallel.get_processor_structure(iproc) ) istruct;
    REQUIRE_NOTHROW( istruct = parallel.get_processor_structure(iproc) );
    REQUIRE( istruct->first_index()==first );
    first = istruct->last_index()+1;
  }
  REQUIRE_THROWS( parallel.get_processor_structure(-1) );
  REQUIRE_THROWS( parallel.get_processor_structure(nprocs) );
};

