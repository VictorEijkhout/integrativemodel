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

TEST_CASE( "decomposition iteration" ) {
  {
    INFO( "1D" );
    mpi_decomposition<1> procs( the_env );
    int count{0};
    REQUIRE_NOTHROW( procs.begin() );
    REQUIRE_NOTHROW( procs.end() );
    for ( auto p : procs ) {
      int lindom;
      INFO( format("domain {} = {}",count,p.as_string()) );
      REQUIRE_NOTHROW( lindom = procs.linear_location_of(p) );
      INFO( "is linearly: " << lindom );
      CHECK( lindom==count );
      count++;
    }
    REQUIRE( count==the_env.nprocs() );
  }
  {
    INFO( "2D" );
    mpi_decomposition<2> procs( the_env );
    int count{0};
    REQUIRE_NOTHROW( procs.begin() );
    REQUIRE_NOTHROW( procs.end() );
    for ( auto p : procs ) {
      int lindom;
      INFO( format("domain {} = {}",count,p.as_string()) );
      REQUIRE_NOTHROW( lindom = procs.linear_location_of(p) );
      INFO( "is linearly: " << lindom );
      CHECK( lindom==count );
      count++;
    }
    REQUIRE( count==the_env.nprocs() );
  }
  {
    INFO( "3D" );
    mpi_decomposition<3> procs( the_env );
    int count{0};
    REQUIRE_NOTHROW( procs.begin() );
    REQUIRE_NOTHROW( procs.end() );
    for ( auto p : procs ) {
      int lindom;
      INFO( format("domain {} = {}",count,p.as_string()) );
      REQUIRE_NOTHROW( lindom = procs.linear_location_of(p) );
      INFO( "is linearly: " << lindom );
      CHECK( lindom==count );
      count++;
    }
    REQUIRE( count==the_env.nprocs() );
  }
}

TEST_CASE( "coordinate subdivision" ) {
  SECTION( "2D" ) {
    const int d=2;
    INFO( "2D on " << the_env.nprocs() << " procs" );
    using I = index_int;
    mpi_decomposition<d> decomp(the_env);
    /*
     * Create a coordinate with component in dimension `d'
     * being of size 10 * p_d
     */
    // step 1 : std::array
    const int size_d{10}; array<I,d> coord;
    /* check */ int proc_total{1};
    for ( int id=0; id<d; id++ ) {
      int proc_d;
      REQUIRE_NOTHROW( proc_d = decomp.size_of_dimension(id) );
      coord.at(id) = proc_d * size_d;
      proc_total *= proc_d;
    }
    /* check */ REQUIRE( proc_total==the_env.nprocs() );
    // step 2 : turn std::array into a coordinate
    REQUIRE_NOTHROW( coordinate<I,d>( coord ) );
    coordinate<I,d> domain(coord);
    // step 3 : test coordinate & split points
    for ( int id=0; id<d; id++ ) {
      I domain_d; REQUIRE_NOTHROW( domain_d = domain.at(id) );
      int proc_d; REQUIRE_NOTHROW( proc_d = decomp.size_of_dimension(id) );
      REQUIRE( domain_d==proc_d*size_d );

      vector<I> splits_d;
      REQUIRE_NOTHROW( splits_d = decomp.split_points_d( coord,id ) );
      REQUIRE( splits_d.size()==proc_d+1 );
      REQUIRE( splits_d.at(0)==0 );
      REQUIRE( splits_d.back()==domain_d );
    }
  };
}

