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
 **** unit tests for mpi-based kernels
 ****
 ****************************************************************/

#include <stdlib.h>
#include <math.h>

#include "catch2/catch_all.hpp"

#include "mpi_kernel.h"

using fmt::format, fmt::print;

using std::make_shared, std::shared_ptr;
using std::string;
using std::vector;

auto &the_env = mpi_environment::instance();

TEST_CASE( "creation","[1]" ) {
  // make an object
  const int points_per_proc = ipower(5,1);
  index_int total_points = points_per_proc*the_env.nprocs();
  coordinate<index_int,1> omega( total_points );
  mpi_decomposition<1> procs( the_env );
  mpi_distribution<1> dist( omega,procs );
  auto x = shared_ptr<object<1>>( make_shared<mpi_object<1>>( dist ) );

  REQUIRE_NOTHROW( mpi_kernel(x) );
}

TEST_CASE( "constant","[2]" ) {
  // make an object
  const int points_per_proc = ipower(5,1);
  index_int total_points = points_per_proc*the_env.nprocs();
  coordinate<index_int,1> omega( total_points );
  mpi_decomposition<1> procs( the_env );
  mpi_distribution<1> dist( omega,procs );
  auto x = shared_ptr<object<1>>( make_shared<mpi_object<1>>( dist ) );

  mpi_kernel c(x);
  REQUIRE_NOTHROW( c.setconstant(5.2) );
}

TEST_CASE( "shift right","[3]" ) {
  {
    INFO( "1D" );
    const int d=1;
    mpi_decomposition<d> procs( the_env );
    coordinate<index_int,1> omega( procs.process_grid()*16 );
    index_int total_points = omega.span();
    domain<d> dom(omega);
    mpi_distribution<d> dist( omega,procs );
    // input and output object on the same distribution
    auto out = shared_ptr<object<d>>( make_shared<mpi_object<d>>( dist ) );
    auto in  = shared_ptr<object<d>>( make_shared<mpi_object<d>>( dist ) );

    // create dependency on operated object
    ioperator<index_int,1> right1(">>1");
    mpi_kernel shift_left(out);
    REQUIRE_NOTHROW( shift_left.add_dependency(in,right1) );
    vector<task_dependency<d>> depends;
    REQUIRE_NOTHROW( shift_left.analyze() );
    REQUIRE_NOTHROW( depends = shift_left.get_dependencies() );
    if (the_env.procid()==the_env.nprocs()-1) {
      // last process has no right neighor,
      // so only process itself
      REQUIRE( depends.size()==1 );
    } else {
      // process itself and right neighbor
      REQUIRE( depends.size()==2 );
    }
  }
}
