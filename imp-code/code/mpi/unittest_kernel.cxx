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
    mpi_decomposition<1> procs( the_env );
    coordinate<index_int,1> omega( procs.domain_layout()*16 );
    index_int total_points = omega.span();
    domain<1> dom(omega);
    mpi_distribution<1> dist( omega,procs );
    // input and output object on the same distribution
    auto out = shared_ptr<object<1>>( make_shared<mpi_object<1>>( dist ) );
    auto in  = shared_ptr<object<1>>( make_shared<mpi_object<1>>( dist ) );

    ioperator<index_int,1> right1(">>1");
    // auto shifted_right = dist.operate( right1 );
    // auto in = shared_ptr<object<1>>( make_shared<mpi_object<1>>( shifted_right ) );
    mpi_kernel shift_left(out);
    REQUIRE_NOTHROW( shift_left.add_dependency(in,right1) );
    REQUIRE_NOTHROW( shift_left.analyze() );
  }
}
