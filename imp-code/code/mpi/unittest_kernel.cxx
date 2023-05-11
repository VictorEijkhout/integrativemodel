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

TEST_CASE( "creation" ) {
  // make an object
  const int points_per_proc = ipower(5,1);
  index_int total_points = points_per_proc*the_env.nprocs();
  coordinate<index_int,1> omega( total_points );
  mpi_decomposition<1> procs( the_env );
  mpi_distribution<1> dist( omega,procs );
  auto x = shared_ptr<object<1>>( make_shared<mpi_object<1>>( dist ) );

  REQUIRE_NOTHROW( mpi_kernel(x) );
}
