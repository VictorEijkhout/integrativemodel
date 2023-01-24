/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2023
 ****
 **** Unit tests for the OpenMP product backend of IMP
 **** based on the CATCH framework (https://github.com/philsquared/Catch)
 ****
 **** unit tests for OpenMP environment
 ****
 ****************************************************************/

#include "catch2/catch_all.hpp"

#include "omp_env.h"

using fmt::format;
using fmt::print;

TEST_CASE( "basic environment stuff","[env][01]" ) {
  auto &the_env = omp_environment::instance();
  print("Confirm that we are running with {} procs\n",
	the_env.nprocs());
}

