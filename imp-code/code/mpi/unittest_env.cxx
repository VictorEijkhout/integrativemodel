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
 **** unit tests for MPI environment
 ****
 ****************************************************************/

#include "catch2/catch_all.hpp"

#include "mpi_env.h"

using fmt::format;
using fmt::print;

TEST_CASE( "basic environment stuff","[env][01]" ) {
  auto &the_env = mpi_environment::instance();
  print("Confirm that we are running as {} out of {} procs\n",
	the_env.procid(),the_env.nprocs());
}

