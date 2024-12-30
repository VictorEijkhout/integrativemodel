/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2024
 ****
 **** Unit tests for the SEQ product backend of IMP
 ****
 ****************************************************************/

#include "catch2/catch_all.hpp"

#include "seq_env.h"

using fmt::format;
using fmt::print;

TEST_CASE( "basic environment stuff","[env][01]" ) {
  auto &the_env = seq_environment::instance();
  print("Confirm that we are running with {} procs\n",
	the_env.nprocs());
  REQUIRE( the_env.nprocs()==1 );
  REQUIRE_THROWS( the_env.procid() );
}
