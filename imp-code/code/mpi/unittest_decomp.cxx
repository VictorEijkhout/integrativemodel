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

TEST_CASE( "parallel structure" ) {
  /*
   * Simple 1D structure
   */
  const int nprocs = 2;
  const int points_per_proc = 3;
  auto twoprocs   = coordinate<int,1>(nprocs);
  REQUIRE( twoprocs.span()==nprocs );
  auto sixpoints = coordinate<index_int,1>(points_per_proc*nprocs);
  REQUIRE( sixpoints.span()==points_per_proc*nprocs );
  auto parallel   = parallel_structure<1>( twoprocs );
  REQUIRE_NOTHROW( parallel.from_global(sixpoints) );
  index_int first = 0;
  for (int iproc=0; iproc<nprocs; iproc++) {
    INFO( format("proc {}",iproc) );
    decltype( parallel.get_processor_structure(iproc) ) istruct;
    REQUIRE_NOTHROW( istruct = parallel.get_processor_structure(iproc) );
    REQUIRE( istruct->first_index()==first );
    REQUIRE( istruct->volume()==points_per_proc );
    first = istruct->last_index()+1;
  }
  REQUIRE_THROWS( parallel.get_processor_structure(-1) );
  REQUIRE_THROWS( parallel.get_processor_structure(nprocs) );
};

