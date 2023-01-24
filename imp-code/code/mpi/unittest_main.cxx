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
 **** Main for the MPI backend tests
 ****
 ****************************************************************/

#include <fenv.h> // floating point exceptions

#define CATCH_CONFIG_RUNNER
#include "catch2/catch_all.hpp"
#include "fmt/format.h"

#define STATIC_VARS_HERE
#include "imp_static_vars.h"
#define mpi_VARS_HERE
#include "mpi_static_vars.h"

#include "mpi_env.h"

void unittest_mpi_setup(int argc,char **argv) {

  mpi_environment::instance().init(argc,argv);

  return;
}

int main(int argc,char **argv) {

  unittest_mpi_setup(argc,argv);

  for (int a=0; a<argc; a++)
    if (!strcmp(argv[a],"--imp")) { argc = a; break; }

  int result;
  try {
    result = Catch::Session().run( argc, argv );
  } catch (std::string c) {
    fmt::print("Unittesting aborted: <<{}>>",c);
  } catch (...) {
    fmt::print("Unittesting aborted.");
  }

  return result;
}
