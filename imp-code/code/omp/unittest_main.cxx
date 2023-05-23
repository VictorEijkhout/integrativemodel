/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-8
 ****
 **** Unit tests for the MPI product backend of IMP
 **** based on the CATCH framework (https://github.com/philsquared/Catch)
 ****
 **** Main for the OpenMP backend tests
 ****
 ****************************************************************/

#include <stdlib.h>

#define CATCH_CONFIG_RUNNER
#include "catch2/catch_all.hpp"
#include "fmt/format.h"
using fmt::print;

#define STATIC_VARS_HERE
#include "imp_static_vars.h"
#define omp_VARS_HERE
#include "omp_static_vars.h"

#include "omp_env.h"

void unittest_omp_setup(int argc,char **argv) {

  try {
    omp_environment::instance().init(argc,argv);
  } catch ( char const* e ) {
    print("Failed to create OpenMP instance aborted with: {}\n",e);
  } catch (...) {
    printf("Failed to create OpenMP instance\n");
  }


  return;
}

int main(int argc,char **argv) {

  try {
    unittest_omp_setup(argc,argv);

    for (int a=0; a<argc; a++)
      if (!strcmp(argv[a],"--imp")) { argc = a; break; }

    int result;
    try {
      print("Catch session started...\n");
      result = Catch::Session().run( argc, argv );
    } catch ( char const* e ) {
      print("Session aborted with: {}\n",e);
    } catch (...) {
      print("Session aborted\n");
    }

    return result;
  } catch (...) { print("Main aborted\n"); };
}

