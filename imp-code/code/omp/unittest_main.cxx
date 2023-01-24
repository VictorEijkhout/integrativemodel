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

#define STATIC_VARS_HERE
#include "imp_static_vars.h"
#define omp_VARS_HERE
#include "omp_static_vars.h"

#include "omp_env.h"

void unittest_omp_setup(int argc,char **argv) {

  omp_environment::instance().init(argc,argv);

  return;
}

int main(int argc,char **argv) {

  unittest_omp_setup(argc,argv);

  for (int a=0; a<argc; a++)
    if (!strcmp(argv[a],"--imp")) { argc = a; break; }

  int result = Catch::Session().run( argc, argv );

  return result;
}
