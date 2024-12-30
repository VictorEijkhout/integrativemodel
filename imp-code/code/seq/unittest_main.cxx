/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2024
 ****
 **** Main for the SEQ backend tests
 ****
 ****************************************************************/

#include <stdlib.h>

#define CATCH_CONFIG_RUNNER
#include "catch2/catch_all.hpp"
#include "fmt/format.h"
using fmt::print;

#define STATIC_VARS_HERE
#include "imp_static_vars.h"
#define seq_VARS_HERE
#include "seq_static_vars.h"

#include "seq_env.h"

void unittest_seq_setup(int argc,char **argv) {

  try {
    seq_environment::instance().init(argc,argv);
  } catch ( char const* e ) {
    print("Failed to create SEQ instance aborted with: {}\n",e);
  } catch (...) {
    printf("Failed to create SEQ instance\n");
  }


  return;
}

int main(int argc,char **argv) {

  try {
    unittest_seq_setup(argc,argv);

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

