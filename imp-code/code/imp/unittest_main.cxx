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
 **** Main for the OpenMP backend tests
 ****
 ****************************************************************/

#define CATCH_CONFIG_RUNNER
#include "catch2/catch_all.hpp"

#define STATIC_VARS_HERE
#include "imp_static_vars.h"

// cppformat
#include "fmt/format.h"

int main(int argc,char **argv) {

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
