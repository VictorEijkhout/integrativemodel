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

#include <stdlib.h>

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

#if 0
  try {
    env = mpi_environment(argc,argv);
  }
  catch (int x) {
    printf("Could not even get started\n"); throw(1); 
  }

  comm = MPI_COMM_WORLD;
  MPI_Comm_size(comm,&ntids);
  MPI_Comm_rank(comm,&mytid);

  // arch = dynamic_cast<mpi_architecture*>(env->get_architecture());
  arch = env.get_architecture();
  mpi_architecture(arch,ntids,mytid);
  mycoord = processor_coordinate(1);
  mycoord.set(0,mytid); // 1-d by default
  mycoord_coord = domain_coordinate( mycoord.data() );
  //fmt::print("MPI proc {} at coord {}\n",mytid,mycoord->as_string());

  try {
    decomp = mpi_decomposition(arch);
  } catch (std::string c) {
    fmt::print("Error <<{}>> while making top mpi decomposition\n",c);
  } catch (...) {
    fmt::print("Unknown error while making top mpi decomposition\n");
  }
  try {
    int gd = decomp.domains_volume();
  } catch (std::string c) {
    fmt::print("Error <<{}>> in mpi setup\n",c);
    throw(1);
  }
#endif

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
