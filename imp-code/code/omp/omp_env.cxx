/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2023
 ****
 **** omp_env.cxx : omp environment management
 ****
 ****************************************************************/

#include <stdarg.h>
#include <unistd.h> // just for sync
#include <iostream>
using std::cout;

#include "omp_env.h"

using fmt::format, fmt::print;
using std::string;
using std::vector;

using gsl::span;

/*!
  An OMP environment has all the components of a base environment
 */
omp_environment::omp_environment()
  : environment() {
};

/*!
  - Initialize OMP
  - set the comm_size/rank functions
  - initialize the singleton instance
*/
void omp_environment::init( int &argc,char **&argv ) {
  nprocs = [this] () -> int {
    int np;
#pragma omp parallel
#pragma omp master
    np = omp_get_num_threads();
    return np;
  };
  environment::instance().init(argc,argv);
};

//! See also the base destructor for trace output.
omp_environment::~omp_environment() {
  printf("OpenMP finalize\n");
};

