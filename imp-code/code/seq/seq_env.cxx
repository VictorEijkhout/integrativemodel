/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2024
 ****
 **** seq_env.cxx : seq environment management
 ****
 ****************************************************************/

#include <stdarg.h>
#include <unistd.h> // just for sync
#include <iostream>
using std::cout;

#include "seq_env.h"

using fmt::format, fmt::print;
using std::string;
using std::vector;

using gsl::span;

/*!
  An SEQ environment has all the cseqonents of a base environment
 */
seq_environment::seq_environment()
  : environment() {
};

/*!
  - Initialize OMP
  - set the comm_size/rank functions
  - set the reduction to return the input
  - initialize the singleton instance
*/
void seq_environment::init( int &argc,char **&argv ) {
  nprocs = [] () { return 1; };
  allreduce_ii = [this] (index_int i) -> index_int {
    return i;
  };
  allreduce_d = [this] (double i) -> index_int {
    return i;
  };
  environment::instance().init(argc,argv);
};

//! See also the base destructor for trace output.
seq_environment::~seq_environment() {
  printf("SEQ finalize\n");
};

