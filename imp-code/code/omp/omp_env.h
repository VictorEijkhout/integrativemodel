// -*- c++ -*-
/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2023
 ****
 **** omp_env.h: Header file for OMP environment handling
 ****
 ****************************************************************/

#pragma once
#include "imp_env.h"
#include <omp.h>

class omp_environment : public environment {
public:
  /*! Static method for getting the singleton instance */
  static omp_environment& instance() {
    static omp_environment the_instance;
    return the_instance;
  }
private:
  omp_environment();
public:
  omp_environment(omp_environment const&) = delete;
  void operator=(omp_environment const&)  = delete;
  void init( int &argc,char **&argv );
  ~omp_environment();

  /* void print_options(); */
  /* void tasks_to_dot_file(); */
  /* void print_all( std::string s); */
};

