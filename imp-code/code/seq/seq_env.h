// -*- c++ -*-
/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2024
 ****
 **** omp_env.h: Header file for SEQ environment handling
 ****
 ****************************************************************/

#pragma once
#include "imp_env.h"

class seq_environment : public environment {
public:
  /*! Static method for getting the singleton instance */
  static seq_environment& instance() {
    static seq_environment the_instance;
    return the_instance;
  }
private:
  seq_environment();
public:
  seq_environment(seq_environment const&) = delete;
  void operator=(seq_environment const&)  = delete;
  void init( int &argc,char **&argv );
  ~seq_environment();
};

