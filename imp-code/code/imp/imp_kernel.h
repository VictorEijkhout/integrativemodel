// -*- c++ -*-
/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2023
 ****
 **** imp_kernel.h: Header file for the kernel base classes
 ****
 ****************************************************************/

#pragma once

#include "imp_object.h"
#include "imp_functions.h"
#include <memory>
#include <vector>

template<int d>
class kernel {
private:
  std::shared_ptr<object<d>> output;
public:
  kernel( std::shared_ptr<object<d>> out );
  
  /*
   * Dependencies
   */
private:
  std::vector<std::shared_ptr<object<d>>> inputs;
public:
  void add_dependency( std::shared_ptr<object<d>> input );
  /*
   * local function
   */
protected:
  std::function< kernel_function_proto(d) > localexecutefn;
  void *localexecutectx{nullptr};
public:
  void set_localexecutefn( std::function< kernel_function_proto(d) > f );

  /*
   * specific types
   * THIS IS UGLY because hard to extend
   */
public:
  void setconstant( double v );
};
