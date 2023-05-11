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
#include <memory>
#include <vector>

template<int d>
class kernel {
private:
  std::shared_ptr<object<d>> output;
  std::vector<std::shared_ptr<object<d>>> inputs;
public:
  kernel( std::shared_ptr<object<d>> out );
  /*
   * specific types
   * THIS IS UGLY because hard to extend
   */
  void setconstant( double v );
};
