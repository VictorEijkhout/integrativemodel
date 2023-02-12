// -*- c++ -*-
/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2023
 ****
 **** imp_object.h: Header file for the object base classes
 ****
 ****************************************************************/

#pragma once

#include "imp_distribution.h"
#include <vector>

template<int d>
class object {
private:
  std::vector<double> data;
public:
  object( const distribution<d>& );
};
