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
class object : public distribution<d> {
private:
  std::vector<double> _data;
public:
  object( const distribution<d>& );
  double* data();
  double const * data() const;
  void set_constant( double x );
  object<d>& operator+=( const object<d>& );
};
