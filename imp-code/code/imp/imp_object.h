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
  std::vector<double>& data();
  const std::vector<double>& data() const;
  double* raw_data();
  double const * raw_data() const;

  void set_constant( double x );
  object<d>& operator+=( const object<d>& );
  virtual double local_norm() const;
  virtual double local_inner_product( const object<d>& ) const;
};

template<int d>
void norm( object<d>& scalar,const object<d>& thing,const environment& env );
template<int d>
void inner_product
    ( object<d>& scalar,const object<d>& thing,const object<d>& other,const environment& env );
