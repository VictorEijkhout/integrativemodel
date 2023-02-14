// -*- c++ -*-
/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2023
 ****
 **** omp_object.h: Header file for the OMP derived class
 ****
 ****************************************************************/

#pragma once

#include "imp_object.h"
#include "omp_distribution.h"

template<int d>
class omp_object : public object<d> {
public:
  omp_object( const omp_distribution<d>& );
};

