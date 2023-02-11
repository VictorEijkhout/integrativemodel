// -*- c++ -*-
/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2023
 ****
 **** omp_distribution.h: Header file for the OMP derived class
 ****
 ****************************************************************/

#pragma once

#include "omp_decomp.h"
#include "imp_distribution.h"

template<int d>
class omp_distribution : public distribution<d> {
public:
  omp_distribution( const coordinate<index_int,d>&, const omp_decomposition<d>& );
};

