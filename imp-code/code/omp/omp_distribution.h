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

#include "imp_decomp.h"
#include "imp_distribution.h"

template<int d>
class omp_distribution : public distribution<d> {
public:
  omp_distribution( const coordinate<index_int,d>&, const omp_decomposition<d>& );
public:
  const indexstructure<index_int,d>& local_domain() const;
protected:
  indexstructure<index_int,d> _local_domain;
};

