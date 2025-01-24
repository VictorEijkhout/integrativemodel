// -*- c++ -*-
/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2025
 ****
 **** omp_distribution.h: Header file for the OMP derived class
 ****
 ****************************************************************/

#pragma once

#include "omp_decomp.h"
#include "imp_distribution.h"

template<int d>
class omp_distribution : private distribution<d> {
  using base = distribution<d>;
public:
  using base::global_domain;
  using base::operate;
  omp_distribution( const domain<d>&, const decomposition<d>&,
		    distribution_type=distribution_type::orthogonal );
};

template<int d>
omp_distribution<d> replicated_scalar_distribution( const omp_decomposition<d>& );
