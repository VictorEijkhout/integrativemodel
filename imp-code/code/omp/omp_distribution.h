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
  omp_distribution( const domain<d>&, const omp_decomposition<d>&,
		    distribution_type=distribution_type::orthogonal );
  omp_distribution<d> operate( const ioperator<index_int,d>& ) const;
};

template<int d>
omp_distribution<d> replicated_scalar_distribution( const omp_decomposition<d>& );
