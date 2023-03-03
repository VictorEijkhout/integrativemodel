// -*- c++ -*-
/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2023
 ****
 **** mpi_distribution.h: Header file for the MPI derived class
 ****
 ****************************************************************/

#pragma once

#include "mpi_decomp.h"
#include "imp_distribution.h"

template<int d>
class mpi_distribution : public distribution<d> {
public:
  mpi_distribution( const domain<d>&, const decomposition<d>&,
		    distribution_type=distribution_type::orthogonal );
  // new distribution by operating
  mpi_distribution<d> operate( const ioperator<index_int,d>& ) const;
};

template<int d>
mpi_distribution<d> replicated_scalar_distribution( const mpi_decomposition<d>& );
