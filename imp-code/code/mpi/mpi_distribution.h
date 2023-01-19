// -*- c++ -*-
/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2023
 ****
 **** mpi_base.h: Header file for the MPI derived class
 ****
 ****************************************************************/

#pragma once

#include "imp_decomp.h"
#include "imp_distribution.h"

template<int d>
class mpi_distribution : public distribution<d> {
public:
  mpi_distribution( const mpi_decomposition<d>&,const coordinate<index_int,d>& );
  indexstructure<index_int,d> local_domain() const;
};

