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

#include "imp_decomp.h"
#include "imp_distribution.h"

template<int d>
class mpi_distribution : public distribution<d> {
public:
  mpi_distribution( const mpi_decomposition<d>&,const coordinate<index_int,d>& );
public:
  const indexstructure<index_int,d>& local_domain() const;
protected:
  indexstructure<index_int,d> _local_domain;
};

