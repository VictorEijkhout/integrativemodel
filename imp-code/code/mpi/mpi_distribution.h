// -*- c++ -*-
/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2020
 ****
 **** mpi_base.h: Header file for the MPI derived class
 ****
 ****************************************************************/

#pragma once

#include "imp_decomp.h"

template<int d>
class mpi_distribution : virtual public distribution {
public:
  mpi_distribution( const mpi_decomposition<d>&,const coordinate<index_int,d>& );
};

