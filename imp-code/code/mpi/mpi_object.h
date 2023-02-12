// -*- c++ -*-
/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2023
 ****
 **** mpi_object.h: Header file for the MPI derived class
 ****
 ****************************************************************/

#pragma once

#include "imp_object.h"
#include "mpi_distribution.h"

template<int d>
class mpi_object : public object<d> {
public:
  mpi_object( const mpi_distribution<d>& );
};

