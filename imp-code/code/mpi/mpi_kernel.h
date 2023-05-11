// -*- c++ -*-
/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2023
 ****
 **** mpi_kernel.h: Header file for the MPI kernel derived class
 ****
 ****************************************************************/

#pragma once

#include "imp_kernel.h"
#include "mpi_object.h"

template<int d>
class mpi_kernel : public kernel<d> {
public:
  mpi_kernel( std::shared_ptr<object<d>> );
};

