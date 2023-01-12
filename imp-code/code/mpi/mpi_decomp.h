// -*- c++ -*-
/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2022
 ****
 **** mpi_decomp.h: Header file for MPI decompositions
 ****
 ****************************************************************/

#pragma once

#include "mpi.h"
#include "mpi_env.h"
#include "imp_decomp.h"

/*!
  An mpi decomposition has one domain per processor by default,
  unless there is a global over-decomposition parameters.

  No one ever inherits from this, but mode-specific distributions are built from this
  because it contains a distribution factory.
*/
template<int d>
class mpi_decomposition : public decomposition<d> {
public:
  mpi_decomposition( const mpi_environment& env );
  mpi_decomposition( const coordinate<int,d>&,int );
  void set_decomp_factory();
  virtual std::string as_string() const override;
};

