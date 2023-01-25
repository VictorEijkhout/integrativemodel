// -*- c++ -*-
/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2023
 ****
 **** omp_decomp.h: Header file for OMP decompositions
 ****
 ****************************************************************/

#pragma once

#include <optional>

#include "omp.h"
#include "omp_env.h"
#include "imp_decomp.h"

/*!
  An omp decomposition has one domain per processor by default,
  unless there is a global over-decomposition parameters.

  No one ever inherits from this, but mode-specific distributions are built from this
  because it contains a distribution factory.
*/
template<int d>
class omp_decomposition : public decomposition<d> {
public:
  omp_decomposition( const omp_environment& env );
  omp_decomposition( const coordinate<int,d>& );
  void set_decomp_factory();
  virtual std::string as_string() const override;
protected:
  mutable std::optional<coordinate<int,d>> proc_coord = {};
public:
  int procno() const;
  virtual const coordinate<int,d>& this_proc() const override;
};
