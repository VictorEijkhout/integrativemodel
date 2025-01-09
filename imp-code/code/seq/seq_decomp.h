// -*- c++ -*-
/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2025
 ****
 **** seq_decomp.h: Header file for SEQ decompositions
 ****
 ****************************************************************/

#pragma once

#include <optional>

#include "seq_env.h"
#include "imp_decomp.h"

/*!
  An seq decomposition has one domain per processor by default,
  unless there is a global over-decomposition parameters.
*/
template<int d>
class seq_decomposition : public decomposition<d> {
public:
  seq_decomposition( const seq_environment& env );
  seq_decomposition( const coordinate<int,d>& );
  virtual std::string as_string() const override;
protected:
  mutable std::optional<coordinate<int,d>> proc_coord = {};
public:
  int procno() const;
  //  virtual const coordinate<int,d>& this_proc() const override;
};
