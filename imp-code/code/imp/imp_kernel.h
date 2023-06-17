// -*- c++ -*-
/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2023
 ****
 **** imp_kernel.h: Header file for the kernel base classes
 ****
 ****************************************************************/

#pragma once

#include "imp_coord.h" // for the ioperator
#include "imp_object.h"
#include "indexstruct.hpp"
#include "imp_functions.h"
#include <memory>
#include <optional>
#include <tuple>
#include <vector>

template<int d>
struct task_dependency {
  int o_num;
  int p_num;
  indexstructure<index_int,d> data;
};

template<int d>
class dependency {
private:
  std::shared_ptr<object<d>> input_object;
  ioperator<index_int,d> op;
  //! beta distribution for non-origin kernels
  std::optional< distribution<d> > beta{};
  std::optional< std::vector<task_dependency<d>> >depends{};
public:
  dependency( std::shared_ptr<object<d>> input_object,ioperator<index_int,d> op )
    : input_object(input_object),op(op) {};
  void analyze();
  const std::vector<task_dependency<d>>& get_dependencies() const;
};

template<int d>
class kernel {
private:
  std::shared_ptr<object<d>> output;
public:
  kernel( std::shared_ptr<object<d>> out );
  
  /*
   * Dependencies
   */
private:
  std::vector< dependency<d> > inputs;
public:
  void add_dependency( std::shared_ptr<object<d>> input,ioperator<index_int,d> op );
  void analyze();
  const std::vector<task_dependency<d>>& get_dependencies(int id=0) const;
  /*
   * local function
   */
protected:
  std::function< kernel_function_proto(d) > localexecutefn;
  void *localexecutectx{nullptr};
public:
  void set_localexecutefn( std::function< kernel_function_proto(d) > f );

  /*
   * specific types
   * THIS IS UGLY because hard to extend
   */
public:
  void setconstant( double v );
};
