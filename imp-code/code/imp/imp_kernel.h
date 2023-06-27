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
#include <functional>
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
  const auto& get_input() const { return input_object; };
  void analyze();
  const std::vector<task_dependency<d>>& get_dependencies() const;
};

/*!
 * A kernel is a distributed operation
 * taking multiple inputs and giving one output object.
 *
 * The input objects are declared as dependencies.
 */
template<int d>
class kernel {
private:
  std::shared_ptr<object<d>> outvector;
public:
  kernel( std::shared_ptr<object<d>> out );
  
  /*
   * Dependencies
   */
private:
  std::vector< dependency<d> > dependencies;
public:
  void add_dependency( std::shared_ptr<object<d>> input,ioperator<index_int,d> op );
  void analyze();
  const std::vector<task_dependency<d>>& get_dependencies(int id=0) const;

  /*
   * Execution stuff
   */
protected:
  std::function< kernel_function_proto(d) > localexecutefn;
  void *localexecutectx{nullptr};
public:
  void set_localexecutefn( std::function< kernel_function_proto(d) > f );
  void execute();
};

template<int d>
kernel<d> constant_object( std::shared_ptr<object<d>> out, double v );
template<int d>
kernel<d> copy_object( std::shared_ptr<object<d>> out, std::shared_ptr<object<d>> in );
