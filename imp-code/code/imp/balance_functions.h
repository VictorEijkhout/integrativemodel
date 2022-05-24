// -*- c++ -*-
/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-8
 ****
 **** balance_functions.cxx : header file for load balancing support
 ****
 ****************************************************************/

#ifndef BALANCE_FUNCTIONS_H
#define BALANCE_FUNCTIONS_H

#include <memory>

#include "imp_base.h"

#include <Eigen/Dense>
using Eigen::MatrixXd;
using Eigen::VectorXd;

std::shared_ptr<distribution> transform_by_average(std::shared_ptr<distribution>,double*);
std::shared_ptr<distribution> transform_by_diffusion
    (std::shared_ptr<distribution>,std::shared_ptr<object>,MatrixXd,bool=false);
MatrixXd AdjacencyMatrix1D(int p);
MatrixXd AdjacencyMatrix2D(int p);
void report_partition( std::vector<index_int> partition_points, double *times );
void work_moving_weight( kernel_function_args , int globalstep,int laststep );

#endif
