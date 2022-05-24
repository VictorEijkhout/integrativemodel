/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2020
 ****
 **** Statically defined variables for an MPI-based run
 ****
 ****************************************************************/

#ifndef MPI_STATIC_VARS_H
#define MPI_STATIC_VARS_H

#include <mpi.h>
#include "mpi_base.h"

#ifndef EXTERN
#ifdef mpi_STATIC_VARS_HERE
#define EXTERN
#else
#define EXTERN extern
#endif
#endif

EXTERN int mytid,ntids;
EXTERN processor_coordinate mycoord;
EXTERN domain_coordinate mycoord_coord;
EXTERN MPI_Comm comm;
EXTERN mpi_decomposition decomp;
EXTERN mpi_environment env; 
EXTERN architecture arch;

#endif
