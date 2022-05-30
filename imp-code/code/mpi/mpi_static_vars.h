/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2022
 ****
 **** Statically defined variables for an MPI-based run
 ****
 ****************************************************************/

#ifndef MPI_STATIC_VARS_H
#define MPI_STATIC_VARS_H

#include <mpi.h>
// #include "mpi_base.h"

/*
 * Variables are going to be defined
 * statically in unittest_main;
 * everywhere else they will be `extern'
 */
#ifndef EXTERN
#ifdef mpi_STATIC_VARS_HERE
#define EXTERN
#else
#define EXTERN extern
#endif
#endif

/* EXTERN int mytid,ntids; */
/* EXTERN processor_coordinate mycoord; */
/* EXTERN domain_coordinate mycoord_coord; */
/* EXTERN MPI_Comm comm; */
//EXTERN mpi_decomposition decomp;
// EXTERN mpi_environment env; 
// EXTERN architecture arch;

#endif
