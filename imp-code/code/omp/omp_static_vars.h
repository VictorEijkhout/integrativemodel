/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2023
 ****
 **** Statically defined variables for an OpenMP-based run
 ****
 ****************************************************************/

#ifndef OMP_STATIC_VARS_H
#define OMP_STATIC_VARS_H

#include <omp.h>

#ifndef EXTERN
#ifdef omp_STATIC_VARS_HERE
#define EXTERN
#else
#define EXTERN extern
#endif
#endif

/* EXTERN int ntids; */
/* EXTERN omp_decomposition decomp; */
/* EXTERN omp_environment env;  */
/* EXTERN architecture arch; */

#endif
