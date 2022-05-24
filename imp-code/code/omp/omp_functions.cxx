/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-9
 ****
 **** OpenMP implementations of the support functions
 ****
 ****************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include "math.h"

#include "omp_base.h"
#include "imp_functions.h"

/*!
  Sum two vectors.
  \todo we really need to test equality of distributions
*/
void vectorsum( kernel_function_args )
{
  auto invector1 = invectors.at(0);
  auto invector2 = invectors.at(1);
  auto indata1 = invector1->get_data(p);
  auto indata2 = invector2->get_data(p);
  auto outdata = outvector->get_data(p);

  index_int
    f = outvector->first_index_r(p).coord(0),
    l = outvector->last_index_r(p).coord(0);

  for (index_int i=f; i<=l; i++) {
    outdata.at(i) = indata1.at(i)+indata2.at(i);
  }
  *flopcount += l-f+1;
}

//! \todo we need to figure out how to set a flop count here
void local_sparse_matrix_vector_multiply
    ( kernel_function_args,std::shared_ptr<sparse_matrix> mat ) {
  auto invector = invectors.at(0);
  auto matrix = dynamic_cast<omp_sparse_matrix*>(mat.get());
  if (matrix==nullptr)
    throw(fmt::format("Could not upcast sparse matrix to opm"));
  matrix->multiply(invector,outvector,p);
  return;
};

/*!
  An operation between replicated scalars. The actual operation
  is supplied as a character in the context.
 */
void char_scalar_op( kernel_function_args,void *ctx )
{
  auto invector = invectors.at(0);
  char_object_struct *chobst = (char_object_struct*)ctx;
  //printf("scalar operation <<%s>>\n",chobst->op);

  auto
    arg1 = invector->get_data(p),
    arg2 = chobst->obj->get_data(p),
    out  = outvector->get_data(p);
  if (chobst->op=="/") {
    out.at(0) = arg1.at(0) / arg2.at(0);
  } else
    throw("Unrecognized scalar operation\n");
  *flopcount += 2;
};

/*
 * Nbody stuff
 */
#if 0
void scansum( kernel_function_args )
{
  auto invector = invectors.at(0);
  auto indata = invector->get_data(p);
  int insize = invector->volume(p);

  auto outdata = outvector->get_data(p);
  index_int outsize = outvector->volume(p),
    first = outvector->first_index(p).coord(0);

  if (2*outsize!=insize) {
    printf("scansum: in/out not compatible: %d %d\n",insize,outsize); throw(6);}

  for (index_int i=first; i<+first+outsize; i++) {
    outdata.at(i) = indata.at(2*i)+indata.at(2*i+1);
  }
  *flopcount += outsize;
}
#endif

/*!
  Expand an array by 2. Straight copy and doubling.
*/
void scanexpand( kernel_function_args )
{
  auto invector = invectors.at(0);

  auto
    outdata = outvector->get_data(p), indata = invector->get_data(p);
  index_int
    outsize  = outvector->volume(p),   insize = invector->volume(p),
    outfirst = outvector->first_index_r(p).coord(0), infirst = invector->first_index_r(p).coord(0),
    outlast  = outvector->last_index_r(p).coord(0),   inlast = invector->last_index_r(p).coord(0);

  if (outfirst/2<infirst || outfirst/2>inlast || outlast/2<infirst || outlast/2>outlast)
    throw(fmt::format("[{}] scanexpand: outrange [{}-{}] not sourced from [{}-{}]",
		      p.as_string(),outfirst,outlast,infirst,inlast));

  for (index_int i=0; i<outsize; i++) {
    outdata.at(outfirst+i) = indata.at(infirst+i/2);
  }
  *flopcount += outsize;
}
