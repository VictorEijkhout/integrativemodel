/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-9
 ****
 **** mpi_functions.cxx : implementations of the support functions
 ****
 ****************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include "math.h"

#include "mpi_base.h"
#include "imp_functions.h"

//snippet crudecopympi
/*!
  A very crude copy, completely ignoring distributions.
 */
void crudecopy( kernel_function_args,void *ctx )
{
  auto invector = invectors.at(0);
  const auto outdistro = outvector->get_distribution(),
    indistro = invector->get_distribution();

  auto indata = invector->get_data(p);
  auto outdata = outvector->get_data(p);
  
  int ortho = 1;
  if (ctx!=nullptr) ortho = *(int*)ctx;

  index_int n = outdistro->volume(p);

  //  printf("[%d] veccopy of size %ld: %e\n",p,n*ortho,indata[0]);
  for (index_int i=0; i<n*ortho; i++) {
    outdata.at(i) = indata.at(i);
  }
  *flopcount += n*ortho;
}
//snippet end

/*!
  Sum two vectors.
*/
void vectorsum( kernel_function_args )
{
  auto invector = invectors.at(0);
  const auto outdistro = outvector->get_distribution(),
    indistro = invector->get_distribution();

  auto indata1 = invector->get_data(p);
  auto invector2 = invectors.at(1);
  auto indata2 = invector2->get_data(p);
  auto outdata = outvector->get_data(p);

  index_int n = indistro->volume(p);

  for (index_int i=0; i<n; i++) {
    outdata.at(i) = indata1.at(i)+indata2.at(i);
  }
  *flopcount += n;
}

void local_sparse_matrix_vector_multiply
    ( kernel_function_args,std::shared_ptr<sparse_matrix> mat ) {
  auto invector = invectors.at(0);
  const auto outdistro = outvector->get_distribution(),
    indistro = invector->get_distribution();

  auto matrix = dynamic_cast<mpi_sparse_matrix*>(mat.get());
  if (matrix==nullptr)
    throw(fmt::format("Could not upcast sparse matrix to mpi"));
  matrix->multiply(invector,outvector,p);
  return;

};

/*!
  An operation between replicated scalars. The actual operation
  is supplied as a character in the context.
  - '/' : division
 */
void char_scalar_op( kernel_function_args, void *ctx )
{
  auto invector = invectors.at(0);
  const auto outdistro = outvector->get_distribution(),
    indistro = invector->get_distribution();

  char_object_struct *chobst = (char_object_struct*)ctx;
  //printf("scalar operation <<%s>>\n",chobst->op);

  auto arg1 = invector->get_data(p),
    arg2 = chobst->obj->get_data(p),
    out  = outvector->get_data(p);
  if (chobst->op=="/") {
    out.at(0) = arg1.at(0) / arg2.at(0);
  } else
    throw("Unrecognized scalar operation\n");
};

#if 0
/*
 * Nbody stuff
 */
/*!
  Compute the center of mass of an array of particles by comparing two-and-two

  - k=1: add charges
  - k=2: 0=charges added, 1=new center
 */
void scansumk( kernel_function_args,int k )
{
  auto invector = invectors.at(0);
  const auto outdistro = outvector->get_distribution(),
    indistro = invector->get_distribution();

  auto indata = invector->get_data(p);
  auto outdata = outvector->get_data(p);
  int insize = indistro->volume(p);
  int outsize = outdistro->volume(p);

  if (k<1)
    throw(fmt::format("Illegal scansum value k={}",k));
  if (k>2)
    throw(fmt::format("Unimplemented scansum value k={}",k));

  if (2*outsize!=insize) {
    printf("scansum: in/out not compatible: %d %d\n",insize,outsize); throw(6);}

  // fmt::print("scansum on {}: {} elements, sum starts with {}\n",
  // 	     p->coord(0),outsize,outdata[0]);
  double cm;
  //printf("[%d] step %d",p->coord(0),step);
  for (index_int i=0; i<outsize; i++) {
    double v1 = indata.at(2*i*k), v2 = indata.at((2*i+1)*k);
    //printf(", scan %e %e",v1,v2);
    outdata.at(k*i) = v1+v2;
    if (k==2) {
      double x1 = indata.at(2*i*k+1), x2 = indata.at((2*i+1)*k+1);
      outdata.at(i*k+1) = (x1*v2+x2*v1)/(x1+x2);
    }
  } //printf("\n");
  *flopcount += k*outsize;
}

//! Short-cut of \ref scansumk for k=1
void scansum( kernel_function_args ) {
  scansumk(step,p,invectors,outvector,1,flopcount);
}
#endif

void scanexpand( kernel_function_args )
{
  auto invector = invectors.at(0);
  const auto outdistro = outvector->get_distribution(),
    indistro = invector->get_distribution();

  auto outdata = outvector->get_data(p),
    indata = invector->get_data(p);
  index_int
    outsize = outdistro->volume(p),   insize = indistro->volume(p),
    outfirst = outdistro->first_index_r(p).coord(0),
    infirst = indistro->first_index_r(p).coord(0),
    outlast = outdistro->last_index_r(p).coord(0),
    inlast = indistro->last_index_r(p).coord(0);

  if (outfirst/2<infirst || outfirst/2>inlast || outlast/2<infirst || outlast/2>outlast) {
    fmt::memory_buffer w;
    format_to(w,"[{}] scanexpand: outrange [{}-{}] not sourced from [{}-{}]",
	    p.as_string(),outfirst,outlast,infirst,inlast);
    throw(to_string(w).data());
  }

  for (index_int i=0; i<outsize; i++) {
    index_int src = (outfirst+i)/2;
    outdata.at(i) = indata.at(src-infirst);
  }
}

/*
 * CG stuff
 */
// void central_difference
//     (int step,processor_coordinate &p,std::vector<object*> *invectors,auto outvector,
//      double *flopcount)
// {
//   central_difference_damp(step,p,invectors,outvector,flopcount,1.);
// }

// //! \todo turn void* into double
// void central_difference_damp
//     (int step,processor_coordinate &p,std::vector<object*> *invectors,auto outvector,
//      double *flopcount,double damp)
// {
//   auto invector = invectors.at(0);
  // const auto outdistro = outvector->get_distribution(),
  //   indistro = invector->get_distribution();
//   auto outdata = outvector->get_data(p), *indata = invector->get_data(p);

//   // figure out where the current subvector fits in the global numbering
//   index_int
//     tar0 = outdistro->first_index_r(p).coord(0)-outvector->numa_first_index().coord(0),
//     src0 = outdistro->first_index_r(p).coord(0)-invector->numa_first_index().coord(0),
//     len = outdistro->volume(p);
//   index_int 
//     myfirst = outdistro->first_index_r(p).coord(0),
//     mylast = outdistro->last_index(p).coord(0),
//     glast = outvector->global_volume()-1;
  
//   // setting the boundaries is somewhat tricky
//   index_int lo=0,hi=len;
//   if (myfirst==0) { // dirichlet boundary condition
//     outdata.at(tar0) = ( 2*indata.at(src0) - indata.at(src0+1) )*damp;
//     *flopcount += 2;
//     lo++;
//   }
//   if (mylast==glast) {
//     outdata.at(tar0+len-1) = ( 2*indata.at(src0+len-1) - indata.at(src0+len-2) )*damp;
//     *flopcount += 2;
//     hi--;
//   }

//   // ... but then we have a regular three-point stencil
//   for (index_int i=lo; i<hi; i++) {
//     outdata.at(tar0+i) = ( 2*indata.at(src0+i) - indata.at(src0+i-1) - indata.at(src0+i+1) )*damp;
//   }

//   *flopcount += 3*(hi-lo+1);
// }

//! \todo this sorely needs to be made independent
void local_diffusion( kernel_function_args )
{
  auto invector = invectors.at(0);
  const auto outdistro = outvector->get_distribution(),
    indistro = invector->get_distribution();

  auto outdata = outvector->get_data(p);
  auto indata = invector->get_data(p);

  index_int
    tar0 = 0, //outdistro->first_index_r(p)-outvector->global_first_index(),
    src0 = 1, //outdistro->first_index_r(p)-invector->global_first_index(),
    len = outdistro->volume(p);
  return;

  for (index_int i=0; i<len; i++)
    outdata.at(tar0+i) =
      2*indata.at(src0+i) - indata.at(src0+i-1) - indata.at(src0+i+1);
  *flopcount += 3*len;
}

