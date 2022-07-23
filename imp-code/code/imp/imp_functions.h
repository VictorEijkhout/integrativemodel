/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-7
 ****
 **** imp_functions.cxx : header file for imp support functions
 ****
 ****************************************************************/

// #ifndef IMP_FUNCTIONS_H
// #define IMP_FUNCTIONS_H
#pragma one

//snippet numa123index
#define INDEX1D( i,offsets,nsize ) \
  i-offsets[0]
#define INDEX1Dk( i,offsets,nsize,k )		\
  (k)*(i-offsets[0])
#define INDEX2D( i,j,offsets,nsize ) \
  (i-offsets[0])*nsize[1]+j-offsets[1]
#define INDEX3D( i,j,k,offsets,nsize ) \
  ( (i-offsets[0])*nsize[1]+j-offsets[1] )*nsize[2] + k-offsets[2]
#define COORD1D( i,gsize ) \
  ( i )
#define COORD2D( i,j,gsize ) \
  ( (i)*gsize[1] + j )
#define COORD3D( i,j,k,gsize ) \
  ( ( (i)*gsize[1] + j )*gsize[2] + k )
//snippet end

#include <memory>
#include <vector>

#include "utils.h"

template<typename I,int d>
class coordinate;
template<typename I,int d>
class object;
template<typename I,int d>
class sparse_matrix;

template<typename I,int d>
index_int INDEXanyD(coordinate<I,d> &i,coordinate<I,d> &off,coordinate<I,d> &siz);

#define kernel_function_proto void(int,coordinate<int,d>&,std::vector<std::shared_ptr<object>>&,std::shared_ptr<object>,double*)
#define kernel_function_args int step,coordinate<int,d> &p,std::vector<std::shared_ptr<object>> &invectors,std::shared_ptr<object> outvector,double *flopcount
#define kernel_function_types int,coordinate<int,d>&,std::vector<std::shared_ptr<object>>&,std::shared_ptr<object>,double*

#define kernel_function_call step,p,invectors,outvector,flopcount

template<typename I,int d>
typedef void(*kernel_function)(int,coordinate<I,d>&,std::vector<std::shared_ptr<object>>&,std::shared_ptr<object>,double*);

typedef struct {std::string op; std::shared_ptr<object> obj; } char_object_struct;
typedef struct {double *s1; double *s2; std::shared_ptr<object>obj; } doubledouble_object_struct;
typedef struct {char c1; std::shared_ptr<object> s1;
  char c2; std::shared_ptr<object>s2;
  std::shared_ptr<object> obj; } charcharxyz_object_struct;

template<typename I,int d>
void vecnoset( kernel_function_types );
template<typename I,int d>
void vecsetlinear( kernel_function_types );
template<typename I,int d>
template<typename I,int d>
void vecsetlinear2d( kernel_function_types );
template<typename I,int d>
void vecdelta( kernel_function_types, coordinate<I,d>&);
template<typename I,int d>
void vecsetconstant( kernel_function_types, double);
template<typename I,int d>
void vecsetconstantzero( kernel_function_types );
template<typename I,int d>
void vecsetconstantone( kernel_function_types );
template<typename I,int d>
void vecsetconstantp( kernel_function_types );
template<typename I,int d>
void veccopy( kernel_function_types );
template<typename I,int d>
void crudecopy( kernel_function_types );

template<typename I,int d>
void vecscaleby( kernel_function_types );
template<typename I,int d>
void vecscalebytwo( kernel_function_types );
template<typename I,int d>
void vecscalebyc( kernel_function_types,double );
template<typename I,int d>
void vecscaledownby( kernel_function_types );
template<typename I,int d>
void vecscaledownbyc( kernel_function_types,double );

template<typename I,int d>
void vectorsum( kernel_function_types );
template<typename I,int d>
void vectorroot( kernel_function_types );
template<typename I,int d>
void vecaxbyz( kernel_function_types,void* );
template<typename I,int d>
void summing( kernel_function_types );
template<typename I,int d>
void rootofsumming( kernel_function_types );
template<typename I,int d>
void local_inner_product( kernel_function_types );
template<typename I,int d>
void local_norm( kernel_function_types );
template<typename I,int d>
void local_normsquared( kernel_function_types );
template<typename I,int d>
void local_sparse_matrix_vector_multiply( kernel_function_types,std::shared_ptr<sparse_matrix> );
template<typename I,int d>
void sparse_matrix_multiply
    (coordinate<int,d>&,std::shared_ptr<object>,std::shared_ptr<object>,double*);

template<typename I,int d>
void char_scalar_op( kernel_function_args,void* );

template<typename I,int d>
void print_trace_message( kernel_function_types,void * );

// nbody stuff
template<typename I,int d>
class indexstruct;
template<typename I,int d>
std::shared_ptr<indexstruct> doubleinterval(index_int i);
template<typename I,int d>
std::shared_ptr<indexstruct> halfinterval(index_int i);
template<typename I,int d>
void scansum( kernel_function_types );
template<typename I,int d>
void scansumk( kernel_function_types,int );
template<typename I,int d>
void scanexpand( kernel_function_types );

// cg stuff
template<typename I,int d>
void central_difference_damp( kernel_function_types,double );
template<typename I,int d>
void central_difference( kernel_function_types );
template<typename I,int d>
void central_difference_anyd( kernel_function_types );
template<typename I,int d>
void local_diffusion( kernel_function_types );

// #endif
