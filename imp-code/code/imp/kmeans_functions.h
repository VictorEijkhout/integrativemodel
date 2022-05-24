#ifndef KMEANS_KERNEL_H
#define KMEANS_KERNEL_H

#define INDEXpointclusterdist(ipoint,icluster,nlpoint,nlcluster) \
  (ipoint)*(nlcluster)+icluster

/*
  Local kernels. 
  For OMP, the kernel needs to know on what processing element it executes.
  In the MPI case this assignment is unique, so the "p" parameter is a dummy.
*/

// if we have to pass two objects to a kernel
typedef struct { std::shared_ptr<object> one; std::shared_ptr<object> two; } two_object_struct;

// Kernel for initial data generation
void generate_random_coordinates( kernel_function_args );

// Kernels
void distance_to_center( kernel_function_args,void* );
void distance_calculation( kernel_function_args );
void group_calculation( kernel_function_args );
void coordinate_masking( kernel_function_args );
void center_calculation_partial( kernel_function_args );

void masked_reduction_1d( kernel_function_args );

/*
 * Utility stuff
 */
void set_dummy_centers( std::shared_ptr<object> centers );
void set_initial_centers( std::shared_ptr<object> centers,processor_coordinate &p );

void kmeans_gen_local(double *outdata,index_int first,index_int last);
void kmeans_avg_local(double *indata,double *outdata,index_int first,index_int last,double *nops);
void kmeans_vec_sum
    (data_pointer outdata,data_pointer indata, index_int first,index_int last);

#endif
