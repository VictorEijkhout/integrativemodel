/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-8
 ****
 **** General template for power method
 ****
 ****************************************************************/

/*! \page power Power method

  We run a number of iterations of the power method 
  on a diagonal and tridiagonal matrix. This exercises
  the sparse matrix vector product and the collective routines.
*/

#include "template_common_header.h"

/****
 **** Main program
 ****/

//! \test We have a test for a power metohd. See \subpage power.
//! \todo this uses mytid, so is only for MPI. extend.
int main(int argc,char **argv) {

  environment::print_application_options =
    [] () {
    printf("  -trace       : print norms\n");
    printf("\n");
  };

  /* The environment does initializations, argument parsing, and customized printf
   */
  IMP_environment *env = new IMP_environment(argc,argv);
  env->set_name("power");
  int trace = env->has_argument("trace");
  
  IMP_architecture *arch = dynamic_cast<IMP_architecture*>(env->get_architecture());
  int mytid = arch->mytid();
  decomposition decomp = IMP_decomposition(arch);
  auto mycoord = decomp.coordinate_from_linear(mytid);
  
  int nlocal = 10000, nsteps = 20;
  //snippet powerobjects
  auto 
    blocked = shared_ptr<distribution>( new IMP_block_distribution(decomp,nlocal,-1) ),
    scalar = shared_ptr<distribution>( new IMP_replicated_distribution(decomp) );

  // create vectors, sharing storage
  vector<shared_ptr<object>> xs(2*nsteps);
  double *data0,*data1;
  xs[0] = shared_ptr<object>( new IMP_object(blocked) );
  xs[1] = shared_ptr<object>( new IMP_object(blocked) );

  data0 = xs[0]->get_data(mycoord);
  // data1 = xs[1]->get_data(mytid);
  for (int i=0; i<nlocal; i++) {
    data0[i] = 1.;
  }
  for (int step=1; step<nsteps; step++) {
    xs[2*step] = shared_ptr<object>( new IMP_object(blocked) ); //,data0);
    xs[2*step+1] = shared_ptr<object>( new IMP_object(blocked) ); //,data1);
  }

  // create lambda values
  vector<shared_ptr<object>> lambdas(nsteps);
  double **lambdavalue = new double*[nsteps];
  for (int step=0; step<nsteps; step++) {
    lambdas[step] = shared_ptr<object>( new IMP_object(scalar) );
    lambdas[step]->allocate();
    lambdavalue[step] = lambdas[step]->get_raw_data();
  }
  //snippet end
  
  int test;
  index_int
    my_first = blocked->first_index_r(mycoord)[0],
    my_last = blocked->last_index_r(mycoord)[0];

  for (test=1; test<=2; test++) {

    // need to recreate the queue and matrix for each test
    auto A = shared_ptr<sparse_matrix>( new IMP_sparse_matrix( blocked ) );
    algorithm queue = IMP_algorithm(decomp);

    fmt::print("Create matrix for test={}\n",test);
    if (test==1) { // diagonal matrix
      for (index_int row=my_first; row<=my_last; row++) {
	A->add_element( row,row,2.0 );
      }
    } else if (test==2) { // threepoint matrix
      index_int globalsize = blocked->global_volume();
      for (int row=my_first; row<=my_last; row++) {
	int col;
	col = row;     A->add_element(row,col,2.);
	col = row+1; if (col<globalsize)
		       A->add_element(row,col,-1.);
	col = row-1; if (col>=0)
		       A->add_element(row,col,-1.);
      }
    }

    //snippet powerqueue
    queue.add_kernel( shared_ptr<kernel>( new IMP_origin_kernel(xs[0]) ) );
    for (int step=0; step<nsteps; step++) {  
      shared_ptr<kernel> matvec, scaletonext,getlambda;
      // matrix-vector product
      matvec = shared_ptr<kernel>( new IMP_spmvp_kernel( xs[2*step],xs[2*step+1],A ) );
      queue.add_kernel(matvec);
      // inner product with previous vector
      getlambda = shared_ptr<kernel>( new IMP_innerproduct_kernel( xs[2*step],xs[2*step+1],lambdas[step] ) );
      if (trace) {
	queue.add_kernel( shared_ptr<kernel>( new IMP_trace_kernel(lambdas[step],fmt::format("Lambda-{}",step)) ) );
      }
      queue.add_kernel(getlambda);
      if (step<nsteps-1) {
	// scale down for the next iteration
        scaletonext = shared_ptr<kernel>( new IMP_scaledown_kernel( lambdavalue[step],xs[2*step+1],xs[2*step+2] ) );
        queue.add_kernel(scaletonext);
      }
    }
    //snippet end

    queue.analyze_dependencies(trace);

    queue.execute();

    // printf("Lambda values (version %d): ",test);
    // for (int step=0; step<nsteps; step++) printf("%e ",lambdavalue[step]);
    // printf("\n");
  }

  delete env;
  return 0;

}

