/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2020
 ****
 **** Unit tests for the MPI product backend of IMP
 **** based on the CATCH framework (https://github.com/philsquared/Catch)
 ****
 **** conjugate gradient tests 
 **** (individual kernels are tested in unittest_ops)
 ****
 ****************************************************************/

#include <stdlib.h>
#include <math.h>

#include "catch.hpp"

#include "mpi_base.h"
#include "mpi_ops.h"
#include "mpi_static_vars.h"
#include "unittest_functions.h"

using std::make_shared;
using std::shared_ptr;
using std::string;
using std::vector;

using fmt::format;
using fmt::print;
using fmt::memory_buffer;
using fmt::format_to;
using fmt::to_string;

TEST_CASE( "trace kernels","[kernel][1]" ) {
  shared_ptr<distribution> scalar;
  REQUIRE_NOTHROW( scalar = shared_ptr<distribution>( new mpi_replicated_distribution(decomp) ) );
  shared_ptr<object> rr;
  REQUIRE_NOTHROW( rr = shared_ptr<object>( new mpi_object(scalar) ) );
  REQUIRE_NOTHROW( rr->allocate() );
  data_pointer data;
  REQUIRE_NOTHROW( data = rr->get_data(mycoord) );
  data.at(0) = 3.14;

  mpi_algorithm *queue; REQUIRE_NOTHROW( queue = new mpi_algorithm(decomp) );
  REQUIRE_NOTHROW( queue->add_kernel( shared_ptr<kernel>( new mpi_origin_kernel(rr) ) ) );
  shared_ptr<kernel> trace; REQUIRE_NOTHROW( trace = shared_ptr<kernel>( new mpi_trace_kernel(rr,string("norm")) ) );
  trace->set_name(string("trace-cg-1"));
  REQUIRE_NOTHROW( queue->add_kernel(trace) );
  REQUIRE_NOTHROW( queue->analyze_dependencies() );
  REQUIRE_NOTHROW( queue->execute() );
}

TEST_CASE( "orthogonality relations","[cg][kernel][ortho][2]" ) {

  INFO( "mytid: " << mytid );

  int nlocal = 2;
  shared_ptr<distribution> blocked = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,nlocal,-1) );
  auto
    x = shared_ptr<object>( new mpi_object(blocked) ),
    z = shared_ptr<object>( new mpi_object(blocked) ),
    r = shared_ptr<object>( new mpi_object(blocked) );
  REQUIRE_NOTHROW( x->allocate() );
  REQUIRE_NOTHROW( z->allocate() );
  REQUIRE_NOTHROW( r->allocate() );
  
  shared_ptr<distribution> scalar = shared_ptr<distribution>( new mpi_replicated_distribution(decomp) );
  auto 
    rr  = shared_ptr<object>( new mpi_object(scalar) ),
    zr = shared_ptr<object>( new mpi_object(scalar) ),
    xr = shared_ptr<object>( new mpi_object(scalar) ),
    alpha = shared_ptr<object>( new mpi_object(scalar) ),
    sbzero = shared_ptr<object>( new mpi_object(scalar) ),
    one = shared_ptr<object>( new mpi_object(scalar) );
  double one_value = 1.;
  one->set_value( one_value );
  auto makeone = shared_ptr<kernel>( new mpi_origin_kernel(one) );
  index_int
    my_first = blocked->first_index_r(mycoord).coord(0),
    my_last = blocked->last_index_r(mycoord).coord(0);
  const char *mode;

  {
    // z,r independent: r linear, z constant 1
    data_pointer rdata,zdata;
    REQUIRE_NOTHROW( rdata = r->get_data(mycoord) );
    for (index_int i=0; i<nlocal; i++)
      rdata.at(i) = mytid*nlocal+i+1;
    auto maker = shared_ptr<kernel>( new mpi_origin_kernel(r) );
    REQUIRE_NOTHROW( zdata = z->get_data(mycoord) );
    for (index_int i=0; i<nlocal; i++)
      zdata.at(i) = 1.;
    auto makez = shared_ptr<kernel>( new mpi_origin_kernel(z) );

    // rr = r' r
    shared_ptr<kernel> rr_inprod;
    REQUIRE_NOTHROW( rr_inprod = shared_ptr<kernel>( new mpi_normsquared_kernel( r,rr ) ) );    

    // zr = r' z
    shared_ptr<kernel> zr_inprod;
    REQUIRE_NOTHROW( zr_inprod = shared_ptr<kernel>( new mpi_innerproduct_kernel( r,z,zr ) ) );
    
    // alpha = (r'z)/(r'r)
    shared_ptr<kernel> coef_compute;
    REQUIRE_NOTHROW( coef_compute = shared_ptr<kernel>( new mpi_scalar_kernel( zr,"/",rr,alpha ) ) );
    
    // x = z - alpha r
    shared_ptr<kernel> update;
    REQUIRE_NOTHROW( update = shared_ptr<kernel>( new mpi_axbyz_kernel( '+',one,z,'-',alpha,r, x ) ) );
    
    // answer = x' r = r' ( z - alpha r ) = rz - zr/rr rr
    shared_ptr<kernel> check;
    REQUIRE_NOTHROW( check = shared_ptr<kernel>( new mpi_innerproduct_kernel( x,r,xr ) ) );
    
    SECTION( "by kernels" ) {
      mode = "by kernels";
      REQUIRE_NOTHROW( rr_inprod->analyze_dependencies() );
      REQUIRE_NOTHROW( zr_inprod->analyze_dependencies() );
      REQUIRE_NOTHROW( coef_compute->analyze_dependencies() );
      REQUIRE_NOTHROW( update->analyze_dependencies() );
      REQUIRE_NOTHROW( check->analyze_dependencies() );

      REQUIRE_NOTHROW( rr_inprod->execute() );      
      REQUIRE_NOTHROW( zr_inprod->execute() );      
      REQUIRE_NOTHROW( coef_compute->execute() );
      REQUIRE_NOTHROW( update->execute() );
      REQUIRE_NOTHROW( check->execute() );
    }
    SECTION( "by task queue" ) {
      mode = "by queue";
      algorithm queue = mpi_algorithm(decomp);
      REQUIRE_NOTHROW( queue.add_kernel(makeone) );
      REQUIRE_NOTHROW( queue.add_kernel(maker) );
      REQUIRE_NOTHROW( queue.add_kernel(makez) );
      REQUIRE_NOTHROW( queue.add_kernel(rr_inprod) );
      REQUIRE_NOTHROW( queue.add_kernel(zr_inprod) );
      REQUIRE_NOTHROW( queue.add_kernel(coef_compute) );
      REQUIRE_NOTHROW( queue.add_kernel(update) );
      REQUIRE_NOTHROW( queue.add_kernel(check) );

      REQUIRE_NOTHROW( queue.analyze_dependencies() );
      REQUIRE_NOTHROW( queue.execute() );      
    }

    INFO( "mode: " << mode );

    // check a bunch of intermediate results
    data_pointer quad; int g=blocked->global_volume();
    double zr_value = g*(g+1)/2, rr_value = g*(g+1)*(2*g+1)/6,
      alpha_value = zr_value/rr_value;
    CHECK_NOTHROW( quad = zr->get_data(mycoord) );
    CHECK( quad.at(0)==Approx( zr_value ) );
    CHECK_NOTHROW( quad = rr->get_data(mycoord) );
    CHECK( quad.at(0)==Approx( rr_value ) );
    CHECK_NOTHROW( quad = alpha->get_data(mycoord) );
    CHECK( quad.at(0)==Approx( alpha_value ) );
  }

  // => r' x = 0
  {
    data_pointer isthiszero;
    CHECK( xr->volume(mycoord)==1 );
    REQUIRE_NOTHROW( isthiszero = xr->get_data(mycoord) );
    CHECK( isthiszero.at(0)==Approx(0.) );
  }
}

TEST_CASE( "A-orthogonality relations","[cg][kernel][ortho][sparse][3]" ) {

  INFO( "mytid: " << mytid );

  int nlocal = 2;
  shared_ptr<distribution> blocked = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,nlocal,-1) );
  auto 
    x = shared_ptr<object>( new mpi_object(blocked) ),
    z = shared_ptr<object>( new mpi_object(blocked) ),
    r = shared_ptr<object>( new mpi_object(blocked) );
  REQUIRE_NOTHROW( x->allocate() );
  REQUIRE_NOTHROW( z->allocate() );
  REQUIRE_NOTHROW( r->allocate() );
  
  shared_ptr<distribution> scalar = shared_ptr<distribution>( new mpi_replicated_distribution(decomp) );
  auto 
    rr  = shared_ptr<object>( new mpi_object(scalar) ),
    zr = shared_ptr<object>( new mpi_object(scalar) ),
    xr = shared_ptr<object>( new mpi_object(scalar) ),
    alpha = shared_ptr<object>( new mpi_object(scalar) ), 
    sbzero = shared_ptr<object>( new mpi_object(scalar) ),
    one = shared_ptr<object>( new mpi_object(scalar) );
  double one_value = 1.;
  one->set_value( one_value );
  auto makeone = shared_ptr<kernel>( new mpi_origin_kernel(one) );
  index_int
    my_first = blocked->first_index_r(mycoord).coord(0),
    my_last = blocked->last_index_r(mycoord).coord(0);
  const char *mode;

  shared_ptr<sparse_matrix> A; int test;
  REQUIRE_NOTHROW( A = shared_ptr<sparse_matrix>( new mpi_sparse_matrix( blocked ) ) );

  index_int globalsize = domain_coordinate( blocked->global_size() ).at(0);
  for (int row=my_first; row<=my_last; row++) {
    int col;
    col = row;
    REQUIRE_NOTHROW( A->add_element(row,col,2.) );
    col = row+1;
    if (col<globalsize)
      REQUIRE_NOTHROW( A->add_element(row,col,-1.) );
    col = row-1;
    if (col>=0)
      REQUIRE_NOTHROW( A->add_element(row,col,-1.) );
  }

  {
    // z,r independent: r linear, z constant 1
    data_pointer rdata,zdata;
    REQUIRE_NOTHROW( rdata = r->get_data(mycoord) );
    for (index_int i=0; i<nlocal; i++)
      rdata.at(i) = mytid*nlocal+i+1;
    auto maker = shared_ptr<kernel>( new mpi_origin_kernel(r) );
    REQUIRE_NOTHROW( zdata = z->get_data(mycoord) );
    for (index_int i=0; i<nlocal; i++)
      zdata.at(i) = 1.;
    auto makez = shared_ptr<kernel>( new mpi_spmvp_kernel(r,z,A) );

    // rr = r' r
    shared_ptr<kernel> rr_inprod;
    REQUIRE_NOTHROW( rr_inprod = shared_ptr<kernel>( new mpi_normsquared_kernel( r,rr ) ) );    

    // zr = r' z
    shared_ptr<kernel> zr_inprod;
    REQUIRE_NOTHROW( zr_inprod = shared_ptr<kernel>( new mpi_innerproduct_kernel( r,z,zr ) ) );
    
    // alpha = (r'z)/(r'r)
    shared_ptr<kernel> coef_compute;
    REQUIRE_NOTHROW( coef_compute = shared_ptr<kernel>( new mpi_scalar_kernel( zr,"/",rr,alpha ) ) );
    
    // x = z - alpha r
    shared_ptr<kernel> update;
    REQUIRE_NOTHROW( update = shared_ptr<kernel>( new mpi_axbyz_kernel( '+',one,z,'-',alpha,r, x ) ) );
    
    // answer = x' r = r' ( z - alpha r ) = rz - zr/rr rr
    shared_ptr<kernel> check;
    REQUIRE_NOTHROW( check = shared_ptr<kernel>( new mpi_innerproduct_kernel( x,r,xr ) ) );
    
    algorithm queue = mpi_algorithm(decomp);
    REQUIRE_NOTHROW( queue.add_kernel(makeone) );
    REQUIRE_NOTHROW( queue.add_kernel(maker) );
    REQUIRE_NOTHROW( queue.add_kernel(makez) );
    REQUIRE_NOTHROW( queue.add_kernel(rr_inprod) );
    REQUIRE_NOTHROW( queue.add_kernel(zr_inprod) );
    REQUIRE_NOTHROW( queue.add_kernel(coef_compute) );
    REQUIRE_NOTHROW( queue.add_kernel(update) );
    REQUIRE_NOTHROW( queue.add_kernel(check) );
    
    REQUIRE_NOTHROW( queue.analyze_dependencies() );
    REQUIRE_NOTHROW( queue.execute() );      
  }

  // => r' x = 0
  {
    data_pointer isthiszero;
    CHECK( xr->volume(mycoord)==1 );
    REQUIRE_NOTHROW( isthiszero = xr->get_data(mycoord) );
    CHECK( isthiszero.at(0)==Approx(0.) );
  }
}

TEST_CASE( "power method","[sparse][10]" ) {
  INFO( "mytid=" << mytid );

  int nlocal = 10, nsteps = 4;
  auto 
    blocked = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,nlocal,-1) ),
    scalar = shared_ptr<distribution>( new mpi_replicated_distribution(decomp) );

  shared_ptr<sparse_matrix> A; int test;
  REQUIRE_NOTHROW( A = shared_ptr<sparse_matrix>( new mpi_sparse_matrix( blocked ) ) );
  index_int
    my_first = blocked->first_index_r(mycoord).coord(0),
    my_last = blocked->last_index_r(mycoord).coord(0);

  const char *mat;
  SECTION( "diagonal matrix" ) {
    test = 1; mat = "diagonal";
    for (index_int row=my_first; row<=my_last; row++) {
      REQUIRE_NOTHROW( A->add_element( row,row,2.0 ) );
    }
  }
  SECTION( "threepoint matrix" ) {
    test = 2; mat = "threepoint";
    index_int globalsize = domain_coordinate( blocked->global_size() ).at(0);
    for (int row=my_first; row<=my_last; row++) {
      int col;
      col = row;     A->add_element(row,col,2.);
      col = row+1; if (col<globalsize)
		     A->add_element(row,col,-1.);
      col = row-1; if (col>=0)
		     A->add_element(row,col,-1.);
    }
  }
  INFO( "matrix test: " << mat );

  // create vectors, sharing storage
  auto xs = vector<shared_ptr<object>>(2*nsteps);
  for (int step=0; step<nsteps; step++) {
    REQUIRE_NOTHROW( xs[2*step] = shared_ptr<object>( new mpi_object(blocked) ) );
    xs[2*step]->set_name(format("xs[{}]",2*step));

    REQUIRE_NOTHROW( xs[2*step+1] = shared_ptr<object>( new mpi_object(blocked) ) );
    xs[2*step+1]->set_name(format("xs[{}]",2*step+1));
  }
  data_pointer data0;
  REQUIRE_NOTHROW( xs[0]->allocate() ); // starting vector is allocated, everything else in halo
  REQUIRE_NOTHROW( data0 = xs[0]->get_data(mycoord) );
  for (int i=0; i<nlocal; i++) data0.at(i) = 1.;

  // create lambda values
  auto
    norms = vector<shared_ptr<object>>(2*nsteps),
    lambdas = vector<shared_ptr<object>>(nsteps);
  for (int step=0; step<nsteps; step++) {
    shared_ptr<object> inobj,outobj;

    REQUIRE_NOTHROW( inobj = shared_ptr<object>( new mpi_object(scalar) ) );
    REQUIRE_NOTHROW( inobj->allocate() );
    REQUIRE_NOTHROW( inobj->set_name(format("in-object-{}",step)));
    norms[2*step] = inobj;

    REQUIRE_NOTHROW( outobj = shared_ptr<object>( new mpi_object(scalar) ) );
    REQUIRE_NOTHROW( outobj->allocate() );
    REQUIRE_NOTHROW( outobj->set_name(format("out-object-{}",step)));
    norms[2*step+1] = outobj;

    REQUIRE_NOTHROW( lambdas[step] = shared_ptr<object>( new mpi_object(scalar) ) );
    REQUIRE_NOTHROW( lambdas[step]->allocate() );
  }
  
  algorithm queue = mpi_algorithm(decomp);  
  // originate the first vector
  REQUIRE_NOTHROW( queue.add_kernel( shared_ptr<kernel>( new mpi_origin_kernel(xs[0]) ) ) );
  // loop through the rest
  for (int step=0; step<nsteps; step++) {  
    shared_ptr<kernel> matvec, scaletonext,getnorm,computelambda;
    // matrix-vector product
    REQUIRE_NOTHROW( matvec = shared_ptr<kernel>( new mpi_spmvp_kernel( xs[2*step],xs[2*step+1],A ) ) );
    REQUIRE_NOTHROW( matvec->set_name(format("mvp-{}.",step)) );
    REQUIRE_NOTHROW( queue.add_kernel(matvec) );
    // norms to compare
    REQUIRE_NOTHROW( getnorm = shared_ptr<kernel>( new mpi_norm_kernel( xs[2*step],norms[2*step] ) ) );
    REQUIRE_NOTHROW( queue.add_kernel(getnorm) );
    REQUIRE_NOTHROW( getnorm = shared_ptr<kernel>( new mpi_norm_kernel( xs[2*step+1],norms[2*step+1] ) ) );
    REQUIRE_NOTHROW( queue.add_kernel(getnorm) );
    REQUIRE_NOTHROW( computelambda =
		     shared_ptr<kernel>( new mpi_scalar_kernel( norms[2*step+1],"/",norms[2*step],lambdas[step] ) ) );
    REQUIRE_NOTHROW( queue.add_kernel(computelambda) );
    if (step<nsteps-1) {
      // scale down for the next iteration
      REQUIRE_NOTHROW(scaletonext =
		      shared_ptr<kernel>( new mpi_scaledown_kernel( lambdas[step],xs[2*step+1],xs[2*step+2] ) ) );
      REQUIRE_NOTHROW( queue.add_kernel(scaletonext) );
    }
  }
  REQUIRE_NOTHROW( queue.analyze_dependencies() );
  for (int step=0; step<nsteps; step++) {
    INFO( "step " << step << ", power input is " << xs[2*step]->data_status_as_string()
	  << ", power output is " << xs[2*step+1]->data_status_as_string() );
    if (step==0) // starting vector was explicitly allocated
      CHECK( xs[2*step]->has_data_status_allocated() );
    else
      CHECK( ( !arch->get_can_embed_in_beta() ||  xs[2*step]->has_data_status_inherited() ) );
  }

  REQUIRE_NOTHROW( queue.execute() );
  if (test==1) {
    for (int step=0; step<nsteps; step++) {
      INFO( "step: " << step );
      auto
	indata = xs[2*step]->get_data(mycoord),
	outdata = xs[2*step+1]->get_data(mycoord);
      data_pointer n0,n1,l;
      REQUIRE_NOTHROW( n0 = norms[2*step]->get_data(mycoord) );
      REQUIRE_NOTHROW( n1 = norms[2*step+1]->get_data(mycoord) );
      REQUIRE_NOTHROW( l = lambdas[step]->get_data(mycoord) );
      CHECK( n1.at(0)==Approx(2*n0.at(0)) );
      CHECK( l.at(0)!=Approx(0.) );
    }
  }
}

TEST_CASE( "power method with data reuse","[reuse][11]" ) {
  INFO( "mytid=" << mytid );

  int nlocal = 10, nsteps = 4;
  auto 
    blocked = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,nlocal,-1) ),
    scalar = shared_ptr<distribution>( new mpi_replicated_distribution(decomp) );

  shared_ptr<sparse_matrix> A; int test;
  REQUIRE_NOTHROW( A = shared_ptr<sparse_matrix>( new mpi_sparse_matrix( blocked ) ) );
  index_int
    my_first = blocked->first_index_r(mycoord).coord(0),
    my_last = blocked->last_index_r(mycoord).coord(0);

  // const char *mat;
  // // SECTION( "diagonal matrix" ) {
  // //   test = 1; mat = "diagonal";
  // //   for (index_int row=my_first; row<=my_last; row++) {
  // //     REQUIRE_NOTHROW( A->add_element( row,row,2.0 ) );
  // //   }
  // // }
  // // SECTION( "threepoint matrix" ) {
  // {
  //   test = 2; mat = "threepoint";
  //   index_int globalsize = domain_coordinate( blocked->global_size() ).at(0);
  //   for (int row=my_first; row<=my_last; row++) {
  //     int col;
  //     col = row;     REQUIRE_NOTHROW( A->add_element(row,col,2.) );
  //     col = row+1; if (col<globalsize)
  // 		     REQUIRE_NOTHROW( A->add_element(row,col,-1.) );
  //     col = row-1; if (col>=0)
  // 		     REQUIRE_NOTHROW( A->add_element(row,col,-1.) );
  //   }
  // }
  // INFO( "matrix test: " << mat );

  // create vectors, sharing storage
  shared_ptr<object> xvector,axvector;
  REQUIRE_NOTHROW( xvector = shared_ptr<object>( new mpi_object(blocked) ) );
  xvector->set_name("x0");
  REQUIRE_NOTHROW( axvector = shared_ptr<object>( new mpi_object(blocked) ) );
  axvector->set_name("Ax0");

  data_pointer data0;
  REQUIRE_NOTHROW( xvector->allocate() );
  REQUIRE_NOTHROW( data0 = xvector->get_data(mycoord) );
  for (int i=0; i<nlocal; i++) data0.at(i) = 1.;

  // create lambda values
  auto
    norms = vector<shared_ptr<object>>(2*nsteps),
    lambdas = vector<shared_ptr<object>>(nsteps);
  auto 
    normvalue   = make_shared<vector<double>>( vector<double>(2*nsteps) ),
    lambdavalue = make_shared<vector<double>>( vector<double>(nsteps) );
  for (int step=0; step<nsteps; step++) {
    shared_ptr<object> inobj,outobj; 

    REQUIRE_NOTHROW( inobj = shared_ptr<object>( new mpi_object(scalar) ) );
    { memory_buffer w;
      format_to(w,"in-object-{}",step);
      REQUIRE_NOTHROW( inobj->set_name(to_string(w)) );
    }
    norms[2*step] = inobj;

    REQUIRE_NOTHROW( outobj = shared_ptr<object>( new mpi_object(scalar) ) );
    { memory_buffer w;
      format_to(w,"out-object-{}",step);
      REQUIRE_NOTHROW( outobj->set_name(to_string(w)) );
    }
    norms[2*step+1] = outobj;

    REQUIRE_NOTHROW
      ( lambdas[step] = shared_ptr<object>( new mpi_object(scalar,lambdavalue,step) ) );
  }
  
  algorithm queue = mpi_algorithm(decomp);  
  // originate the first vector
  REQUIRE_NOTHROW( queue.add_kernel( shared_ptr<kernel>( new mpi_origin_kernel(xvector) ) ) );
  // loop through the rest
  for (int step=0; step<nsteps; step++) {  
    shared_ptr<kernel> matvec, scaletonext,getnorm,computelambda;
    if (step>0) {
      REQUIRE_NOTHROW( axvector = shared_ptr<object>( new mpi_object(blocked,axvector) ) );
      xvector->set_name(format("axvector-{}",step));
    }
    // matrix-vector product
    REQUIRE_NOTHROW( matvec = shared_ptr<kernel>( new mpi_diffusion_kernel( xvector,axvector ) ) );
    REQUIRE_NOTHROW( matvec->set_name(format("mvp-{}",step)) );
    REQUIRE_NOTHROW( queue.add_kernel(matvec) );
    // norms to compare
    REQUIRE_NOTHROW( getnorm = shared_ptr<kernel>( new mpi_norm_kernel( xvector,norms[2*step] ) ) );
    REQUIRE_NOTHROW( queue.add_kernel(getnorm) );
    REQUIRE_NOTHROW( getnorm = shared_ptr<kernel>( new mpi_norm_kernel( axvector,norms[2*step+1] ) ) );
    REQUIRE_NOTHROW( queue.add_kernel(getnorm) );
    REQUIRE_NOTHROW( computelambda =
		     shared_ptr<kernel>( new mpi_scalar_kernel( norms[2*step+1],"/",norms[2*step],lambdas[step] ) ) );
    REQUIRE_NOTHROW( queue.add_kernel(computelambda) );
    REQUIRE_NOTHROW( queue.analyze_kernel_dependencies() );
    if (step<nsteps-1) {
      // scale down for the next iteration
      REQUIRE_NOTHROW( xvector = shared_ptr<object>( new mpi_object(blocked,xvector) ) );
      xvector->set_name(format("xvector-{}",step));
      REQUIRE_NOTHROW(scaletonext =
		      shared_ptr<kernel>( new mpi_scaledown_kernel( lambdas[step],axvector,xvector ) ) );
      REQUIRE_NOTHROW( queue.add_kernel(scaletonext) );
    }
  }
  REQUIRE_NOTHROW( queue.analyze_dependencies() );
  REQUIRE_NOTHROW( queue.execute() );
  if (mytid==0) {
    printf("Lambda values (version %d): ",test);
    for (int step=0; step<nsteps; step++) printf("%e ",lambdavalue->at(step));
    printf("\n");
  }
}

TEST_CASE( "diffusion","[15]" ) {

  int nlocal = 10000, nsteps = 1,nglobal = nlocal*arch->nprocs();

  shared_ptr<distribution> blocked = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,nglobal) );
  auto xs = new vector<shared_ptr<object>>;
  for (int step=0; step<=nsteps; step++) {
    xs->push_back( shared_ptr<object>( new mpi_object(blocked) ) );
  }
  // set initial condition to a delta function
  REQUIRE_NOTHROW( xs->at(0)->allocate() );
  auto data = xs->at(0)->get_data(mycoord);
  for (index_int i=0; i<nlocal; i++)
    data.at(i) = 0.0;
  if (mytid==0)
    data.at(0) = 1.;

  algorithm queue = mpi_algorithm(decomp);

  queue.add_kernel( shared_ptr<kernel>( new mpi_origin_kernel(xs->at(0)) ) );
  for (int step=0; step<nsteps; step++) {
    REQUIRE_NOTHROW( queue.add_kernel( shared_ptr<kernel>( new mpi_diffusion_kernel( xs->at(step),xs->at(step+1) ) ) ) );
  }
  REQUIRE_NOTHROW( queue.analyze_dependencies() );
  auto tsks = queue.get_tasks();
  for (auto t : tsks ) {
    if (!t->has_type_origin()) {
      auto msgs = t->get_receive_messages();
      if (mytid==0 || mytid==ntids-1)
	CHECK( msgs.size()==2 );
      else
	CHECK( msgs.size()==3 );
      //for (auto m=msgs->begin(); m!=msgs->end(); ++m) {
      for ( auto m : msgs ) {
	auto snd = m->get_sender();
	if (snd.coord(0)==mytid-1 || snd.coord(0)==mytid+1) {
	  CHECK( m->get_global_struct()->local_size(0)==1 );
	}
      }
    }
  }

  REQUIRE_NOTHROW( queue.execute() );
}

TEST_CASE( "cg algorithm","[sparse][20]" ) {

  int nlocal = 10;

  // a bunch of vectors, block distributed
  auto blocked = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,nlocal,-1) );
  auto 
    x = shared_ptr<object>( new mpi_object(blocked) ), 
    xnew = shared_ptr<object>( new mpi_object(blocked) ),
    z = shared_ptr<object>( new mpi_object(blocked) ),
    r = shared_ptr<object>( new mpi_object(blocked) ),
    rnew = shared_ptr<object>( new mpi_object(blocked) ),
    p = shared_ptr<object>( new mpi_object(blocked) ), 
    pnew = shared_ptr<object>( new mpi_object(blocked) ),
    q = shared_ptr<object>( new mpi_object(blocked) ), 
    qold = shared_ptr<object>( new mpi_object(blocked) );

  r->allocate();

  // scalars, all redundantly replicated
  auto scalar = shared_ptr<distribution>( new mpi_replicated_distribution(decomp) );
  auto 
    rr  = shared_ptr<object>( new mpi_object(scalar) ), 
    rrp = shared_ptr<object>( new mpi_object(scalar) ),
    pap = shared_ptr<object>( new mpi_object(scalar) ),
    alpha = shared_ptr<object>( new mpi_object(scalar) ), 
    beta = shared_ptr<object>( new mpi_object(scalar) );
  int n_iterations=5;
  auto 
    rnorms = vector<shared_ptr<object>>(n_iterations),
    rrzero = vector<shared_ptr<object>>(n_iterations),
    ppzero = vector<shared_ptr<object>>(n_iterations);
  for (int it=0; it<n_iterations; it++) {
    rnorms[it] = shared_ptr<object>( new mpi_object(scalar) );
    rrzero[it] = shared_ptr<object>( new mpi_object(scalar) );
    ppzero[it] = shared_ptr<object>( new mpi_object(scalar) );
  }

  // the sparse matrix
  shared_ptr<sparse_matrix> A;
  { 
    index_int globalsize = domain_coordinate( blocked->global_size() ).at(0);
    int mytid = arch->mytid();
    index_int
      first = blocked->first_index_r(mycoord).coord(0),
      last = blocked->last_index_r(mycoord).coord(0);

    A = shared_ptr<sparse_matrix>( new mpi_sparse_matrix(blocked) );
    REQUIRE_NOTHROW( x->allocate() ); REQUIRE_NOTHROW( r->allocate() );
    auto
      xdata = x->get_data(mycoord),
      rdata = r->get_data(mycoord);
    for (int row=first; row<=last; row++) {
      xdata.at(row-first) = 1.; rdata.at(row-first) = 1.;
      int col;
      col = row;     A->add_element(row,col,2.);
      col = row+1; if (col<globalsize)
    		     A->add_element(row,col,-1.);
      col = row-1; if (col>=0)
    		     A->add_element(row,col,-1.);
    }
  }
  
  auto one_value = make_shared<vector<double>>( vector<double>(1) );
  one_value->at(0) = 1.;
  auto one = shared_ptr<object>( new mpi_object(scalar,one_value ));
  
  // let's define the steps of the loop body
  algorithm queue = mpi_algorithm(decomp);
  queue.add_kernel( shared_ptr<kernel>( new mpi_origin_kernel(one) ) );
  queue.add_kernel( shared_ptr<kernel>( new mpi_origin_kernel(x) ) );
  queue.add_kernel( shared_ptr<kernel>( new mpi_origin_kernel(r) ) );
  queue.add_kernel( shared_ptr<kernel>( new mpi_copy_kernel(r,p) ) );

  SECTION( "one iteration without copy" ) {
    auto precon = shared_ptr<kernel>( new mpi_copy_kernel( r,z ) );
    queue.add_kernel(precon);

    auto rho_inprod = shared_ptr<kernel>( new mpi_innerproduct_kernel( r,z,rr ) );
    rho_inprod->set_name("compute rho");
    queue.add_kernel(rho_inprod);

    auto pisz = shared_ptr<kernel>( new mpi_copy_kernel( z,pnew ) );
    pisz->set_name("copy z to p");
    queue.add_kernel(pisz);

    auto matvec = shared_ptr<kernel>( new mpi_spmvp_kernel( pnew,q,A ) );
    queue.add_kernel(matvec);

    auto pap_inprod = shared_ptr<kernel>( new mpi_innerproduct_kernel( pnew,q,pap ) );
    queue.add_kernel(pap_inprod); pap_inprod->set_name("pap inner product");

    auto alpha_calc = shared_ptr<kernel>( new mpi_scalar_kernel( rr,"/",pap,alpha ) );
    queue.add_kernel(alpha_calc); alpha_calc->set_name("compute alpha");

    auto xupdate = shared_ptr<kernel>( new mpi_axbyz_kernel( '+',one,x, '-',alpha,pnew, xnew ) );
    queue.add_kernel(xupdate); xupdate->set_name("update x");

    auto rupdate = shared_ptr<kernel>( new mpi_axbyz_kernel( '+',one,r, '-',alpha,q, rnew ) );
    queue.add_kernel(rupdate); rupdate->set_name("update r");

    auto rrtest = shared_ptr<kernel>( new mpi_innerproduct_kernel( z,rnew,rrzero[0] ) );
    queue.add_kernel(rrtest) ; rrtest->set_name("test rr orthogonality");

    auto xcopy = shared_ptr<kernel>( new mpi_copy_kernel( xnew,x ) );
    queue.add_kernel(xcopy); xcopy     ->set_name("copy x");

    auto rcopy = shared_ptr<kernel>( new mpi_copy_kernel( rnew,r ) );
    queue.add_kernel(rcopy); rcopy     ->set_name("copy r");
  
    auto rnorm = shared_ptr<kernel>( new mpi_norm_kernel( r,rnorms[0] ) );
    queue.add_kernel(rnorm); rnorm->set_name("r norm");

    queue.analyze_dependencies();
    queue.execute();

    data_pointer data;

    data = z->get_data(mycoord);
    for (index_int i=0; i<nlocal; i++)
      CHECK( data.at(i)==Approx(1.) );

    data = rr->get_data(mycoord);
    CHECK( data.at(0)==Approx(ntids*nlocal) );

    data = pnew->get_data(mycoord);
    for (index_int i=0; i<nlocal; i++)
      CHECK( data.at(i)==Approx(1.) );

    data = q->get_data(mycoord);
    { int lo,hi,i;
      if (mytid==0) lo=1; else lo=0;
      if (mytid==ntids-1) hi=nlocal-2; else hi=nlocal-1;
      for (i=0; i<lo; i++) {
	INFO( "tid: " << mytid << ", i=" << i );
	CHECK( data.at(i)==Approx(1.) );
      }
      for (i=lo; i<=hi; i++) {
	INFO( "tid: " << mytid << ", i=" << i );
	CHECK( data.at(i)==Approx(0.) );
      }
      for (i=hi+1; i<nlocal; i++) {
	INFO( "tid: " << mytid << ", i=" << i );
	CHECK( data.at(i)==Approx(1.) );
      }
    }

    data = pap->get_data(mycoord);
    CHECK( data.at(0)==Approx(2.) );

    data = alpha->get_data(mycoord);
    CHECK( data.at(0)==Approx(ntids*nlocal/2.) );

    data = rrzero[0]->get_data(mycoord);
    CHECK( data.at(0)==Approx(0.) );

  }

  SECTION( "two iterations" ) {

    data_pointer data;

    auto
      x0 = shared_ptr<object>( new mpi_object(blocked) ),
      r0 = shared_ptr<object>( new mpi_object(blocked) ),
      z0 = shared_ptr<object>( new mpi_object(blocked) ),
      p0 = shared_ptr<object>( new mpi_object(blocked) ),
      q0 = shared_ptr<object>( new mpi_object(blocked) ),
      xnew0 = shared_ptr<object>( new mpi_object(blocked) ),
      rnew0 = shared_ptr<object>( new mpi_object(blocked) ),
      pnew0 = shared_ptr<object>( new mpi_object(blocked) ),
      rr0    = shared_ptr<object>( new mpi_object(scalar) ),
      rrp0   = shared_ptr<object>( new mpi_object(scalar) ),
      pap0   = shared_ptr<object>( new mpi_object(scalar) ),
      alpha0 = shared_ptr<object>( new mpi_object(scalar) ),
      beta0  = shared_ptr<object>( new mpi_object(scalar) );

    shared_ptr<object> x,r, z,p,q, pnew /* xnew,rnew,rrp have to persist */,
      rr,rrp,pap,alpha,beta;
    x = shared_ptr<object>( new mpi_object(blocked,x0) );
    r = shared_ptr<object>( new mpi_object(blocked,r0) );
    z = shared_ptr<object>( new mpi_object(blocked,z0) );
    p = shared_ptr<object>( new mpi_object(blocked,p0) );
    q = shared_ptr<object>( new mpi_object(blocked,q0) );
    rr    = shared_ptr<object>( new mpi_object(scalar,rr0) );
    pap   = shared_ptr<object>( new mpi_object(scalar,pap0) );
    alpha = shared_ptr<object>( new mpi_object(scalar,alpha0) );
    beta  = shared_ptr<object>( new mpi_object(scalar,beta0) );

    shared_ptr<kernel> xorigin,rorigin,
      rnorm,precon,rho_inprod,pisz,matvec,pap_inprod,alpha_calc,beta_calc,
      xupdate,rupdate,pupdate, xcopy,rcopy,pcopy,rrcopy;

    xorigin = shared_ptr<kernel>( new mpi_origin_kernel( x ) );
    xorigin->set_name("origin x0");
    queue.add_kernel(xorigin);
    rorigin = shared_ptr<kernel>( new mpi_origin_kernel( r ) );
    rorigin->set_name("origin r0");
    queue.add_kernel(rorigin);

    REQUIRE_NOTHROW( xnew0->allocate() );
    xnew = shared_ptr<object>( new mpi_object(blocked,xnew0) );
    REQUIRE_NOTHROW( rnew0->allocate() );
    rnew = shared_ptr<object>( new mpi_object(blocked,rnew0) );
    REQUIRE_NOTHROW( pnew0->allocate() );
    pnew  = shared_ptr<object>( new mpi_object(blocked,pnew0) );

    rnorm = shared_ptr<kernel>( new mpi_norm_kernel( r,rnorms[0] ) );
    queue.add_kernel(rnorm); rnorm->set_name("r norm");

    precon = shared_ptr<kernel>( new mpi_preconditioning_kernel( r,z ) );
    queue.add_kernel(precon);

    rho_inprod = shared_ptr<kernel>( new mpi_innerproduct_kernel( r,z,rr ) );
    queue.add_kernel(rho_inprod); rho_inprod->set_name("compute rho");

    pisz = shared_ptr<kernel>( new mpi_copy_kernel( z,pnew ) );
      queue.add_kernel(pisz); pisz->set_name("copy z to p");
  
    REQUIRE_NOTHROW( rrp0->allocate() );
    rrp = shared_ptr<object>( new mpi_object(scalar,rrp0) );

    matvec = shared_ptr<kernel>( new mpi_spmvp_kernel( pnew,q,A ) );
    queue.add_kernel(matvec);

    pap_inprod = shared_ptr<kernel>( new mpi_innerproduct_kernel( pnew,q,pap ) );
    queue.add_kernel(pap_inprod); pap_inprod->set_name("pap inner product");

    alpha_calc = shared_ptr<kernel>( new mpi_scalar_kernel( rr,"/",pap,alpha ) );
    queue.add_kernel(alpha_calc); alpha_calc->set_name("compute alpha");

    xupdate = shared_ptr<kernel>( new mpi_axbyz_kernel( '+',one,x, '-',alpha,pnew, xnew ) );
    queue.add_kernel(xupdate); xupdate->set_name("update x");

    rupdate = shared_ptr<kernel>( new mpi_axbyz_kernel( '+',one,r, '-',alpha,q, rnew ) );
    queue.add_kernel(rupdate); rupdate->set_name("update r");

    REQUIRE_NOTHROW( queue.analyze_kernel_dependencies() );

    xcopy = shared_ptr<kernel>( new mpi_copy_kernel( xnew,x ) );
      queue.add_kernel(xcopy); xcopy->set_name("copy x");
      rcopy = shared_ptr<kernel>( new mpi_copy_kernel( rnew,r ) );
      queue.add_kernel(rcopy); rcopy->set_name("copy r");
      pcopy = shared_ptr<kernel>( new mpi_copy_kernel( pnew,p ) );
      queue.add_kernel(pcopy); pcopy->set_name("copy p");

      xnew = shared_ptr<object>( new mpi_object(blocked,xnew0) );
      rnew = shared_ptr<object>( new mpi_object(blocked,rnew0) );
      pnew  = shared_ptr<object>( new mpi_object(blocked,pnew0) );

      rnorm = shared_ptr<kernel>( new mpi_norm_kernel( r,rnorms[1] ) );
    queue.add_kernel(rnorm); rnorm->set_name("r norm");

    precon = shared_ptr<kernel>( new mpi_preconditioning_kernel( r,z ) );
    queue.add_kernel(precon);

    rho_inprod = shared_ptr<kernel>( new mpi_innerproduct_kernel( r,z,rr ) );
    queue.add_kernel(rho_inprod); rho_inprod->set_name("compute rho");

    beta_calc = shared_ptr<kernel>( new mpi_scalar_kernel( rr,"/",rrp,beta ) );
      queue.add_kernel(beta_calc); beta_calc ->set_name("compute beta");

      pupdate = shared_ptr<kernel>( new mpi_axbyz_kernel( '+',one,z, '+',beta,p, pnew ) );
      queue.add_kernel(pupdate); pupdate   ->set_name("update p");

      rrcopy = shared_ptr<kernel>( new mpi_copy_kernel( rr,rrp ) );
      queue.add_kernel(rrcopy); rrcopy    ->set_name("save rr value");
  
      rrp = shared_ptr<object>( new mpi_object(scalar,rrp0) );

      matvec = shared_ptr<kernel>( new mpi_spmvp_kernel( pnew,q,A ) );
    queue.add_kernel(matvec);

    pap_inprod = shared_ptr<kernel>( new mpi_innerproduct_kernel( pnew,q,pap ) );
    queue.add_kernel(pap_inprod); pap_inprod->set_name("pap inner product");

    alpha_calc = shared_ptr<kernel>( new mpi_scalar_kernel( rr,"/",pap,alpha ) );
    queue.add_kernel(alpha_calc); alpha_calc->set_name("compute alpha");

    xupdate = shared_ptr<kernel>( new mpi_axbyz_kernel( '+',one,x, '-',alpha,pnew, xnew ) );
    queue.add_kernel(xupdate); xupdate->set_name("update x");

    rupdate = shared_ptr<kernel>( new mpi_axbyz_kernel( '+',one,r, '-',alpha,q, rnew ) );
    queue.add_kernel(rupdate); rupdate->set_name("update r");

    queue.analyze_dependencies();
    queue.execute();
  }

}

