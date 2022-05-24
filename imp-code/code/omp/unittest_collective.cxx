/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2020
 ****
 **** Unit tests for the OpenMP product backend of IMP
 **** based on the CATCH framework (https://github.com/philsquared/Catch)
 ****
 **** unit tests for collective operations
 ****
 ****************************************************************/

#include <stdlib.h>
#include <math.h>

#include "catch.hpp"

#include "omp_base.h"
#include "omp_ops.h"
#include "omp_static_vars.h"
#include "unittest_functions.h"

using std::vector;

using std::shared_ptr;

using fmt::format;
using fmt::print;

TEST_CASE( "Analyze gather dependencies","[collective][dependence][10]") {
  /*
   * this test case is basically spelling out "analyze_dependencies"
   * for the case of an allgather
   */

  //snippet gatherdists
  shared_ptr<distribution> alpha,gamma;
  shared_ptr<object> scalar,gathered;
  // alpha distribution is one scalar per processor
  REQUIRE_NOTHROW( alpha = shared_ptr<distribution>( new omp_block_distribution(decomp,ntids) ) );
  REQUIRE_NOTHROW( scalar = shared_ptr<object>( new omp_object(alpha) ) );
  // gamma distribution is gathering those scalars
  REQUIRE_NOTHROW( gamma = shared_ptr<distribution>( new omp_gathered_distribution(decomp) ) );
  REQUIRE_NOTHROW( gathered = shared_ptr<object>( new omp_object(gamma) ) );
  auto gather = shared_ptr<kernel>( new omp_kernel(scalar,gathered) );
  REQUIRE_NOTHROW( gather->set_explicit_beta_distribution( gamma ) );
  //snippet end

  // in the alpha structure I have only myself
  for (int mytid=0; mytid<ntids; mytid++) {
    processor_coordinate mycoord;
    REQUIRE_NOTHROW( mycoord = decomp.coordinate_from_linear(mytid) );
    CHECK( alpha->get_processor_structure(mycoord)->volume()==1 );
    CHECK( alpha->get_processor_structure(mycoord)->first_index_r()[0]==mytid );
  }

  REQUIRE_NOTHROW( gather->set_last_dependency().endow_beta_object(gathered) );

  shared_ptr<task> gather_task;
  CHECK_NOTHROW( gather->split_to_tasks() );
  CHECK_NOTHROW( gather_task = gather->get_tasks().at(0) );

  shared_ptr<distribution> beta_dist;
  REQUIRE_NOTHROW( beta_dist = gather->get_beta_distribution() );

  for (int mytid=0; mytid<ntids; mytid++) {
    processor_coordinate mycoord;
    REQUIRE_NOTHROW( mycoord = decomp.coordinate_from_linear(mytid) );
    vector<shared_ptr<message>> msgs;
    auto betablock = beta_dist->get_processor_structure(mycoord);
    INFO( format("{} finding msgs for {}",mycoord.as_string(),betablock->as_string()) );
    CHECK_NOTHROW( msgs = alpha->messages_for_segment
		   ( 0,mycoord,self_treatment::INCLUDE,betablock,betablock) );
    CHECK( msgs.size()==ntids );
    for ( auto msg : msgs ) { //int imsg=0; imsg<msgs->size(); imsg++) {
      CHECK( msg->get_receiver()==mycoord );
    }
  }
}

TEST_CASE( "Analyze gather dependencies in one go","[collective][dependence][11]") {

  // alpha distribution is one scalar per processor
  auto alpha = 
    shared_ptr<distribution>( new omp_block_distribution(decomp,ntids) );
  // gamma distribution is gathering those scalars
  auto gamma = shared_ptr<distribution>( new omp_gathered_distribution(decomp) );
  // general stuff
  auto
    scalar = shared_ptr<object>( new omp_object(alpha) ),
    gathered = shared_ptr<object>( new omp_object(gamma) );
  shared_ptr<kernel> gather = shared_ptr<kernel>( new omp_kernel(scalar,gathered) );
  gather->set_explicit_beta_distribution( gamma );
  //  REQUIRE_NOTHROW( gather->derive_beta_distribution() );

  CHECK_NOTHROW( gather->analyze_dependencies() );
  shared_ptr<task> gather_task;
  CHECK( gather->get_tasks().size()==ntids );
  CHECK_NOTHROW( gather_task = gather->get_tasks().at(0) );

  int nsends,sum;
  sum=0;
  auto msgs = gather_task->get_receive_messages();
  for ( auto msg : msgs ) {
    sum += msg->get_sender().coord(0);
  }
  CHECK( sum==(ntids*(ntids-1)/2) );

  // CHECK_NOTHROW( gather_task->set_send_messages
  // 		 (gather_task->create_send_structure_for_task(0,mytid,alpha) ) );
  // CHECK_NOTHROW( gather_task->get_send_messages() );
  // vector<shared_ptr<message>> *msgs = gather_task->get_send_messages();
  // CHECK( msgs->size()==ntids );
  // sum=0;
  // for (vector<shared_ptr<message>>::iterator msg=msgs->begin(); msg!=msgs->end(); ++msg) {
  //   sum += (*msg)->get_receiver();
  // }
  // CHECK( sum==(ntids*(ntids-1)/2) );

  //  CHECK_NOTHROW( gather_task->set_last_dependency().endow_beta_object(gathered) );
  CHECK( gather_task->get_beta_object(0)!=nullptr );
  index_int hsize;
  for (int mytid=0; mytid<ntids; mytid++) {
    processor_coordinate mycoord;
    REQUIRE_NOTHROW( mycoord = decomp.coordinate_from_linear(mytid) );
    CHECK_NOTHROW( hsize = gather_task->get_out_object()->volume(mycoord) );
    CHECK( hsize==ntids );
    CHECK_NOTHROW( hsize = gather_task->get_beta_object(0)->volume(mycoord) );
    CHECK( hsize==ntids );
    // CHECK_NOTHROW( hsize = gather_task->get_invector()->volume(mycoord) );
    // CHECK( hsize==1 );
  }
}

TEST_CASE( "Actually gather and reduce something","[collective][dependence][reduction][12]") {
  /*
    This test is not quite correct. The `summing' routine
    really wants a scalar output, not a gathered array.
   */

  // alpha distribution is one scalar per processor
  auto alpha = shared_ptr<distribution>( new omp_block_distribution(decomp,ntids) );
  // gamma distribution is gathering those scalars
  auto gamma = shared_ptr<distribution>( new omp_gathered_distribution(decomp) );
  // general stuff
  auto
    scalar = shared_ptr<object>( new omp_object(alpha) ),
    gathered = shared_ptr<object>( new omp_object(gamma) );
  shared_ptr<kernel> gather;

  omp_algorithm *queue = new omp_algorithm(decomp);
  REQUIRE_NOTHROW( queue->add_kernel( shared_ptr<kernel>( new omp_origin_kernel(scalar) ) ) );
  for (int mytid=0; mytid<ntids; mytid++) {
    processor_coordinate mycoord;
    REQUIRE_NOTHROW( mycoord = decomp.coordinate_from_linear(mytid) );
    decltype( scalar->get_data(mycoord) ) in;
    CHECK( scalar->volume(mycoord)==1 );
    REQUIRE_NOTHROW( in = scalar->get_data(mycoord) );
    in.at(0) = (double)mytid;
  }

  shared_ptr<task> gather_task;
  const char *path;
  SECTION( "spell out the kernel" ) {
    path = "spelled out";
    gather = shared_ptr<kernel>( new omp_kernel(scalar,gathered) );
    gather->set_explicit_beta_distribution( gamma );
    gather->set_localexecutefn( &summing );
  }
  // SECTION( "use a reduction kernel" ) {
  //   path = "kernel";
  //   REQUIRE_NOTHROW( gather = shared_ptr<kernel>( new omp_reduction_kernel(scalar,gathered) ) );
  // }
  // INFO( "gather strategy: " << path );

  REQUIRE_NOTHROW( queue->add_kernel( gather ) );
  REQUIRE_NOTHROW( queue->analyze_dependencies() );
  CHECK( gather->get_tasks().size()==ntids );
  // REQUIRE_NOTHROW( queue->execute() );
  // shared_ptr<vector<double>> out = gathered->get_data(new processor_coordinate_zero(1));
  // CHECK( (*out)==(ntids*(ntids-1)/2) );
}

TEST_CASE( "Gather more than one something","[collective][dependence][13]") {
  printf("13: multi-component reduction disabled\n"); return;

  // alpha distribution is k scalars per processor
  int k, P = ntids;
  const char *path;
  SECTION( "reproduce single gather" ) {
    path = "single"; k = 1;
  }
  printf("[13] multiple gather disabled\n");
  // SECTION( "new: multiple gather" ) {
  //   path = "multiple" ; k = 4;
  // }
  INFO( "path: " << path );
  
  shared_ptr<distribution> alpha ,gamma;
  alpha = shared_ptr<distribution>( new omp_block_distribution(decomp,k,P) );
  // gamma distribution is gathering those scalars
  REQUIRE_NOTHROW( gamma = shared_ptr<distribution>( new omp_gathered_distribution(decomp,k) ) );
  CHECK( gamma->has_type_replicated() );
  CHECK( gamma->get_orthogonal_dimension()==k );
  // general stuff
  shared_ptr<object> scalars,gathered;
  REQUIRE_NOTHROW( scalars = shared_ptr<object>( new omp_object(alpha) ) );
  REQUIRE_NOTHROW( gathered = shared_ptr<object>( new omp_object(gamma) ) );
  shared_ptr<kernel> gather;
  REQUIRE_NOTHROW(  gather = shared_ptr<kernel>( new omp_kernel(scalars,gathered) ) );
  REQUIRE_NOTHROW( gather->set_explicit_beta_distribution( gamma ) );
  gather->set_localexecutectx( (void*)&k );
  gather->set_localexecutefn( &veccopy );

  shared_ptr<task> gather_task;
  // REQUIRE_NOTHROW( gather->set_last_dependency().ensure_beta_distribution(gathered) );
  // CHECK_NOTHROW( gather->split_to_tasks() );
  REQUIRE_NOTHROW( gather->analyze_dependencies() );
  CHECK( gather->get_tasks().size()==P );
  omp_algorithm *queue = new omp_algorithm(decomp);
  REQUIRE_NOTHROW( queue->add_kernel( shared_ptr<kernel>( new omp_origin_kernel(scalars) ) ) );
  REQUIRE_NOTHROW( queue->add_kernel( gather ) );
  REQUIRE_NOTHROW( queue->analyze_dependencies() );
  // decltype( scalars->get_data(new processor_coordinate_zero(1)) ) indata;
  data_pointer indata;
  REQUIRE_NOTHROW( indata = scalars->get_data(new processor_coordinate_zero(1)) );
  for (int p=0; p<ntids; p++) {
    for (int ik=0; ik<k; ik++) {
      INFO( "ik: " << ik );
      indata.at(p*k+ik) = (ik+1)*(double)p;
    }
  }
  REQUIRE_NOTHROW( queue->execute() );
  // decltype( scalars->get_data(new processor_coordinate_zero(1)) ) outdata;
  data_pointer outdata;
  for (int p=0; p<ntids; p++) {
    processor_coordinate mycoord;
    REQUIRE_NOTHROW( mycoord = decomp.coordinate_from_linear(p) );
    REQUIRE_NOTHROW( outdata = gathered->get_data(mycoord) );
    //printf("%d checking out  data at %ld\n",p,(long)out);
    INFO( "on proc " << p );
    for (int ik=0; ik<k; ik++) {
      INFO( "ik: " << ik );
      CHECK( outdata.at(ik+p*k)==(ik+1)*p );
      //CHECK( out[ik]==((ik+1)*ntids*(ntids-1)/2) );
    }
  }
}

TEST_CASE( "Gather and sum as two kernels; first kernel","[collective][dependence][14]") {

  // input distribution is one scalar per processor
  shared_ptr<distribution> distributed,collected,replicated;
  REQUIRE_NOTHROW( distributed = shared_ptr<distribution>( new omp_block_distribution(decomp,ntids) ) );
  auto scalar      = shared_ptr<object>( new omp_object(distributed) );
  for (int mytid=0; mytid<ntids; mytid++) {
    processor_coordinate mycoord;
    REQUIRE_NOTHROW( mycoord = decomp.coordinate_from_linear(mytid) );
    CHECK( scalar->volume(mycoord)==1 );
  }

  {
    //shared_ptr<vector<double>> scalar_data;
    decltype( scalar->get_data(processor_coordinate_zero(1)) ) scalar_data;
    REQUIRE_NOTHROW( scalar_data = scalar->get_data(processor_coordinate_zero(1)) );
    for (int mytid=0; mytid<ntids; mytid++) {
      // could actually do get_data(mycoord) here, but still have to index:
      scalar_data.at(mytid) = (double)mytid+.5;
    }
  }
  
  algorithm queue = omp_algorithm(decomp);
  REQUIRE_NOTHROW( queue.add_kernel( shared_ptr<kernel>( new omp_origin_kernel(scalar) ) ) );

  // intermediate distribution is gathering those scalars replicated
  REQUIRE_NOTHROW( collected = shared_ptr<distribution>( new omp_gathered_distribution(decomp) ) );

  auto gathered  = shared_ptr<object>( new omp_object(collected) );
  CHECK( gathered->get_orthogonal_dimension()==1 );
  REQUIRE_NOTHROW( gathered->allocate() );

  decltype( gathered->get_data( processor_coordinate_zero(1) ) ) data0;
  //shared_ptr<vector<double>> data0;
  for (int mytid=0; mytid<ntids; mytid++) {
    INFO( "mytid=" << mytid );
    processor_coordinate mycoord;
    REQUIRE_NOTHROW( mycoord = decomp.coordinate_from_linear(mytid) );
    CHECK( gathered->volume(mycoord)==ntids );
    //shared_ptr<vector<double>> data;
    decltype( gathered->get_data(mycoord) ) data;
    REQUIRE_NOTHROW( data = gathered->get_data(mycoord) );
    if (mytid==0) data0 = data;
    CHECK( ((long)data.data()-(long)data0.data())==(size_t)(mytid*ntids*sizeof(double)) );
  }

  // first kernel: gather
  shared_ptr<kernel> gather;
  REQUIRE_NOTHROW( gather = shared_ptr<kernel>( new omp_kernel(scalar,gathered) ) );
  REQUIRE_NOTHROW( gather->set_name("gather from scalar") );
  REQUIRE_NOTHROW( gather->set_explicit_beta_distribution( collected ) );
  gather->set_localexecutefn( &veccopy );

  REQUIRE_NOTHROW( queue.add_kernel( gather ) );

  //  "analyze intermediate"
  REQUIRE_NOTHROW( queue.analyze_dependencies() );
  vector<shared_ptr<task>> tasks;
  REQUIRE_NOTHROW( tasks = queue.get_tasks() );
  REQUIRE( tasks.size()==2*ntids );
  for (auto t : tasks) {
    if (!t->has_type_origin()) {
      vector<shared_ptr<message>> msgs;
      REQUIRE_NOTHROW( msgs = t->get_receive_messages() );
      REQUIRE( msgs.size()==ntids );
      vector<int> senders(ntids,0);
      //for (int i=0; i<ntids; i++) senders.at(i) = 0;
      for (auto m : msgs) {
	int sender = m->get_sender().coord(0);
	REQUIRE_NOTHROW( senders.at( sender )++ ); // mark all senders
      }
      for (int i=0; i<ntids; i++)
	REQUIRE( senders.at(i)==1 ); // make sure all senders occur once
    }
  }
  // execute just the intermediate
  REQUIRE_NOTHROW( queue.execute() );
    
  //    shared_ptr<vector<double>> prev_data;
  shared_ptr<vector<double>> gathered_data;
  CHECK( gathered->has_type_replicated() );
  for (int mytid=0; mytid<ntids; mytid++) {
    INFO( "mytid=" << mytid );
    processor_coordinate mycoord;
    REQUIRE_NOTHROW( mycoord = decomp.coordinate_from_linear(mytid) );
    INFO( format("investigating data on pcoord={}",mycoord.as_string()) );
    data_pointer domain_pointer;
    REQUIRE_NOTHROW( domain_pointer = gathered->get_nth_data_pointer(mytid) ); // nth?
    CHECK( gathered->volume(mycoord)==ntids );
    CHECK( domain_pointer.size()==ntids );
    index_int offset = mytid*ntids;
    //    CHECK( domain_pointer.o==offset );
    // every process has the same gathered data
    for (int id=0; id<ntids; id++)
      CHECK( domain_pointer.at(id)==Approx((double)id+.5) );
  }
}

TEST_CASE( "Gather and sum as two kernels; second kernel","[collective][dependence][15]") {

  /*
   * Setup is the same as in [14]
   */

  // input distribution is one scalar per processor
  shared_ptr<distribution> distributed,collected,replicated;
  REQUIRE_NOTHROW( distributed = shared_ptr<distribution>( new omp_block_distribution(decomp,ntids) ) );
  auto scalar      = shared_ptr<object>( new omp_object(distributed) );

  //shared_ptr<vector<double>> gathered_data;
  decltype( scalar->get_data(processor_coordinate_zero(1)) ) scalar_data;
  REQUIRE_NOTHROW( scalar_data = scalar->get_data(processor_coordinate_zero(1)) );
  for (int mytid=0; mytid<ntids; mytid++) {
    // could actually do get_data(mycoord) here, but still have to index:
    scalar_data.at(mytid) = (double)mytid+.5;
  }
  
  algorithm queue = omp_algorithm(decomp);
  REQUIRE_NOTHROW( queue.add_kernel( shared_ptr<kernel>( new omp_origin_kernel(scalar) ) ) );

  // intermediate distribution is gathering those scalars replicated
  REQUIRE_NOTHROW( collected = shared_ptr<distribution>( new omp_gathered_distribution(decomp) ) );
  auto gathered  = shared_ptr<object>( new omp_object(collected) );
  CHECK( gathered->get_orthogonal_dimension()==1 );
  REQUIRE_NOTHROW( gathered->allocate() );

  // final distribution is redundant one scalar
  REQUIRE_NOTHROW( replicated = shared_ptr<distribution>( new omp_replicated_distribution(decomp) ) );
  auto sum      = shared_ptr<object>( new omp_object(replicated) );

  // first kernel: gather
  shared_ptr<kernel> gather;
  REQUIRE_NOTHROW( gather = shared_ptr<kernel>( new omp_kernel(scalar,gathered) ) );
  REQUIRE_NOTHROW( gather->set_name("gather from scalar") );
  REQUIRE_NOTHROW( gather->set_explicit_beta_distribution( collected ) );
  gather->set_localexecutefn( &veccopy );

  REQUIRE_NOTHROW( queue.add_kernel( gather ) );

  { //  SECTION( "add the second kernel" ) {
    // second kernel: local sum
    shared_ptr<kernel> localsum = shared_ptr<kernel>( new omp_kernel(gathered,sum) );
    localsum->set_localexecutefn( &summing );

    const char *beta_strategy = "none";
    //    SECTION( "explicit beta works" ) {
    beta_strategy = "beta is collected distribution";
    localsum->set_explicit_beta_distribution( collected );
    //}
    // SECTION(  "derivation does not work" ) {
    //   beta_strategy = "beta derived";
    //   localsum->set_type_local();
    // }
    INFO( beta_strategy );

    REQUIRE_NOTHROW( queue.add_kernel(localsum) );
    REQUIRE_NOTHROW( queue.analyze_dependencies() );

    for (int mytid=0; mytid<ntids; mytid++) {
      processor_coordinate mycoord;
      REQUIRE_NOTHROW( mycoord = decomp.coordinate_from_linear(mytid) );
      CHECK( localsum->last_dependency().get_beta_distribution()->volume(mycoord)==ntids );
    }
    CHECK( localsum->get_tasks().size()==ntids );

    REQUIRE_NOTHROW( queue.execute() );
    print("Early return from [15]\n"); return;


    //    shared_ptr<vector<double>> out;
    for (int mytid=0; mytid<ntids; mytid++) {
      processor_coordinate mycoord;
      REQUIRE_NOTHROW( mycoord = decomp.coordinate_from_linear(mytid) );
      decltype(sum->get_data(mycoord)) out;
      INFO( "thread " << mytid );
      CHECK( sum->volume(mycoord)==1 );
      REQUIRE_NOTHROW( out = sum->get_data(mycoord) );
      CHECK( out.at(0)==(ntids*(ntids-1)/2) );
    }
  }
}

TEST_CASE( "Inner product in three kernels","[collective][30]" ) {

  // let's go for an irregular distribution
  vector<index_int> local_sizes;
  for (int mytid=0; mytid<ntids; mytid++)
    local_sizes.push_back(10+2*mytid);
  shared_ptr<distribution> alpha,localscalar,gathered_distribution,summed_scalar;
  REQUIRE_NOTHROW( alpha = shared_ptr<distribution>
		   ( new omp_block_distribution(decomp,local_sizes) ) );
  REQUIRE_NOTHROW( localscalar = shared_ptr<distribution>
		   ( new omp_block_distribution(decomp,1,-1) ) );
  REQUIRE_NOTHROW( gathered_distribution = shared_ptr<distribution>
		   ( new omp_gathered_distribution(decomp) ) );
  REQUIRE_NOTHROW( summed_scalar = shared_ptr<distribution>
		   ( new omp_replicated_distribution(decomp) ) );
  {
    index_int g = ntids*10+ntids*(ntids-1);
    CHECK( alpha->global_volume()==g );
  }
  auto
    x = shared_ptr<object>( new omp_object(alpha) ),
    y = shared_ptr<object>( new omp_object(alpha) );
  x->set_name("inprod-x"); y->set_name("inprod-y");

  // set local data
  shared_ptr<vector<double>> xdata,ydata;
  REQUIRE_NOTHROW( xdata = x->get_numa_data_pointer() ); //(new processor_coordinate_zero(1)) );
  for (index_int i=0; i<x->global_volume(); i++)
    xdata->at(i) = 2.;
  REQUIRE_NOTHROW( ydata = y->get_numa_data_pointer() ); //(new processor_coordinate_zero(1)) );
  for (index_int i=0; i<y->global_volume(); i++)
    ydata->at(i) = i;

  // output object
  auto globalvalue = shared_ptr<object>( new omp_object(summed_scalar) );
  globalvalue->set_name("inprod-value");

  algorithm *inprod = new omp_algorithm(decomp);
  REQUIRE_NOTHROW( inprod->add_kernel( shared_ptr<kernel>( new omp_origin_kernel(x) ) ) );
  REQUIRE_NOTHROW( inprod->add_kernel( shared_ptr<kernel>( new omp_origin_kernel(y) ) ) );
  
  SECTION( "explicit intermediates" ) {
    std::cout << "explicit intermediates\n";
    // intermediate objects
    auto
      localvalue = shared_ptr<object>( new omp_object(localscalar) ),
      gatheredvalue = shared_ptr<object>( new omp_object(gathered_distribution) );

    // local product is local, second vector comes in as context
    auto local_product = shared_ptr<kernel>( new omp_kernel(x,localvalue) );
    local_product->set_localexecutefn( &local_inner_product );

    REQUIRE_NOTHROW( local_product->set_last_dependency().set_explicit_beta_distribution(x) );
    REQUIRE_NOTHROW( local_product->add_in_object(y) );
    REQUIRE_NOTHROW( local_product->set_last_dependency().set_explicit_beta_distribution(y) );

    REQUIRE_NOTHROW( inprod->add_kernel(local_product) );
    
    // SECTION( "three kernels" ) {
    //   std::cout << "three kernels\n";
    //   omp_kernel
    // 	*gather = shared_ptr<kernel>( new omp_kernel(localvalue,gatheredvalue) ),
    // 	*sum = shared_ptr<kernel>( new omp_kernel(gatheredvalue,globalvalue) );
    //   gather->set_explicit_beta_distribution(gathered_distribution);
    //   gather->set_localexecutefn( &veccopy );
    //   REQUIRE_NOTHROW( gather->ensure_beta_distribution(gatheredvalue) );
    //   REQUIRE_NOTHROW( gather->analyze_dependencies() );
    //   // sum is local
    //   sum->set_type_local();
    //   sum->set_localexecutefn( &summing );

    //   inprod->add_kernel(sum);
    //   inprod->add_kernel(gather);
    // }

    // SECTION( "two kernels" ) {
    std::cout << "two kernels\n";
    auto gather_and_sum = shared_ptr<kernel>( new omp_kernel(localvalue,globalvalue) );
    gather_and_sum->set_explicit_beta_distribution(gathered_distribution);
    gather_and_sum->set_localexecutefn( &summing );

    inprod->add_kernel(gather_and_sum);
      // }
  }

  SECTION( "inner product kernel" ) {
    shared_ptr<kernel> innerproduct;
    REQUIRE_NOTHROW( innerproduct = shared_ptr<kernel>
		     ( new omp_innerproduct_kernel(x,y,globalvalue) ) );
    REQUIRE_NOTHROW( inprod->add_kernel( innerproduct ) );
  }
  
  REQUIRE_NOTHROW( inprod->analyze_dependencies() );
  REQUIRE_NOTHROW( inprod->execute() );

  index_int g = alpha->global_volume(); 
  shared_ptr<vector<double>> zdata;
  REQUIRE_NOTHROW( zdata = globalvalue->get_numa_data_pointer() );
  CHECK( globalvalue->global_volume()==1 );
  CHECK( zdata->at(0)==g*(g-1.) );

}

TEST_CASE( "Inner product with different strategies","[collective][31]" ) {

  architecture aa;
  REQUIRE_NOTHROW( aa = env->make_architecture() );
  
  std::string path;
  SECTION( "point-to-point" ) { path = std::string("point-to-point");
    aa.set_collective_strategy_ptp();
  }
  SECTION( "grouping" ) { path = std::string("grouping");
    aa.set_collective_strategy_group();
  }
  SECTION( "treewise" ) { path = std::string("treewise");
    aa.set_collective_strategy_recursive();
  }
  SECTION( "collectivewise" ) { path = std::string("collective");
    aa.set_collective_strategy(collective_strategy::MPI);
  }
  INFO( "collectives are done " << path );

  decomposition decomp = omp_decomposition(aa);

  // let's go for an irregular distribution
  vector<index_int> local_sizes;
  for (int mytid=0; mytid<ntids; mytid++)
    local_sizes.push_back(10+2*mytid);
  auto
    alpha = shared_ptr<distribution>( new omp_block_distribution(decomp,local_sizes) ),
    local_scalar = shared_ptr<distribution>( new omp_block_distribution(decomp,1,-1) ),
    gathered_scalar = shared_ptr<distribution>( new omp_gathered_distribution(decomp) ),
    summed_scalar = shared_ptr<distribution>( new omp_replicated_distribution(decomp) );
  {
    index_int g = ntids*10+ntids*(ntids-1);
    CHECK( alpha->global_volume()==g );
  }
  auto
    x = shared_ptr<object>( new omp_object(alpha) ),
    y = shared_ptr<object>( new omp_object(alpha) );
  REQUIRE_NOTHROW( x->allocate() );
  REQUIRE_NOTHROW( y->allocate() );
  x->set_name("inprod-x"); y->set_name("inprod-y");

  // set local data
  shared_ptr<vector<double>> xdata,ydata;
  REQUIRE_NOTHROW( xdata = x->get_numa_data_pointer() ); //(new processor_coordinate_zero(1)) );
  for (index_int i=0; i<x->global_volume(); i++)
    xdata->at(i) = 2.;
  REQUIRE_NOTHROW( ydata = y->get_numa_data_pointer() ); //(new processor_coordinate_zero(1)) );
  for (index_int i=0; i<y->global_volume(); i++)
    ydata->at(i) = i;

  // output object
  auto global_sum = shared_ptr<object>( new omp_object(summed_scalar) );
  global_sum->set_name("inprod-value");

  algorithm *inprod = new omp_algorithm(decomp);
  REQUIRE_NOTHROW( inprod->add_kernel( shared_ptr<kernel>( new omp_origin_kernel(x) ) ) );
  REQUIRE_NOTHROW( inprod->add_kernel( shared_ptr<kernel>( new omp_origin_kernel(y) ) ) );

  shared_ptr<kernel> innerproduct;
  REQUIRE_NOTHROW( innerproduct = shared_ptr<kernel>( new omp_innerproduct_kernel(x,y,global_sum) ) );
  REQUIRE_NOTHROW( inprod->add_kernel( innerproduct ) );

  REQUIRE_NOTHROW( inprod->analyze_dependencies() );
  REQUIRE_NOTHROW( inprod->execute() );

  index_int g = alpha->global_volume(); 
  shared_ptr<vector<double>> zdata;
  REQUIRE_NOTHROW( zdata = global_sum->get_numa_data_pointer() ); //(new processor_coordinate_zero(1)) );
  CHECK( global_sum->global_volume()==1 );
  CHECK( zdata->at(0)==g*(g-1.) );

}

