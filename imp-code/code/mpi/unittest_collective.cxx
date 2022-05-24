/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2022
 ****
 **** Unit tests for the MPI product backend of IMP
 **** based on the CATCH framework (https://github.com/philsquared/Catch)
 ****
 **** unit tests for collective operations
 ****
 ****************************************************************/

#include <stdlib.h>
#include <math.h>

#include "catch2/catch_all.hpp"

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

TEST_CASE( "Analyze gather dependencies","[collective][dependence][10]") {
  /*
   * this test case is basically spelling out "analyze_dependencies"
   * for the case of an allgather
   */

  //snippet gatherdists
  // alpha distribution is one scalar per processor
  auto alpha = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,env->get_architecture()->nprocs()) );
  auto scalar = shared_ptr<object>( new mpi_object(alpha) );
  REQUIRE_NOTHROW( scalar->allocate() );
  // gamma distribution is gathering those scalars
  auto gamma = shared_ptr<distribution>( new mpi_gathered_distribution(decomp) );
  auto gathered = shared_ptr<object>( new mpi_object(gamma) );
  auto gather = shared_ptr<kernel>( new mpi_kernel(scalar,gathered) );
  REQUIRE_NOTHROW( gather->set_explicit_beta_distribution( gamma ) );
  //snippet end

  // in the alpha structure I have only myself
  decltype( alpha->get_dimension_structure(0) ) alpha_struct;
  REQUIRE_NOTHROW( alpha_struct = alpha->get_dimension_structure(0) );
  CHECK( alpha_struct->get_processor_structure(mytid)->local_size()==1 );
  CHECK( alpha_struct->get_processor_structure(mytid)->first_index()==mytid );

  REQUIRE_NOTHROW( gather->analyze_dependencies() );

  shared_ptr<task> gather_task;
  CHECK_NOTHROW( gather_task = gather->get_tasks().at(0) );
  {
    vector<shared_ptr<message>> msgs;
    // decltype( gather->get_beta_object()->get_distribution() ) d; VLE problem with const
    shared_ptr<distribution> d;
    REQUIRE_NOTHROW( d = gather->get_beta_object()->get_distribution() );
    decltype( d->get_processor_structure(mycoord) ) betablock;
    REQUIRE_NOTHROW( betablock = d->get_processor_structure(mycoord) );
    CHECK_NOTHROW( msgs = alpha->messages_for_segment
		   (0,mycoord,self_treatment::INCLUDE,betablock,betablock) );
    CHECK( msgs.size()==ntids );
    for ( auto msg : msgs ) {
      CHECK( msg->get_receiver()==mycoord );
    }
  }
  // same but now as one call
  //CHECK_NOTHROW( gather_task->derive_receive_messages(/*0,mytid*/) );
  int nsends;
  CHECK_NOTHROW( nsends = gather_task->get_nsends() );
  CHECK( nsends==env->get_architecture()->nprocs() );

}

TEST_CASE( "Analyze gather dependencies in one go","[collective][dependence][11]") {

  // alpha distribution is one scalar per processor
  auto alpha = 
    shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,-1,env->get_architecture()->nprocs()) );
  // gamma distribution is gathering those scalars
  shared_ptr<distribution> gamma = shared_ptr<distribution>( new mpi_gathered_distribution(decomp) );
  // general stuff
  auto scalar = shared_ptr<object>( new mpi_object(alpha) );
  auto gathered = shared_ptr<object>( new mpi_object(gamma) );
  auto gather = shared_ptr<kernel>( new mpi_kernel(scalar,gathered) );
  gather->set_explicit_beta_distribution( gamma );

  shared_ptr<task> gather_task;
  CHECK_NOTHROW( gather->analyze_dependencies() );
  CHECK( gather->get_tasks().size()==1 );
  CHECK_NOTHROW( gather_task = gather->get_tasks().at(0) );

  int nsends,sum;
  sum=0;
  // for (vector<shared_ptr<message>>::iterator msg=gather_task->get_receive_messages()->begin();
  //      msg!=gather_task->get_receive_messages()->end(); ++msg) {
  for ( auto msg : gather_task->get_receive_messages() ) {
    sum += msg->get_sender().coord(0);
  }
  CHECK( sum==(ntids*(ntids-1)/2) );

  CHECK_NOTHROW( gather_task->get_send_messages() );
  vector<shared_ptr<message>> msgs = gather_task->get_send_messages();
  CHECK( msgs.size()==ntids );
  sum=0;
  //for (vector<shared_ptr<message>>::iterator msg=msgs->begin(); msg!=msgs->end(); ++msg) {
  for ( auto msg : msgs ) {
    sum += msg->get_receiver().coord(0);
  }
  CHECK( sum==(ntids*(ntids-1)/2) );

  //  CHECK_NOTHROW( gather_task->set_last_dependency().create_beta_vector(gathered) );
  CHECK( gather_task->get_beta_object(0)!=nullptr );
  index_int hsize;
  CHECK_NOTHROW( hsize = gather_task->get_out_object()->volume(mycoord) );
  CHECK( hsize==ntids );
  CHECK_NOTHROW( hsize = gather_task->get_beta_object(0)->volume(mycoord) );
  CHECK( hsize==ntids );
}

TEST_CASE( "Actually gather something","[collective][dependence][12]") {

  // alpha distribution is one scalar per processor
  auto alpha = 
    shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,-1,env->get_architecture()->nprocs()) );
  // gamma distribution is gathering those scalars
  shared_ptr<distribution> gamma = shared_ptr<distribution>( new mpi_gathered_distribution(decomp) );
  // general stuff
  auto scalar = shared_ptr<object>( new mpi_object(alpha) );
  auto gathered = shared_ptr<object>( new mpi_object(gamma) );
  REQUIRE_NOTHROW( scalar->allocate() );
  auto gather = shared_ptr<kernel>( new mpi_kernel(scalar,gathered) );
  gather->set_explicit_beta_distribution( gamma );
  gather->set_localexecutefn( summing );

  shared_ptr<task> gather_task;
  CHECK_NOTHROW( gather->analyze_dependencies() );
  CHECK( gather->get_tasks().size()==1 );
  CHECK_NOTHROW( gather_task = gather->get_tasks().at(0) );
  data_pointer in; REQUIRE_NOTHROW( in = scalar->get_data(mycoord) );
  in.at(0) = (double)mytid;
  CHECK_NOTHROW( gather_task->execute() );
  data_pointer out; REQUIRE_NOTHROW( out = gathered->get_data(mycoord) );
  CHECK( out.at(0)==(ntids*(ntids-1)/2) );
}

TEST_CASE( "Gather more than one something","[collective][dependence][ortho][13]") {

  INFO( "mytid=" << mytid );
  // alpha distribution is k scalars per processor
  int k; const char *testcase;
  SECTION( "k=1" ) {
    k = 1; testcase = "k=1";
  }
  SECTION( "k=4" ) {
    k = 4; testcase = "k=4";
  }
  INFO( "test case: " << testcase );

  // k scalars per proc
  shared_ptr<distribution> alpha = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,k,1,-1) );
  auto scalars = shared_ptr<object>( new mpi_object(alpha) );
  REQUIRE_NOTHROW( scalars->allocate() );
  CHECK( alpha->local_allocation()==k );
  data_pointer in; REQUIRE_NOTHROW( in = scalars->get_data(mycoord) );
  for (int ik=0; ik<k; ik++)
    in.at(ik) = (ik+1)*(double)mytid;
  // gamma distribution is gathering those scalars
  shared_ptr<distribution> gamma = shared_ptr<distribution>( new mpi_gathered_distribution(decomp,k,1) );
  auto gathered = shared_ptr<object>( new mpi_object(gamma) );
  CHECK( gamma->local_allocation()==k*ntids );

  auto gather = shared_ptr<kernel>( new mpi_kernel(scalars,gathered) );
  gather->set_explicit_beta_distribution( gamma );
  gather->set_localexecutectx( (void*)&k );
  gather->set_localexecutefn( &veccopy );

  shared_ptr<task> gather_task;
  CHECK_NOTHROW( gather->analyze_dependencies() );
  CHECK( gather->get_tasks().size()==1 );
  CHECK_NOTHROW( gather_task = gather->get_tasks().at(0) );
  CHECK_NOTHROW( gather->execute() );
  auto out = gathered->get_data(mycoord);
  for (int p=0; p<ntids; p++) {
    INFO( "p contribution from " << p );
    for (int ik=0; ik<k; ik++) {
      INFO( "ik: " << ik );
      CHECK( out.at(ik+p*k)==(ik+1)*p );
    }
  }
}

TEST_CASE( "Gather and sum as two kernels","[collective][dependence][14]") {

  // we're going to be tinkering with the distribution so let's make a copy 
  decomposition recomp;
  REQUIRE_NOTHROW( recomp = mpi_decomposition(decomp) );

  bool use_mpi{false}; string path;
  SECTION( "native" ) { path = "native collectives";
    use_mpi = false;
  }
  SECTION( "use mpi" ) { path = "mpi collectives";
    use_mpi = true;
    REQUIRE_NOTHROW( recomp.set_collective_strategy( collective_strategy::MPI  ) );
  }
  INFO( "mytid: " << mytid );
  INFO( format("[14] analyze dependencies using: {}",path) );

  // input distribution is one scalar per processor
  auto distributed = shared_ptr<distribution>( make_shared<mpi_block_distribution>(recomp,1,-1) );
  auto scalar = shared_ptr<object>( new mpi_object(distributed) );
  REQUIRE_NOTHROW( scalar->allocate() );
  data_pointer scalar_data; REQUIRE_NOTHROW( scalar_data = scalar->get_data(mycoord) );
  scalar_data.at(0) = (double)mytid;

  // intermediate distribution is gathering those scalars replicated
  auto collected = shared_ptr<distribution>( new mpi_gathered_distribution(recomp) );
  auto gathered = shared_ptr<object>( new mpi_object(collected) );

  // first kernel: gather
  {
    auto gather = shared_ptr<kernel>( new mpi_kernel(scalar,gathered) );
    REQUIRE_NOTHROW( gather->set_explicit_beta_distribution( collected ) );
    //    REQUIRE_NOTHROW( gather->set_last_dependency().set_is_collective() );

    gather->set_localexecutefn( &veccopy );
    CHECK_NOTHROW( gather->analyze_dependencies() );
    return;
    REQUIRE( (!use_mpi) );

    // there have to be tasks that are collective
    vector<shared_ptr<task>> tsks; shared_ptr<task> tsk;
    REQUIRE_NOTHROW( tsks = gather->get_tasks() );
    CHECK( tsks.size()==1 );
    REQUIRE_NOTHROW( tsk = tsks.at(0) );
    auto deps = tsk->get_dependencies(); bool icol=false;
    for ( auto d : deps ) {
      REQUIRE_NOTHROW( icol = icol || d.get_is_collective() );
    }
    CHECK( ( !use_mpi || icol) );

    CHECK_NOTHROW( gather->execute() );

    // let's look at the gathered data
    CHECK( gathered->volume(mycoord)==ntids );
    CHECK( gathered->get_orthogonal_dimension()==1 );
    {
      auto gathered_data = gathered->get_data(mycoord);
      for (int i=0; i<ntids; i++)
	CHECK( gathered_data.at(i)==(double)i );
    }
  }
  
  // final distribution is redundant one scalar
  auto replicated = shared_ptr<distribution>( new mpi_replicated_distribution(recomp) );
  auto sum  = shared_ptr<object>( new mpi_object(replicated) );

  // second kernel: local sum
  auto localsum = shared_ptr<kernel>( new mpi_kernel(gathered,sum) );
  localsum->set_localexecutefn( &summing );
  shared_ptr<task> sum_task;

  const char *beta_strategy = "none";
  beta_strategy = "beta is collected distribution";
  localsum->set_explicit_beta_distribution( collected );
  INFO( beta_strategy );

  CHECK_NOTHROW( localsum->analyze_dependencies() );
  CHECK( localsum->get_beta_object()->get_distribution()->volume(mycoord)==ntids );
  CHECK( localsum->get_tasks().size()==1 );
  CHECK_NOTHROW( sum_task = localsum->get_tasks().at(0) );

  vector<shared_ptr<message>> msgs;
  REQUIRE_NOTHROW( msgs = sum_task->get_receive_messages() );
  CHECK( msgs.size()==1 );
  CHECK( msgs.at(0)->get_sender()==mycoord );
  CHECK_NOTHROW( sum_task->execute() );

  auto in = scalar->get_data(mycoord),
    out = sum->get_data(mycoord);
  CHECK( out.at(0)==(ntids*(ntids-1)/2) );
}

TEST_CASE( "Analyze MPI non-blocking collective dependencies","[collective][nonblock][20]" ) {

  INFO( "mytid: " << mytid);

  decomposition recomp;
  REQUIRE_NOTHROW( recomp = mpi_decomposition(decomp) );

  bool use_mpi{false}; string path;
  SECTION( "native" ) { path = "native collectives";
    use_mpi = false;
  }
  SECTION( "use mpi" ) { path = "mpi collectives";
    use_mpi = true;
    REQUIRE_NOTHROW( recomp.set_collective_strategy( collective_strategy::MPI  ) );
  }
  INFO( format("[20] analyze dependencies for {}",path) );

  auto local_scalar = shared_ptr<distribution>( new mpi_scalar_distribution(recomp) ),
    reduc_scalar = shared_ptr<distribution>( new mpi_replicated_distribution(recomp,1) );
  auto local_value = shared_ptr<object>( new mpi_object(local_scalar) );
  auto reduc_value = shared_ptr<object>( new mpi_object(reduc_scalar) );
  data_pointer sdata;
  REQUIRE_NOTHROW( sdata = local_value->get_data(mycoord) );
  sdata.at(0) = mytid;

  shared_ptr<kernel> reduce;
  REQUIRE_NOTHROW( reduce = shared_ptr<kernel>( new mpi_kernel(local_value,reduc_value) ) );
  //  REQUIRE_NOTHROW( reduce->set_last_dependency().set_is_collective() );
  auto collected = shared_ptr<distribution>( new mpi_gathered_distribution(recomp) );
  REQUIRE_NOTHROW( reduce->set_explicit_beta_distribution(collected) );
  REQUIRE_NOTHROW( reduce->set_localexecutefn( &summing ) );
  REQUIRE_NOTHROW( reduce->analyze_dependencies() );

  vector<shared_ptr<task>> tsks; shared_ptr<task> tsk;
  REQUIRE_NOTHROW( tsks = reduce->get_tasks() );
  REQUIRE( tsks.size()>0 );
  REQUIRE_NOTHROW( tsk = tsks.at(0) );
  auto deps = tsk->get_dependencies(); int icol=0;
  for ( auto d : deps ) {
    REQUIRE_NOTHROW( icol += d.get_is_collective() );
  }
  CHECK( icol>0 );
  
  vector<shared_ptr<message>> msgs;
  REQUIRE_NOTHROW( msgs = tsk->get_receive_messages() );
  CHECK( msgs.size()==ntids );
  int count_skip{0};
  for ( const auto &m : msgs )
    count_skip += m->is_skippable();
  if (use_mpi)
    REQUIRE( count_skip==msgs.size()-1 );
  else
    REQUIRE( count_skip==0 );

  REQUIRE_NOTHROW( reduce->execute(true) );
  data_pointer rdata;
  REQUIRE_NOTHROW( rdata = reduc_value->get_data(mycoord) );
  CHECK( rdata.at(0)==Approx( ntids*(ntids-1)/2 ) );
  //  print("premature exit from [20]\n"); return;
}

TEST_CASE( "Treewise collectives","[collective][tree][25]" ) {
  architecture *aa;
  REQUIRE_NOTHROW( aa = env->make_architecture() );
  
  decomposition decomp ;
  REQUIRE_NOTHROW( decomp = mpi_decomposition(aa) );

  index_int nlocal=100,nglobal = nlocal*ntids;
  shared_ptr<distribution> alpha, local_scalar, gathered_scalar, summed_scalar;
  REQUIRE_NOTHROW
    ( alpha = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,nglobal) ) );
  REQUIRE_NOTHROW
    ( local_scalar = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,1,-1) ) );
  REQUIRE_NOTHROW
    ( gathered_scalar = shared_ptr<distribution>( new mpi_gathered_distribution(decomp) ) );
  REQUIRE_NOTHROW
    ( summed_scalar = shared_ptr<distribution>( new mpi_replicated_distribution(decomp) ) );

  shared_ptr<object> x,  y ;
  REQUIRE_NOTHROW( x = shared_ptr<object>( new mpi_object(alpha) ) );
  REQUIRE_NOTHROW( y = shared_ptr<object>( new mpi_object(alpha) ) );

  REQUIRE_NOTHROW( x->allocate() );
  REQUIRE_NOTHROW( y->allocate() );
  x->set_name("inprod-x"); y->set_name("inprod-y");

  // output object
  auto global_sum = shared_ptr<object>( new mpi_object(summed_scalar) );
  global_sum->set_name("inprod-value");

  const char *path;
  SECTION( "inner product kernel" ) {
    path = "inner product kernel";
    auto innerproduct = shared_ptr<kernel>( new mpi_innerproduct_kernel(x,y,global_sum) );
    REQUIRE_NOTHROW( innerproduct->analyze_dependencies() );
    REQUIRE_NOTHROW( innerproduct->execute() );
  }
  INFO( "path is: " << path );
}

TEST_CASE( "Inner product in three kernels","[collective][30]" ) {
  INFO( "mytid: " << mytid );

  // let's go for an irregular distribution
  vector<index_int> localsizes(ntids);
  for (int tid=0; tid<ntids; tid++)
    REQUIRE_NOTHROW( localsizes.at(tid) = 10+2*tid );
  domain_coordinate sizevector(localsizes);

  shared_ptr<distribution> alpha;
  auto
    local_scalar = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,1,-1) ),
    gathered_scalar = shared_ptr<distribution>( new mpi_gathered_distribution(decomp) ),
    summed_scalar = shared_ptr<distribution>( new mpi_replicated_distribution(decomp) );
  REQUIRE_NOTHROW
    ( alpha = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,localsizes) ) );

  index_int my_first = alpha->first_index_r(mycoord).coord(0);
  {
    index_int g = ntids*10+ntids*(ntids-1);
    CHECK( alpha->global_volume()==g );
  }
  auto x = shared_ptr<object>( new mpi_object(alpha) );
  auto y = shared_ptr<object>( new mpi_object(alpha) );
  REQUIRE_NOTHROW( x->allocate() );
  REQUIRE_NOTHROW( y->allocate() );
  x->set_name("inprod-x"); y->set_name("inprod-y");

  // set local data
  data_pointer xdata,ydata;
  REQUIRE_NOTHROW( xdata = x->get_data(mycoord) );
  for (index_int i=0; i<x->volume(mycoord); i++)
    xdata.at(i) = 2.;
  REQUIRE_NOTHROW( ydata = y->get_data(mycoord) );
  for (index_int i=0; i<y->volume(mycoord); i++)
    ydata.at(i) = my_first+i;

  // output object
  auto global_sum = shared_ptr<object>( new mpi_object(summed_scalar) );
  global_sum->set_name("inprod-value");

  const char *path;
  SECTION( "explicit intermediates" ) {
    path = "explicit intermediates";
    // intermediate objects
    shared_ptr<object> local_value;
    REQUIRE_NOTHROW( local_value = shared_ptr<object>( new mpi_object(local_scalar) ) );
    //REQUIRE_NOTHROW( gatheredvalue = new mpi_object(gathered_scalar) );

    // local product is local
    auto local_product = shared_ptr<kernel>( new mpi_kernel(x,local_value) );
    REQUIRE_NOTHROW( local_product->set_last_dependency().set_explicit_beta_distribution(x->get_distribution()) );
    REQUIRE_NOTHROW( local_product->add_in_object(y) );
    REQUIRE_NOTHROW( local_product->set_last_dependency().set_explicit_beta_distribution(y->get_distribution()) );
    local_product->set_localexecutefn( &local_inner_product );
    REQUIRE_NOTHROW( local_product->analyze_dependencies() );
    
    shared_ptr<kernel> gather_and_sum;

    // VLE can we turn this into an inner section?
    // REQUIRE_NOTHROW( gather_and_sum = shared_ptr<kernel>( new mpi_kernel(local_value,global_sum) ) );
    // REQUIRE_NOTHROW( gather_and_sum->set_explicit_beta_distribution(gathered_scalar) );
    // gather_and_sum->set_localexecutefn( &summing );

    // VLE here is how it is done in the mpi_innerproduct_kernel
    REQUIRE_NOTHROW( gather_and_sum = shared_ptr<kernel>( new mpi_reduction_kernel(local_value,global_sum) ) );

    REQUIRE_NOTHROW( gather_and_sum->analyze_dependencies() );
    REQUIRE_NOTHROW( local_product->execute() );
    REQUIRE_NOTHROW( gather_and_sum->execute() );
  }

  SECTION( "inner product kernel" ) {
    path = "inner product kernel";
    auto innerproduct = shared_ptr<kernel>( new mpi_innerproduct_kernel(x,y,global_sum) );
    REQUIRE_NOTHROW( innerproduct->analyze_dependencies() );
    REQUIRE_NOTHROW( innerproduct->execute() );
  }
  INFO( "path is: " << path );

  index_int g = domain_coordinate( alpha->global_size() ).at(0); 
  data_pointer zdata;
  REQUIRE_NOTHROW( zdata = global_sum->get_data(mycoord) );
  CHECK( global_sum->volume(mycoord)==1 );
  CHECK( zdata.at(0)==g*(g-1.) );

}

TEST_CASE( "Inner product with different strategies","[collective][31]" ) {

  architecture *aa;
  REQUIRE_NOTHROW( aa = env->make_architecture() );
  
  std::string path;
  SECTION( "point-to-point" ) { path = std::string("point-to-point");
    aa->set_collective_strategy_ptp();
  }
  SECTION( "grouping" ) { path = std::string("grouping");
    aa->set_collective_strategy_group();
  }
  SECTION( "treewise" ) { path = std::string("treewise");
    aa->set_collective_strategy_recursive();
  }
  SECTION( "collectivewise" ) { path = std::string("MPI collective");
    aa->set_collective_strategy(collective_strategy::MPI);
  }
  INFO( "mytid: " << mytid );
  INFO( "collectives are done: " << path );

  decomposition decomp ;
  REQUIRE_NOTHROW( decomp = mpi_decomposition(aa) );

  // let's go for an irregular distribution
  shared_ptr<distribution> alpha, local_scalar, gathered_scalar, summed_scalar;
  vector<index_int> localsizes(ntids);
  for (int tid=0; tid<ntids; tid++)
    REQUIRE_NOTHROW( localsizes.at(tid) = 10+2*tid );
  REQUIRE_NOTHROW
    ( alpha = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,localsizes) ) );
  REQUIRE_NOTHROW
    ( local_scalar = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,1,-1) ) );
  REQUIRE_NOTHROW
    ( gathered_scalar = shared_ptr<distribution>( new mpi_gathered_distribution(decomp) ) );
  REQUIRE_NOTHROW
    ( summed_scalar = shared_ptr<distribution>( new mpi_replicated_distribution(decomp) ) );

  index_int my_first = alpha->first_index_r(mycoord).coord(0);
  {
    index_int g = ntids*10+ntids*(ntids-1);
    CHECK( domain_coordinate( alpha->global_size() ).at(0)==g );
  }
  shared_ptr<object> x,  y ;
  REQUIRE_NOTHROW( x = shared_ptr<object>( new mpi_object(alpha) ) );
  REQUIRE_NOTHROW( y = shared_ptr<object>( new mpi_object(alpha) ) );

  REQUIRE_NOTHROW( x->allocate() );
  REQUIRE_NOTHROW( y->allocate() );
  x->set_name("inprod-x"); y->set_name("inprod-y");

  // set local data
  data_pointer xdata,ydata;
  REQUIRE_NOTHROW( xdata = x->get_data(mycoord) );
  for (index_int i=0; i<x->volume(mycoord); i++)
    xdata.at(i) = 2.;
  REQUIRE_NOTHROW( ydata = y->get_data(mycoord) );
  for (index_int i=0; i<y->volume(mycoord); i++)
    ydata.at(i) = my_first+i;
  { double s=0;
    for (int i=0; i<y->volume(mycoord); i++)
      s += xdata.at(i)*ydata.at(i);
    print("{}: local value={}\n",mycoord.as_string(),s); }

  // output object
  shared_ptr<object> global_sum;
  REQUIRE_NOTHROW( global_sum = shared_ptr<object>( new mpi_object(summed_scalar) ) );
  global_sum->set_name("inprod-value");

  shared_ptr<kernel> innerproduct ;
  REQUIRE_NOTHROW( innerproduct = shared_ptr<kernel>( new mpi_innerproduct_kernel(x,y,global_sum) ) );
  REQUIRE_NOTHROW( innerproduct->analyze_dependencies() );
  REQUIRE_NOTHROW( innerproduct->execute() );

  index_int g = domain_coordinate( alpha->global_size() ).at(0); 
  data_pointer zdata;
  REQUIRE_NOTHROW( zdata = global_sum->get_data(mycoord) );
  CHECK( global_sum->volume(mycoord)==1 );
  CHECK( zdata.at(0)==g*(g-1.) );
  print("{}\n",zdata.at(0));
}
