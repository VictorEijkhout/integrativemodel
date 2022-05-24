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
 **** application level operations
 ****
 ****************************************************************/

#include <stdlib.h>
#include <math.h>

#include "catch.hpp"

#include "mpi_base.h"
#include "mpi_ops.h"
#include "mpi_static_vars.h"
#include "unittest_functions.h"

using fmt::format;
using fmt::print;
using fmt::memory_buffer;
using fmt::format_to;
using fmt::to_string;

using std::string;

using std::make_shared;
using std::shared_ptr;
using std::vector;

TEST_CASE( "setting values","[set][01]" ) {
  index_int nlocal=10;
  shared_ptr<distribution> blocked = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,nlocal,-1) );
  auto constant = shared_ptr<object>( new mpi_object(blocked) );
  data_pointer data;
  double v=1.;
  REQUIRE_NOTHROW( constant->allocate() );
  REQUIRE_NOTHROW( constant->set_value(v) );
  CHECK( constant->volume(mycoord)==nlocal );
  REQUIRE_NOTHROW( data = constant->get_data(mycoord) );
  for (int i=0; i<nlocal; i++)
    CHECK( data.at(i)==Approx(v) );
}

TEST_CASE( "Copy kernel","[kernel][copy][31]" ) {

  INFO( "mytid: " << mytid );

  int nlocal = 1000;
  shared_ptr<distribution> blocked;
  REQUIRE_NOTHROW( blocked = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,nlocal,-1) ) );
  auto
    x = shared_ptr<object>( new mpi_object(blocked) ),
    xnew = shared_ptr<object>( new mpi_object(blocked) ),
    z = shared_ptr<object>( new mpi_object(blocked) ),
    r = shared_ptr<object>( new mpi_object(blocked) ),
    rnew = shared_ptr<object>( new mpi_object(blocked) ),
    p = shared_ptr<object>( new mpi_object(blocked) ),
    pold = shared_ptr<object>( new mpi_object(blocked) ),
    q = shared_ptr<object>( new mpi_object(blocked) ),
    qold = shared_ptr<object>( new mpi_object(blocked) );
  
  // scalars, all redundantly replicated
  shared_ptr<distribution>
    scalar = shared_ptr<distribution>( new mpi_replicated_distribution(decomp) );
  auto rr  = shared_ptr<object>( new mpi_object(scalar) );
  auto rrp = shared_ptr<object>( new mpi_object(scalar) );
  auto rnorm = shared_ptr<object>( new mpi_object(scalar) );
  auto pap = shared_ptr<object>( new mpi_object(scalar) );
  auto alpha = shared_ptr<object>( new mpi_object(scalar) );
  auto beta = shared_ptr<object>( new mpi_object(scalar) );
  
  REQUIRE_NOTHROW( rr->allocate() );
  REQUIRE_NOTHROW( rrp->allocate() );
  REQUIRE_NOTHROW( r->allocate() );
  REQUIRE_NOTHROW( z->allocate() );
  shared_ptr<kernel> rrcopy;
  int n;
  data_pointer rdata,sdata;
  
  SECTION( "scalar" ) {
    n = 1;
    rdata = rr->get_data(mycoord); sdata = rrp->get_data(mycoord);
    for (int i=0; i<n; i++)
      rdata.at(i) = 2.*(mytid*n+i);
    REQUIRE_NOTHROW( rrcopy = shared_ptr<kernel>( new mpi_copy_kernel( rr,rrp ) ) );
    CHECK( rrcopy->has_type_compute() );
  }
  SECTION( "vector" ) {
    n = nlocal;
    rdata = r->get_data(mycoord); sdata = z->get_data(mycoord);
    for (int i=0; i<n; i++)
      rdata.at(i) = 2.*(mytid*n+i);
    REQUIRE_NOTHROW( rrcopy = shared_ptr<kernel>( new mpi_copy_kernel( r,z ) ) );
  }
  
  REQUIRE_NOTHROW( rrcopy->analyze_dependencies() );
  REQUIRE_NOTHROW( rrcopy->execute() );
  for (int i=0; i<n; i++)
    CHECK( sdata.at(i)==Approx(2.*(mytid*n+i)) );
}

TEST_CASE( "Scale kernel","[task][kernel][execute][33]" ) {

  INFO( "mytid=" << mytid );

  int nlocal=12; const char *mode = "default";
  auto no_op = ioperator("none");
  shared_ptr<distribution> block = 
    shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,nlocal*ntids) );
  auto xdata = data_allocate(nlocal);
  //  auto xdata = new double[nlocal];
  auto
    xvector = shared_ptr<object>( new mpi_object(block,xdata) ),
    yvector = shared_ptr<object>( new mpi_object(block) );
  auto
    my_first = block->first_index_r(mycoord), my_last = block->last_index_r(mycoord);
  CHECK( my_first[0]==mytid*nlocal );
  CHECK( my_last[0]==(mytid+1)*nlocal-1 );
  for (int i=0; i<nlocal; i++)
    xdata->at(i) = pointfunc33(i,my_first[0]);
  shared_ptr<kernel> scale;
  data_pointer halo_data,ydata;

  SECTION( "scale by constant" ) {

    SECTION( "kernel by pieces" ) {
      mode = "constant by pieces";
      scale = shared_ptr<kernel>( new mpi_kernel(xvector,yvector) );
      scale->set_name("33scale");
      dependency &d = scale->set_last_dependency();
      REQUIRE_NOTHROW( d.set_explicit_beta_distribution(block) );

      SECTION( "constant in the function" ) {
      mode = "constant by function";
	scale->set_localexecutefn( &vecscalebytwo );
      }

      SECTION( "constant passed as context" ) {
	mode = "constant by context";
	double x = 2;
	scale->set_localexecutefn
	  ( [x] ( kernel_function_args ) -> void {
	    return vecscalebyc( kernel_function_call,x ); } );
      }

      SECTION( "constant passed inverted" ) {
	mode = "constant by inverted";
	double x = 1./2;
	scale->set_localexecutefn
	  ( [x] ( kernel_function_args ) -> void {
	    return vecscaledownbyc( kernel_function_call,x ); } );
      }
    }
    SECTION( "kernel as kernel from double-star" ) {
      mode = "scale kernel with scalar";
      double x = 2;
      scale = shared_ptr<kernel>( new mpi_scale_kernel(x,xvector,yvector) );
    }
    SECTION( "kernel as kernel from replicated object" ) {
      mode = "scale kernel with object";
      double xval = 2;
      auto scalar = shared_ptr<distribution>( new mpi_replicated_distribution(decomp) );
      auto x = shared_ptr<object>( new mpi_object(scalar) );
      REQUIRE_NOTHROW( x->set_value(xval) );
      CHECK( scalar->has_type_replicated() );
      CHECK( x->get_distribution()->has_type_replicated() );
      REQUIRE_NOTHROW( scale = shared_ptr<kernel>( new mpi_scale_kernel(x,xvector,yvector) ) );
      REQUIRE_NOTHROW( scale->analyze_dependencies() );
      vector<dependency> deps;
      REQUIRE_NOTHROW( deps = scale->get_dependencies() );
      CHECK( deps.size()==2 );
    }

    INFO( "mode is " << mode );
    REQUIRE_NOTHROW( scale->analyze_dependencies() );
    CHECK( scale->get_tasks().size()==1 );

    shared_ptr<task> scale_task;
    CHECK_NOTHROW( scale_task = scale->get_tasks().at(0) );
    CHECK_NOTHROW( scale->execute() );
    // CHECK_NOTHROW( halo_data = scale_task->get_beta_object(0)->get_data(mycoord) );
    // CHECK_NOTHROW( ydata = yvector->get_data(mycoord) );
    // {
    //   int i;
    //   for (i=0; i<nlocal; i++) {
    // 	CHECK( halo_data.at(i) == Approx( pointfunc33(i,my_first) ) );
    //   }
    // }
  }

  SECTION( "axpy kernel" ) { 
    double x=2;
    scale = shared_ptr<kernel>( new mpi_axpy_kernel(xvector,yvector,&x) );
    scale->analyze_dependencies();
    scale->execute();
  }

  INFO( "mode is " << mode );
  {
    int i;
    CHECK_NOTHROW( ydata = yvector->get_data(mycoord) );
    for (i=0; i<nlocal; i++) {
      INFO( "local iteration " << i );
      CHECK( ydata.at(i) == Approx( 2*pointfunc33(i,my_first[0])) );
    }
  }
}

TEST_CASE( "Stats kernel","[task][kernel][execute][34][hide]" ) {

  INFO( "mytid=" << mytid );
  index_int nlocal = 10;

  auto block_structure = shared_ptr<distribution>
    ( make_shared<mpi_block_distribution>(decomp,nlocal,-1) );
  auto data = shared_ptr<object>( new mpi_object(block_structure) );
  shared_ptr<kernel> setinput;

  REQUIRE_NOTHROW( setinput = shared_ptr<kernel>( new mpi_origin_kernel(data) ) );
  REQUIRE_NOTHROW( setinput->set_localexecutefn(&vecsetlinear) );
  REQUIRE_NOTHROW( setinput->analyze_dependencies() );
  REQUIRE_NOTHROW( setinput->execute() );

  auto stat_structure = shared_ptr<distribution>( new mpi_gathered_distribution(decomp) );
  auto data_stats = shared_ptr<object>( new mpi_object(stat_structure) );

  SECTION( "spell out the stats kernel" ) {
    // sum everything locally to a single scalar
    auto scalar_structure = shared_ptr<distribution>
      ( make_shared<mpi_block_distribution>(decomp,1,-1) );
    auto local_value = shared_ptr<object>( new mpi_object(scalar_structure) );
    auto local_stats = shared_ptr<kernel>( new mpi_kernel(data,local_value) );

    REQUIRE_NOTHROW( local_stats->set_localexecutefn( &summing ) );
    REQUIRE_NOTHROW( local_stats->set_explicit_beta_distribution(data->get_distribution()) );
    REQUIRE_NOTHROW( local_stats->analyze_dependencies() );
    {
      shared_ptr<task> t;
      REQUIRE_NOTHROW( t = local_stats->get_tasks().at(0) );
      auto snds = t->get_send_messages(), rcvs = t->get_receive_messages();
      CHECK( snds.size()==1 );
      CHECK( rcvs.size()==1 );
    }
    REQUIRE_NOTHROW( local_stats->execute() );
    {
      double sum=0;
      for (index_int n=0; n<nlocal; n++)
	sum += mytid*nlocal+n;
      data_pointer data;
      REQUIRE_NOTHROW( data = local_value->get_data(mycoord) );
      CHECK( data.at(0)==Approx(sum) );
    }

    auto global_stats = shared_ptr<kernel>( new mpi_kernel(local_value,data_stats) );

    REQUIRE_NOTHROW( global_stats->set_localexecutefn( &veccopy ) );
    REQUIRE_NOTHROW( global_stats->set_explicit_beta_distribution(data_stats->get_distribution()) );
    REQUIRE_NOTHROW( global_stats->analyze_dependencies() );
    CHECK( data_stats->volume(mycoord)==ntids );
    {
      shared_ptr<task> t;
      REQUIRE_NOTHROW( t = global_stats->get_tasks().at(0) );
      auto snds = t->get_send_messages(), rcvs = t->get_receive_messages();
      CHECK( snds.size()==ntids );
      CHECK( rcvs.size()==ntids );
    }
    REQUIRE_NOTHROW( global_stats->execute() );
  }

  SECTION( "actual kernel" ) {
    shared_ptr<kernel> compute_stats;
    REQUIRE_NOTHROW( compute_stats = shared_ptr<kernel>( new mpi_stats_kernel(data,data_stats,summing) ) );
    REQUIRE_NOTHROW( compute_stats->analyze_dependencies() );
    REQUIRE_NOTHROW( compute_stats->execute() );
  }

  {
    data_pointer data;
    REQUIRE_NOTHROW( data = data_stats->get_data(mycoord) );
    for (int t=0; t<ntids; t++) {
      double sum=0;
      for (index_int n=0; n<nlocal; n++)
	sum += t*nlocal+n;
      CHECK( data.at(t)==Approx(sum) );
    }
  }
}

TEST_CASE( "Beta from sparse matrix","[beta][sparse][41]" ) {

  int localsize=20,gsize=localsize*ntids;
  shared_ptr<distribution> block = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,localsize,-1) );
  auto in_obj = shared_ptr<object>( new mpi_object(block) );
  auto out_obj = shared_ptr<object>( new mpi_object(block) );
  REQUIRE_NOTHROW( in_obj->allocate() );
  {
    auto indata = in_obj->get_data(mycoord); index_int n = in_obj->volume(mycoord);
    for (index_int i=0; i<n; i++) indata.at(i) = .5;
  }
  auto kern = shared_ptr<kernel>( shared_ptr<kernel>( new mpi_kernel(in_obj,out_obj) ) );
  kern->set_name("sparse-stuff");
  shared_ptr<sparse_matrix> pattern;
  kern->set_localexecutefn
    ( [pattern] (kernel_function_args) -> void {
      return local_sparse_matrix_vector_multiply(kernel_function_call,pattern); } );

  index_int myfirst,mylast;
  REQUIRE_NOTHROW( myfirst = block->first_index_r(mycoord)[0] );
  REQUIRE_NOTHROW( mylast = block->last_index_r(mycoord)[0] );
  SECTION( "connect right" ) {
    // let's try two things that are a prereq for mpi_sparse_matrix
    REQUIRE_NOTHROW( block->proc_coord() );
    REQUIRE_NOTHROW( block->get_processor_structure(block->proc_coord()) );
    REQUIRE_NOTHROW( pattern = shared_ptr<sparse_matrix>
		     ( new mpi_sparse_matrix(block,block->global_size()[0]) ) );
    for (index_int i=myfirst; i<=mylast; i++) {
      REQUIRE_NOTHROW( pattern->add_element(i,i,1.) );
      if (i+1<gsize) {
	REQUIRE_NOTHROW( pattern->add_element(i,i+1,-1.) );
      } else {
	//REQUIRE_THROWS( pattern->add_element(i,i+1,-1.) );
      }
    }
    REQUIRE_NOTHROW( kern->set_last_dependency().set_index_pattern(pattern) );
    SECTION( "just inspect halo" ) {
      REQUIRE_NOTHROW( kern->analyze_dependencies() );
      //REQUIRE_NOTHROW( kern->set_last_dependency().create_beta_vector(out_obj->get_distribution()) );
      shared_ptr<object> halo;
      REQUIRE_NOTHROW( halo = kern->last_dependency().get_beta_object() );
      shared_ptr<multi_indexstruct> mystruct;
      REQUIRE_NOTHROW( mystruct = halo->get_processor_structure(mycoord) );
      REQUIRE( mystruct!=nullptr );
      REQUIRE( !mystruct->is_empty() );
      if (mytid<ntids-1)
	CHECK( mystruct->get_component(0)
	       ->equals( shared_ptr<indexstruct>( new contiguous_indexstruct(myfirst,mylast+1)) ) );
      else
	CHECK( mystruct->get_component(0)
	       ->equals( shared_ptr<indexstruct>( new contiguous_indexstruct(myfirst,mylast)) ) );
    }
    SECTION( "all the way" ) {
      REQUIRE_NOTHROW( kern->analyze_dependencies() );

      vector<shared_ptr<task>> tasks;
      REQUIRE_NOTHROW( tasks = kern->get_tasks() );
      CHECK( tasks.size()==1 );
      shared_ptr<task> threetask;
      REQUIRE_NOTHROW( threetask = tasks.at(0) );
      vector<shared_ptr<message>> rmsgs;
      REQUIRE_NOTHROW( rmsgs = threetask->get_receive_messages() );
      if (mytid==ntids-1) {
	CHECK( rmsgs.size()==1 );
      } else {
	CHECK( rmsgs.size()==2 );
      }
      for ( auto msg : rmsgs ) {
	if (msg->get_sender()==msg->get_receiver()) {
	  CHECK( msg->volume()==localsize );
	} else {
	  CHECK( msg->volume()==1 );
	}
      }
      //REQUIRE_NOTHROW( kern->execute() ); // VLE no local function defined, so just copy
      // auto outdata = out_obj->get_data(mycoord); index_int n = out_obj->volume(mycoord);
      // if (mytid==ntids-1) {
      //   for (index_int i=0; i<n-1; i++)
      // 	CHECK( outdata.at(i) == Approx(1.) );
      //   CHECK( outdata.at(n-1) == Approx(.5) );
      // } else {
      //   for (index_int i=0; i<n; i++)
      // 	CHECK( outdata.at(i) == Approx(1.) );
      // }
    }
  }

  SECTION( "connect left" ) {
    REQUIRE_NOTHROW
      ( pattern = shared_ptr<sparse_matrix>( new mpi_sparse_matrix(block,block->global_volume()) ) );
    auto pidx = block->get_dimension_structure(0);
    for (index_int i=pidx->first_index(mytid); i<=pidx->last_index(mytid); i++) {
      pattern->add_element(i,i);
      if (i-1>=0) {
	CHECK_NOTHROW( pattern->add_element(i,i-1) );
      } else {
	REQUIRE_THROWS( pattern->add_element(i,i-1) );
      }
    }
    REQUIRE_NOTHROW( kern->set_last_dependency().set_index_pattern(pattern) );
    REQUIRE_NOTHROW( kern->analyze_dependencies() );

    vector<shared_ptr<task>> tasks;
    REQUIRE_NOTHROW( tasks = kern->get_tasks() );
    CHECK( tasks.size()==1 );
    shared_ptr<task> threetask;
    REQUIRE_NOTHROW( threetask = tasks.at(0) );
    vector<shared_ptr<message>> rmsgs;
    REQUIRE_NOTHROW( rmsgs = threetask->get_receive_messages() );
    if (mytid==0) {
      CHECK( rmsgs.size()==1 );
    } else {
      CHECK( rmsgs.size()==2 );
    }
    //REQUIRE_NOTHROW( kern->execute() ); // VLE no local function defined, so just copy
  }

}

TEST_CASE( "Actual sparse matrix","[beta][sparse][spmvp][42]" ) {
  if (ntids==1) return; // need at least two processors
  INFO( "mytid: " << mytid );
  
  int localsize=10,gsize=localsize*ntids;
  auto block = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,localsize,-1) );
  auto in_obj = shared_ptr<object>( new mpi_object(block) );
  auto out_obj = shared_ptr<object>( new mpi_object(block) );
  REQUIRE_NOTHROW( in_obj->allocate() );
  {
    data_pointer indata;
    REQUIRE_NOTHROW( indata = in_obj->get_data(mycoord) );
    index_int n = in_obj->volume(mycoord);
    for (index_int i=0; i<n; i++)
      REQUIRE_NOTHROW( indata.at(i) = 1. );
  }
  kernel spmvp = mpi_kernel(in_obj,out_obj);
  spmvp.set_name("sparse-mvp");

  int ncols = 1;
  index_int
    mincol = block->first_index_r(mycoord)[0],
    maxcol = block->last_index_r(mycoord)[0],
    maxdom = block->domains_volume();
  CHECK( maxdom>0 );
  // initialize random
  srand((int)(mytid*(double)RAND_MAX/maxdom));

  // create a matrix with zero row sums and 1.5 excess on the diagonal: 
  // ncols elements of -1 off diagonal
  // also build up an indexstruct my_columns
  shared_ptr<sparse_matrix> mat;
  REQUIRE_NOTHROW( mat = shared_ptr<sparse_matrix>
		   ( new mpi_sparse_matrix(block,block->global_size()[0]) ) );
  index_int
    my_first = block->first_index_r(mycoord).coord(0),
    my_last = block->last_index_r(mycoord).coord(0);
  auto my_columns = shared_ptr<indexstruct>( new indexed_indexstruct() );
  for (index_int row=my_first; row<=my_last; row++) {
    // diagonal element
    REQUIRE_NOTHROW( mat->add_element(row,row,(double)ncols+1.5) );
    REQUIRE_NOTHROW( my_columns->add_element( row ) );
    for (index_int ic=0; ic<ncols; ic++) {
      index_int col, xs = (index_int) ( 1.*(localsize-1)*rand() / (double)RAND_MAX );
      CHECK( xs>=0 );
      CHECK( xs<localsize );
      if (mytid<ntids-1) col = my_last+1+xs;
      else               col = xs;
      REQUIRE( col>=0 );
      REQUIRE( col<gsize );
      if ( !((col<my_first) || ( col>my_last )) )
	printf("range error in row %lld: %lld [%lld-%lld]\n",row,col,my_first,my_last);
      REQUIRE( ((col<my_first) || ( col>my_last )) );
      if (col<mincol) mincol = col;
      if (col>maxcol) maxcol = col;
      REQUIRE_NOTHROW( mat->add_element(row,col,-1.) );
      REQUIRE_NOTHROW( my_columns->add_element(col) );
    }
    CHECK( mat->row_sum(row)==Approx(1.5) );
  }

  // test that we can find the matrix columns correctly
  shared_ptr<indexstruct> mstruct;
  REQUIRE_NOTHROW( mstruct = mat->all_local_columns() );
  REQUIRE_NOTHROW( mstruct = mstruct->convert_to_indexed() );
  {
    INFO( fmt::format("matrix columns: {} s/b {}",
		      mstruct->as_string(),my_columns->as_string()) )
    CHECK( mstruct->equals(my_columns) );
  }

  spmvp.set_last_dependency().set_index_pattern( mat );
  spmvp.set_localexecutefn
    ( [mat] ( kernel_function_args ) -> void {
      return local_sparse_matrix_vector_multiply(kernel_function_call,mat); } );
  SECTION( "analyze in bits and pieces" ) {

    REQUIRE_NOTHROW( spmvp.split_to_tasks() );

    REQUIRE_NOTHROW( spmvp.get_tasks().at(0) );
    auto t = spmvp.get_tasks().at(0);
    CHECK( !t->has_type_origin() );

    REQUIRE_NOTHROW( spmvp.get_dependencies().at(0) );
    auto &d = spmvp.get_dependencies().at(0);

    shared_ptr<object> out,in,halo;
    REQUIRE_NOTHROW( out = spmvp.get_out_object() );
    REQUIRE_NOTHROW( in = d.get_in_object() );
    REQUIRE_NOTHROW( in->get_distribution() );
    auto indistro = in->get_distribution();

    string recv_mode;
    // SECTION( "receive in pieces" ) { recv_mode = "in pieces";
    //   CHECK( !d.has_beta_object() );
    //   REQUIRE_NOTHROW( d.endow_beta_object(out) );
    //   CHECK( d.has_beta_object() );
    //   REQUIRE_NOTHROW( halo = d.get_beta_object() );
    //   REQUIRE_NOTHROW( halo->get_processor_structure(mycoord) );
    //   auto beta_block = halo->get_processor_structure(mycoord);
    //   REQUIRE_NOTHROW( halo->get_numa_structure() );
    //   auto numa_block = halo->get_numa_structure();
    //   print("{}: beta={}, numa={}\n",
    // 	    mycoord.as_string(),beta_block->as_string(),numa_block.as_string());
    //   self_treatment doself = self_treatment::INCLUDE;
    //   vector<shared_ptr<message>> msgs;
    //   auto buildup = shared_ptr<multi_indexstruct>( new multi_indexstruct(1) );
    //   auto buildup2 = shared_ptr<multi_indexstruct>( new multi_indexstruct(1) );
    //   for (int p=0; p<ntids; p++) {
    // 	auto pcoord = indistro->coordinate_from_linear(p);
    // 	shared_ptr<multi_indexstruct> pstruct,mintersect;
    // 	REQUIRE_NOTHROW( pstruct = indistro->get_processor_structure(pcoord) );
    // 	REQUIRE_NOTHROW( mintersect = beta_block->intersect(pstruct) );
    // 	REQUIRE_NOTHROW( buildup = buildup->struct_union(mintersect) ); 
    //   }
    //   INFO( format("buildup: {}",buildup->as_string()) );
    //   REQUIRE_NOTHROW( buildup = buildup->force_simplify(/* true */) );
    //   REQUIRE_NOTHROW( msgs = indistro->messages_for_segment
    // 		       ( mycoord,doself,beta_block,numa_block ) );
    //   for ( auto m : msgs ) {
    // 	m->set_in_object(in); m->set_out_object(out);
    // 	m->set_dependency_number(0); // does't matter, probably.
    //   }
    //   REQUIRE_NOTHROW( t->set_receive_messages(msgs) );
    // }
    SECTION( "receive in one" ) { recv_mode = "in one";
      REQUIRE_NOTHROW( d.endow_beta_object(out) );
      //REQUIRE_NOTHROW( d.create_beta_vector(out->get_distribution()) );
      CHECK( d.has_beta_object() );
      REQUIRE_NOTHROW( t->derive_receive_messages(true) );
    }
    INFO( format("Deriving recv messages: {}\n",recv_mode) );
    REQUIRE_NOTHROW( t->derive_send_messages(true) );
  }
  print("premature return from [42]\n"); return;
  SECTION( "analyze and execute and test" ) {
    printf("premature return from 42\n");
    return;
    REQUIRE_NOTHROW( spmvp.analyze_dependencies() );
    printf("premature return from 42\n");
    return;

    vector<shared_ptr<task>> tasks;
    REQUIRE_NOTHROW( tasks = spmvp.get_tasks() );
    CHECK( tasks.size()==1 );
    shared_ptr<task> spmvptask;
    REQUIRE_NOTHROW( spmvptask = tasks.at(0) );
    vector<shared_ptr<message>> rmsgs;
    REQUIRE_NOTHROW( rmsgs = spmvptask->get_receive_messages() );
    for ( auto msg : rmsgs ) {
      if (!(msg->get_sender()==msg->get_receiver())) {
	if (mytid==ntids-1)
	  CHECK( msg->get_sender().coord(0)==0 );
	else 
	  CHECK( msg->get_sender().coord(0)==mytid+1 );
      }
    }

    {
      shared_ptr<distribution> beta_dist;
      REQUIRE_NOTHROW( beta_dist = spmvp.last_dependency().get_beta_object()->get_distribution() );
      shared_ptr<indexstruct> column_indices;
      REQUIRE_NOTHROW( column_indices = 
		       beta_dist->get_dimension_structure(0)->get_processor_structure(mytid) );
    }

    {
      vector<shared_ptr<task>> tasks;
      REQUIRE_NOTHROW( tasks = spmvp.get_tasks() );
      CHECK( tasks.size()==1 );
      shared_ptr<task> threetask;
      REQUIRE_NOTHROW( threetask = tasks.at(0) );
    }
    REQUIRE_NOTHROW( spmvp.execute() );

    {
      auto data = out_obj->get_data(mycoord);
      index_int lsize = out_obj->volume(mycoord);
      for (index_int i=0; i<lsize; i++) {
	INFO( "local i=" << i );
	CHECK( data.at(i) == Approx(1.5) );
      }
    }
  }
}

TEST_CASE( "Sparse matrix kernel","[beta][sparse][spmvp][hide][43]" ) {
  if (ntids==1) { printf("test 43 is multi-processor\n"); return; };

  INFO( fmt::format("{}",mycoord.as_string()) );
  int dim=1, localsize=100, gsize=localsize*ntids;
  INFO( "[43] sparse matrix example with " << localsize << " points local" );
  printf("ex 43 fails when localsize=200\n");
  shared_ptr<distribution> block = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,localsize,-1) );
  auto in_obj = shared_ptr<object>( new mpi_object(block) );
  auto out_obj = shared_ptr<object>( new mpi_object(block) );
  REQUIRE_NOTHROW( in_obj->allocate() );
  {
    auto indata = in_obj->get_data(mycoord); index_int n = in_obj->volume(mycoord);
    for (index_int i=0; i<n; i++) indata.at(i) = 1.;
  }

  // create a matrix with zero row sums, shift diagonal up
  double dd = 1.5;
  auto mat = shared_ptr<sparse_matrix>( new mpi_sparse_matrix(block) );
  index_int
    my_first = block->first_index_r(mycoord).coord(0),
    my_last = block->last_index_r(mycoord).coord(0),
    mincol,maxcol;
  int skip=0;
  // initialize random
  srand((int)(mytid*(double)RAND_MAX/block->domains_volume()));
  auto neighbr = shared_ptr<indexstruct>( new contiguous_indexstruct(0,localsize-1) );
  if (mytid<ntids-1)
    neighbr = neighbr->operate( shift_operator(my_last+1) );
  INFO( fmt::format("neighbour connection will be limited to {}",neighbr->as_string()) );
  for (index_int row=my_first; row<=my_last; row++) {
    int ncols = 4;
    auto nzcol = shared_ptr<indexstruct>( new indexed_indexstruct() );
    for (index_int ic=0; ic<ncols; ic++) {
      index_int col,oops_col,
	col_shift = (index_int) ( 1.*(localsize/2-1)*rand() / (double)RAND_MAX );
      CHECK( col_shift>=0 );
      CHECK( col_shift<localsize );
      if (mytid<ntids-1) { col = my_last+1+col_shift; oops_col = my_last+localsize; }
      else               { col = col_shift;           oops_col = localsize-1; }
      if (ic==0) {
	mincol=col; maxcol = col; }
      while (nzcol->contains_element(col))
	col++; // make sure we hit every column just once
      REQUIRE( col<oops_col );
      if ( (mytid<ntids-1 && col>=my_last+localsize)
	   || (mytid==ntids-1 && col>=localsize) ) { // can't fit this in the next proc
	skip++;
      } else {
	nzcol->add_element(col);
	if (col<mincol) mincol = col;
	if (col>maxcol) maxcol = col;
	REQUIRE_NOTHROW( mat->add_element(row,col,-1.) ); // off elt
      }
    }
    INFO( fmt::format("Elements added: {}",nzcol->as_string()) );
    CHECK( neighbr->contains(nzcol) );
    REQUIRE_NOTHROW( mat->add_element(row,row,ncols+dd-skip) ); // diag elt
  }

  shared_ptr<kernel> spmvp;
  REQUIRE_NOTHROW( spmvp = shared_ptr<kernel>( new mpi_spmvp_kernel(in_obj,out_obj,mat) ) );
  REQUIRE_NOTHROW( spmvp->analyze_dependencies() );

  vector<shared_ptr<task>> tasks;
  REQUIRE_NOTHROW( tasks = spmvp->get_tasks() );
  CHECK( tasks.size()==1 );
  shared_ptr<task> spmvptask;
  REQUIRE_NOTHROW( spmvptask = tasks.at(0) );
  vector<shared_ptr<message>> rmsgs;
  REQUIRE_NOTHROW( rmsgs = spmvptask->get_receive_messages() );
  for ( auto msg : rmsgs ) {
    auto sender = msg->get_sender(), receiver = msg->get_receiver();
    INFO( "receive message " << sender.as_string() << "->" << receiver.as_string() );
    if (!(sender==receiver)) {
      if (mytid==ntids-1) {
	// auto proc0 = processor_coordinate_zero(dim);
	// CHECK( *sender==proc0 );
	CHECK( sender.coord(0)==0 );
      } else {
	// auto procn = mycoord+1;
	// CHECK( *sender==procn );
	CHECK( sender.coord(0)==mytid+1 );
      }
    }
  }
  vector<shared_ptr<message>> smsgs;
  REQUIRE_NOTHROW( smsgs = spmvptask->get_send_messages() );
  for ( auto msg : smsgs ) {
    auto sender = msg->get_sender(), receiver = msg->get_receiver();
    INFO( "receive message " << sender.as_string() << "->" << receiver.as_string() );
    if (!(sender==receiver)) {
      if (mytid==0)
	CHECK( receiver.coord(0)==ntids-1 );
      else 
	CHECK( receiver.coord(0)==mytid-1 );
    }
  }

  {
    shared_ptr<distribution> beta_dist;
    REQUIRE_NOTHROW( beta_dist = spmvp->last_dependency().get_beta_object()->get_distribution() );
    shared_ptr<indexstruct> column_indices;
    REQUIRE_NOTHROW( column_indices =
		 beta_dist->get_dimension_structure(0)->get_processor_structure(mytid) );
    //    printf("column_indices on %d: %s\n",mytid,column_indices->as_string());
    //    CHECK( column_indices->is_sorted() ); 
  }

  REQUIRE_NOTHROW( spmvp->execute() );

  {
    auto data = out_obj->get_data(mycoord);
    index_int lsize = out_obj->volume(mycoord);
    for (index_int i=0; i<lsize; i++) {
      CHECK( data.at(i) == Approx(dd) );
    }
  }
}

TEST_CASE( "matrix kernel analysis","[kernel][spmvp][sparse][44]" ) {
  INFO( "mytid=" << mytid );
  int nlocal = 10, g = ntids*nlocal;
  shared_ptr<distribution> blocked =
    shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,g) );
  auto x = shared_ptr<object>( new mpi_object(blocked) ),
    y = shared_ptr<object>( new mpi_object(blocked) );
  REQUIRE_NOTHROW( x->allocate() );

  // set the matrix to one lower diagonal
  auto Aup = shared_ptr<sparse_matrix>( new mpi_sparse_matrix( blocked ) );
  {
    index_int
      globalsize = domain_coordinate( blocked->global_size() ).at(0),
      my_first = blocked->first_index_r(mycoord).coord(0),
      my_last = blocked->last_index_r(mycoord).coord(0);
    for (index_int row=my_first; row<=my_last; row++) {
      int col;
      // A narrow is tridiagonal
      col = row;   Aup->add_element(row,col,1.);
      col = row-1; if (col>=0) Aup->add_element(row,col,1.);
    }
  }

  shared_ptr<kernel> k;
  REQUIRE_NOTHROW( k = shared_ptr<kernel>( new mpi_spmvp_kernel( x,y,Aup ) ) );
  REQUIRE_NOTHROW( k->analyze_dependencies() );

  // analyze the message structure
  auto tasks = k->get_tasks();
  CHECK( tasks.size()==1 );
  for (auto t=tasks.begin(); t!=tasks.end(); ++t) {
    if ((*t)->get_step()==x->get_object_number()) {
      CHECK( (*t)->get_dependencies().size()==0 );
    } else {
      auto send = (*t)->get_send_messages();
      if (mytid==ntids-1)
	CHECK( send.size()==1 );
      else
	CHECK( send.size()==2 );
      auto recv = (*t)->get_receive_messages();
      if (mytid==0)
	CHECK( recv.size()==1 );
      else
	CHECK( recv.size()==2 );
    }
  }

  SECTION( "limited to proc 0" ) {
    // set the input vector to delta on the first element
    {
      auto d = x->get_data(mycoord);
      for (index_int i=0; i<nlocal; i++) d.at(i) = 0.;
      if (mytid==0) d.at(0) = 1.;
    }

    // check that we get a nicely propagating wave
    REQUIRE_NOTHROW( k->execute() );
    index_int my_first = blocked->first_index_r(mycoord).coord(0);
    auto d = y->get_data(mycoord);
    for (index_int i=0; i<nlocal; i++) {
      index_int g = my_first+i;
      INFO( "global index " << g );
      if (g<2) // s=0, g=0,1
	CHECK( d.at(i)!=Approx(0.) );
      else
	CHECK( d.at(i)==Approx(0.) );
    }
  }
  SECTION( "crossing over" ) {
    // set the input vector to delta on the right edge
    {
      auto xdata = x->get_data(mycoord);
      for (index_int i=0; i<nlocal; i++) xdata.at(i) = 0.;
      xdata.at(nlocal-1) = 1.;
    }

    // check that we get a nicely propagating wave
    REQUIRE_NOTHROW( k->execute() );
    index_int my_first = blocked->first_index_r(mycoord).coord(0);
    auto xdata = y->get_data(mycoord);
    for (index_int i=1; i<nlocal-1; i++)
      CHECK( xdata.at(i)==Approx(0.) );
    if (mytid==0)
      CHECK( xdata.at(0)==Approx(0.) );
    else 
      CHECK( xdata.at(0)==Approx(1.) );
    CHECK( xdata.at(nlocal-1)==Approx(1.) );
  }
}

TEST_CASE( "matrix iteration shift left","[kernel][sparse][45]" ) {
  INFO( "mytid=" << mytid );
  int nlocal = 10, g = ntids*nlocal, nsteps = 2; //2*nlocal+3;
  shared_ptr<distribution> blocked =
    shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,g) );
  auto x = shared_ptr<object>( new mpi_object(blocked) );
  REQUIRE_NOTHROW( x->allocate() );

  // set the input vector to delta on the first element
  {
    auto d = x->get_data(mycoord);
    for (index_int i=0; i<nlocal; i++) d.at(i) = 0.;
    if (mytid==0) d.at(0) = 1.;
  }
  
  vector<shared_ptr<object>> y(nsteps);
  for (int i=0; i<nsteps; i++) {
    y.at(i) = shared_ptr<object>( new mpi_object(blocked) );
  }

  // set the matrix to one lower diagonal
  auto Aup = shared_ptr<sparse_matrix>( new mpi_sparse_matrix( blocked ) );
  {
    index_int
      globalsize = domain_coordinate( blocked->global_size() ).at(0),
      my_first = blocked->first_index_r(mycoord).coord(0),
      my_last = blocked->last_index_r(mycoord).coord(0);
    for (index_int row=my_first; row<=my_last; row++) {
      int col;
      col = row;   Aup->add_element(row,col,1.);
      col = row-1; if (col>=0) Aup->add_element(row,col,1.);
    }
  }

  // make a queue
  mpi_algorithm *queue = new mpi_algorithm(decomp);
  REQUIRE_NOTHROW( queue->add_kernel( shared_ptr<kernel>( new mpi_origin_kernel(x) ) ) );
  shared_ptr<object> inobj = x, outobj;
  for (int i=0; i<nsteps; i++) {
    outobj = y[i];
    shared_ptr<kernel> k;
    REQUIRE_NOTHROW( k = shared_ptr<kernel>( new mpi_spmvp_kernel( inobj,outobj,Aup ) ) );
    fmt::memory_buffer w; format_to(w,"mvp-{}",i); k->set_name( to_string(w) );
    REQUIRE_NOTHROW( queue->add_kernel(k) );
    inobj = outobj;
  }
  REQUIRE_NOTHROW( queue->analyze_dependencies() );

  // analyze the message structure
  auto tasks = queue->get_tasks();
  CHECK( tasks.size()==(nsteps+1) );
  for (auto t : tasks ) { //=tasks.begin(); t!=tasks.end(); ++t) {
    if (t->get_step()==x->get_object_number()) {
      CHECK( t->get_dependencies().size()==0 );
    } else {
      auto deps = t->get_dependencies();
      CHECK( deps.size()==1 );
      auto dep = deps.at(0);

      INFO( "step " << t->get_step() );

      vector<shared_ptr<message>> recv,send;
      REQUIRE_NOTHROW( recv = t->get_receive_messages() );
      REQUIRE_NOTHROW( send = t->get_send_messages() );

      fmt::memory_buffer w;
      format_to(w,"[{}] receiving from: ",mytid);
      for (int i=0; i<recv.size(); i++)
	REQUIRE_NOTHROW( format_to(w,"{},",recv.at(i)->get_sender().coord(0)) );
      format_to(w,". sending to: ");
      for (int i=0; i<send.size(); i++)
	REQUIRE_NOTHROW( format_to(w,"{},",send.at(i)->get_receiver().coord(0)) );

      INFO( "Send/recv: " << to_string(w) );
      if (mytid==0)
      	CHECK( recv.size()==1 );
      else
      	CHECK( recv.size()==2 );

      if (mytid==ntids-1)
      	CHECK( send.size()==1 );
      else
      	CHECK( send.size()==2 );
    }
  }

  // check that we get a nicely propagating wave
  REQUIRE_NOTHROW( queue->execute() );
  index_int my_first = blocked->first_index_r(mycoord).coord(0);
  for (int s=0; s<nsteps; s++) {
    INFO( "step " << s );
    auto d = y[s]->get_data(mycoord);
    for (index_int i=0; i<nlocal; i++) {
      index_int g = my_first+i;
      INFO( "global index " << g );
      if (g<s+2) // s=0, g=0,1
  	CHECK( d.at(i)!=Approx(0.) );
      else
  	CHECK( d.at(i)==Approx(0.) );
    }
  }
}

TEST_CASE( "special matrices","[kernel][spmvp][sparse][46]" ) {
  INFO( "mytid=" << mytid );

  int nlocal = 10, g = ntids*nlocal;
  shared_ptr<distribution> blocked =
    shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,g) );
  index_int my_first = blocked->first_index_r(mycoord).coord(0), my_last = blocked->last_index_r(mycoord).coord(0);
  shared_ptr<sparse_matrix> A;

  SECTION( "lower diagonal" ) {
    REQUIRE_NOTHROW( A = shared_ptr<sparse_matrix>( new mpi_lowerbidiagonal_matrix( blocked, 1,0 ) ) );
    if (mytid==0)
      CHECK( A->nnzeros()==2*nlocal-1 );
    else
      CHECK( A->nnzeros()==2*nlocal );

    SECTION( "kernel analysis" ) {
      auto x = shared_ptr<object>( new mpi_object(blocked) ),
	y = shared_ptr<object>( new mpi_object(blocked) );
      shared_ptr<kernel> k;
      REQUIRE_NOTHROW( k = shared_ptr<kernel>( new mpi_spmvp_kernel( x,y,A ) ) );
      REQUIRE_NOTHROW( k->analyze_dependencies() );

      // analyze the message structure
      auto tasks = k->get_tasks();
      CHECK( tasks.size()==1 );
      for (auto t : tasks ) { //=tasks.begin(); t!=tasks.end(); ++t) {
	if (t->get_step()==x->get_object_number()) {
	  CHECK( t->get_dependencies().size()==0 );
	} else {
	  auto send = t->get_send_messages();
	  if (mytid==ntids-1)
	    CHECK( send.size()==1 );
	  else
	    CHECK( send.size()==2 );
	  auto recv = t->get_receive_messages();
	  if (mytid==0)
	    CHECK( recv.size()==1 );
	  else
	    CHECK( recv.size()==2 );
	}
      }
    }
    SECTION( "run!" ) {
      auto objs = vector< shared_ptr<object> >(g);
      for (int iobj=0; iobj<g; iobj++)
	objs[iobj] = shared_ptr<object>( new mpi_object(blocked) );
      REQUIRE_NOTHROW( objs[0]->allocate() );
      data_pointer data; REQUIRE_NOTHROW( data = objs[0]->get_data(mycoord) );
      for (index_int i=0; i<nlocal; i++) data.at(i) = 0.;
      if (mytid==0) data.at(0) = 1.;
      algorithm *queue = new mpi_algorithm(decomp);
      REQUIRE_NOTHROW( queue->add_kernel( shared_ptr<kernel>( new mpi_origin_kernel(objs[0]) ) ) );
      for (int istep=1; istep<g; istep++) {
	shared_ptr<kernel> k;
	REQUIRE_NOTHROW( k = shared_ptr<kernel>( new mpi_spmvp_kernel(objs[istep-1],objs[istep],A) ) );
	REQUIRE_NOTHROW( queue->add_kernel(k) );
      }
      REQUIRE_NOTHROW( queue->analyze_dependencies() );
      REQUIRE_NOTHROW( queue->execute() );
      for (int istep=0; istep<g; istep++) {
	//INFO( "object " << istep << ": " << objs[istep]->values_as_string(mytid) );
	REQUIRE_NOTHROW( data = objs[istep]->get_data(mycoord) );
	for (index_int i=my_first; i<=my_last; i++) {
	  INFO( "index " << i );
	  if (i==istep) CHECK( data.at(i-my_first)==Approx(1.) );
	  else          CHECK( data.at(i-my_first)==Approx(0.) );
	}
      }
    }
  }

  SECTION( "upper diagonal" ) {
    REQUIRE_NOTHROW
      ( A = shared_ptr<sparse_matrix>( new mpi_upperbidiagonal_matrix( blocked, 0,1 ) ) );
    INFO( format("Matrix:\n{}\n",A->as_string()) );
    if (mytid==ntids-1)
      CHECK( A->nnzeros()==2*nlocal-1 );
    else
      CHECK( A->nnzeros()==2*nlocal );

    SECTION( "kernel analysis" ) {
      auto x = shared_ptr<object>( new mpi_object(blocked) ),
	y = shared_ptr<object>( new mpi_object(blocked) );
      shared_ptr<kernel> k;
      REQUIRE_NOTHROW( k = shared_ptr<kernel>( new mpi_spmvp_kernel( x,y,A ) ) );
      REQUIRE_NOTHROW( k->analyze_dependencies() );

      // analyze the message structure
      auto tasks = k->get_tasks();
      CHECK( tasks.size()==1 );
      for (auto t : tasks ) { //=tasks.begin(); t!=tasks.end(); ++t) {
	if (t->get_step()==x->get_object_number()) {
	  CHECK( t->get_dependencies().size()==0 );
	} else {
	  auto send = t->get_send_messages();
	  if (mytid==0)
	    CHECK( send.size()==1 );
	  else
	    CHECK( send.size()==2 );
	  auto recv = t->get_receive_messages();
	  if (mytid==ntids-1)
	    CHECK( recv.size()==1 );
	  else
	    CHECK( recv.size()==2 );
	}
      }
    }
    SECTION( "run!" ) {
      auto objs = vector<shared_ptr<object>>(g);
      for (int iobj=0; iobj<g; iobj++)
	objs[iobj] = shared_ptr<object>( new mpi_object(blocked) );
      REQUIRE_NOTHROW( objs[0]->allocate() );
      data_pointer data; REQUIRE_NOTHROW( data = objs[0]->get_data(mycoord) );
      for (index_int i=0; i<nlocal; i++) data.at(i) = 0.;
      if (mytid==ntids-1) data.at(nlocal-1) = 1.;
      algorithm *queue = new mpi_algorithm(decomp);
      REQUIRE_NOTHROW( queue->add_kernel( shared_ptr<kernel>( new mpi_origin_kernel(objs[0]) ) ) );
      for (int istep=1; istep<g; istep++) {
	shared_ptr<kernel> k;
	REQUIRE_NOTHROW( k = shared_ptr<kernel>( new mpi_spmvp_kernel(objs[istep-1],objs[istep],A) ) );
	REQUIRE_NOTHROW( queue->add_kernel(k) );
      }
      REQUIRE_NOTHROW( queue->analyze_dependencies() );
      REQUIRE_NOTHROW( queue->execute() );
      for (int istep=0; istep<g; istep++) {
	//INFO( "object " << istep << ": " << objs[istep]->values_as_string(mytid) );
	REQUIRE_NOTHROW( data = objs[istep]->get_data(mycoord) );
	for (index_int i=my_first; i<=my_last; i++) {
	  INFO( "index " << i );
	  if (i==g-1-istep) CHECK( data.at(i-my_first)==Approx(1.) );
	  else            CHECK( data.at(i-my_first)==Approx(0.) );
	}
      }
    }
  }

  SECTION( "toeplitz" ) {
    REQUIRE_NOTHROW( A = shared_ptr<sparse_matrix>( new mpi_toeplitz3_matrix( blocked, 0,2,0 ) ) );
    CHECK( A->nnzeros()==3*nlocal-(mytid==0)-(mytid==ntids-1) );

    SECTION( "kernel analysis" ) {
      auto x = shared_ptr<object>( new mpi_object(blocked) ),
	y = shared_ptr<object>( new mpi_object(blocked) );
      shared_ptr<kernel> k;
      REQUIRE_NOTHROW( k = shared_ptr<kernel>( new mpi_spmvp_kernel( x,y,A ) ) );
      REQUIRE_NOTHROW( k->analyze_dependencies() );

      // analyze the message structure
      auto tasks = k->get_tasks();
      CHECK( tasks.size()==1 );
      for (auto t : tasks ) { //=tasks.begin(); t!=tasks.end(); ++t) {
	if (t->get_step()==x->get_object_number()) {
	  CHECK( t->get_dependencies().size()==0 );
	} else {
	  auto send = t->get_send_messages();
	  CHECK( send.size()==3-(mytid==0)-(mytid==ntids-1) );
	  auto recv = t->get_receive_messages();
	  CHECK( send.size()==3-(mytid==0)-(mytid==ntids-1) );
	}
      }
    }
    SECTION( "run!" ) {
      auto objs = vector<shared_ptr<object>>(g);
      for (int iobj=0; iobj<g; iobj++)
	objs[iobj] = shared_ptr<object>( new mpi_object(blocked) );
      REQUIRE_NOTHROW( objs[0]->allocate() );
      data_pointer data; REQUIRE_NOTHROW( data = objs[0]->get_data(mycoord) );
      for (index_int i=0; i<nlocal; i++) data.at(i) = 1.;
      //      if (mytid==ntids-1) data.at(nlocal-1) = 1.;
      algorithm *queue = new mpi_algorithm(decomp);
      REQUIRE_NOTHROW( queue->add_kernel( shared_ptr<kernel>( new mpi_origin_kernel(objs[0]) ) ) );
      int nsteps = 5;
      for (int istep=1; istep<nsteps; istep++) {
	shared_ptr<kernel> k;
	REQUIRE_NOTHROW( k = shared_ptr<kernel>( new mpi_spmvp_kernel(objs[istep-1],objs[istep],A) ) );
	REQUIRE_NOTHROW( queue->add_kernel(k) );
      }
      REQUIRE_NOTHROW( queue->analyze_dependencies() );
      REQUIRE_NOTHROW( queue->execute() );

      REQUIRE_NOTHROW( data = objs[nsteps-1]->get_data(mycoord) );
      for (index_int i=my_first; i<=my_last; i++) {
	INFO( "index " << i );
	CHECK( data.at(i-my_first)==Approx(pow(2,nsteps-1)) );
      }
    }
  }
}

TEST_CASE( "central difference matrices","[kernel][47]" ) {
  INFO( "mytid=" << mytid );

  int nlocal = 1, g = ntids*nlocal;
  shared_ptr<distribution> blocked =
    shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,g) );
  auto x = shared_ptr<object>( new mpi_object(blocked) ),
    y = shared_ptr<object>( new mpi_object(blocked) );
  REQUIRE_NOTHROW( x->allocate() );
  REQUIRE_NOTHROW( x->set_value(1.) );
  shared_ptr<kernel> diff;
  REQUIRE_NOTHROW( diff = shared_ptr<kernel>( new mpi_centraldifference_kernel(x,y) ) );
  REQUIRE_NOTHROW( diff->analyze_dependencies() );
  auto tsks = diff->get_tasks();
  CHECK( tsks.size()==1 );
  auto tsk = tsks.at(0);
  auto msgs = tsk->get_receive_messages();
  if (mytid==0 || mytid==ntids-1)
    CHECK( msgs.size()==1+(ntids>1) );
  else
    CHECK( msgs.size()==1+(ntids>1)+(ntids>2) );
  REQUIRE_NOTHROW( diff->execute() );
  data_pointer data;
  REQUIRE_NOTHROW( data = y->get_data(mycoord) );
  printf("ops 47 premature end because of data problem\n");
  return;
  // global left end point
  int e = 0;
  if (mytid==0) {
    CHECK( data.at(0)==Approx(1.) ); e++;
  } else if (mytid==ntids-1 && nlocal==1) {
    CHECK( data.at(0)==Approx(1.) );
  } else 
    CHECK( data.at(0)==Approx(0.) );
  // global right endpoint
  if (mytid==ntids-1) {
    CHECK( data.at(nlocal-1)==Approx(1.) ); e++;
  } else if (mytid==0 && nlocal==1) {
    CHECK( data.at(nlocal-1)==Approx(1.) );
  } else 
    CHECK( data.at(nlocal-1)==Approx(0.) );
  // interior
  for (int i=1; i<nlocal-1; i++) {
    CHECK( data.at(i)==Approx(0.) );
  }
  // consistency check on end points
  CHECK( e==(mytid==0)+(mytid==ntids-1) );
  shared_ptr<distribution> scalar = shared_ptr<distribution>( new mpi_replicated_distribution(decomp) );
  auto norm = shared_ptr<object>( new mpi_object(scalar) );
  auto take_norm = shared_ptr<kernel>( shared_ptr<kernel>( new mpi_norm_kernel(y,norm) ) );
  take_norm->analyze_dependencies();
  take_norm->execute();
  REQUIRE_NOTHROW( data = norm->get_data(mycoord) );
  CHECK( data.at(0)==Approx(sqrt(2.)) );
}

TEST_CASE( "central difference as in CG","[kernel][48]" ) {

  index_int nlocal = 5;

  // architecture *aa; decomposition decomp;
  // REQUIRE_NOTHROW( aa = env->make_architecture() );
  // int can_embed = 0;
  // REQUIRE_NOTHROW( aa->set_can_embed_in_beta(can_embed) );
  // decomposition *ddecomp = decomposition(aa);

  decomposition ddecomp = decomposition(decomp);

  const char *path;
  SECTION( "ptp" ) { path = "ptp";
    ddecomp.set_collective_strategy( collective_strategy::ALL_PTP );
  }
  // SECTION( "group" ) { path = "group";
  //   ddecomp.set_collective_strategy( collective_strategy::GROUP );
  // }
  // SECTION( "recursive" ) { path = "recursive";
  //   ddecomp.set_collective_strategy( collective_strategy::RECURSIVE );
  // }
  // SECTION( "mpi" ) { path = "mpi";
  //   ddecomp.set_collective_strategy( collective_strategy::MPI );
  // }
  INFO( "collective strategy: " << path );

  // a bunch of vectors, block distributed
  shared_ptr<distribution> blocked = shared_ptr<distribution>( make_shared<mpi_block_distribution>(ddecomp,nlocal,-1) );
  shared_ptr<object> xt,b0,r0,ax0;
  xt    = shared_ptr<object>( new mpi_object(blocked) );
  xt->set_name(fmt::format("xtrue"));
  xt->allocate(); //xt->set_value(1.);
  b0    = shared_ptr<object>( new mpi_object(blocked) );
  b0->set_name(fmt::format("b0"));


  // scalars, all redundantly replicated
  shared_ptr<distribution> scalar = shared_ptr<distribution>( new mpi_replicated_distribution(ddecomp) );
  auto rr0 = shared_ptr<object>( new mpi_object(scalar) );

  // let's define the steps of the loop body
  algorithm *cg = new mpi_algorithm(ddecomp);
  cg->set_name("Conjugate Gradients Method");

  { auto xorigin = shared_ptr<kernel>( new mpi_origin_kernel( xt ) );
    xorigin->set_localexecutefn(&vecsetlinear);
    cg->add_kernel(xorigin); xorigin->set_name("origin xtrue");
  }

  auto matvec = shared_ptr<kernel>( new mpi_centraldifference_kernel( xt,b0 ) );
  cg->add_kernel(matvec); matvec->set_name("b0=A xtrue");

  {
    auto r0inp = shared_ptr<kernel>( new mpi_norm_kernel( b0,rr0 ) );
    cg->add_kernel( r0inp );
  }
  REQUIRE_NOTHROW( cg->analyze_dependencies() );

  // get rolling.....
  REQUIRE_NOTHROW( cg->execute() );

  // something weird going on with the matrix-vector product
  vector<shared_ptr<task>> tsks; REQUIRE_NOTHROW( tsks = matvec->get_tasks() );
  CHECK( tsks.size()==1 );
  auto tsk = tsks.at(0);
  vector<shared_ptr<message>> msgs;
  // check send messages
  REQUIRE_NOTHROW( msgs = tsk->get_send_messages() );
  if (mytid==0 || mytid==ntids-1) {
    CHECK( msgs.size()==2 );
  } else {
    CHECK( msgs.size()==3 );
  }
  auto myfirst = blocked->first_index_r(mycoord),
    mylast = blocked->last_index_r(mycoord);
  for ( auto m : msgs ) {
    INFO( fmt::format("send message: {}",m->as_string()) );
    CHECK( m->get_sender()==mycoord );
    processor_coordinate other;
    REQUIRE_NOTHROW( other = m->get_receiver() );
    auto left = mycoord-1, right = mycoord+1;
    shared_ptr<multi_indexstruct> global_struct,local_struct;
    CHECK_NOTHROW( global_struct = m->get_global_struct() );
    CHECK_NOTHROW( local_struct = m->get_local_struct() );
    if (other==mycoord) {
      CHECK( global_struct->volume()==nlocal );
      CHECK( local_struct->volume()==nlocal );
      CHECK( global_struct->first_index_r()==myfirst );
      CHECK( global_struct->last_index_r()==mylast );
    } else if (other==left) {
      CHECK( global_struct->volume()==1 );
      CHECK( local_struct->volume()==1 );
      // we send the first element
      CHECK( global_struct->first_index_r()==myfirst );
      CHECK( global_struct->last_index_r()==myfirst );
    } else if (other==right) {
      CHECK( global_struct->volume()==1 );
      CHECK( local_struct->volume()==1 );
      // we send the last element
      CHECK( global_struct->first_index_r()==mylast );
      CHECK( global_struct->last_index_r()==mylast );
    } else
      throw(fmt::format("{}: strange receiver {}",mycoord.as_string(),other.as_string()));
  }
  // check receive messages
  REQUIRE_NOTHROW( msgs = tsk->get_receive_messages() );
  if (mytid==0 || mytid==ntids-1) {
    CHECK( msgs.size()==2 );
  } else {
    CHECK( msgs.size()==3 );
  }
  for ( auto m : msgs ) {
    INFO( fmt::format("receive message: {}",m->as_string()) );
    CHECK( m->get_receiver()==mycoord );
    processor_coordinate other;
    REQUIRE_NOTHROW( other = m->get_sender() );
    auto left = mycoord-1, right = mycoord+1;
    shared_ptr<multi_indexstruct> global_struct,local_struct;
    CHECK_NOTHROW( global_struct = m->get_global_struct() );
    CHECK_NOTHROW( local_struct = m->get_local_struct() );
    if (other==mycoord) {
      CHECK( global_struct->volume()==nlocal );
      CHECK( local_struct->volume()==nlocal );
      CHECK( global_struct->first_index_r()==myfirst );
      CHECK( global_struct->last_index_r()==mylast );
    } else if (other==left) {
      CHECK( global_struct->volume()==1 );
      CHECK( local_struct->volume()==1 );
      // we receive the first element
      auto leftbnd = myfirst-1;
      CHECK( global_struct->first_index_r()==leftbnd );
      CHECK( global_struct->last_index_r()==leftbnd );
    } else if (other==right) {
      CHECK( global_struct->volume()==1 );
      CHECK( local_struct->volume()==1 );
      // we receive the last element
      auto rightbnd = mylast+1;
      CHECK( global_struct->first_index_r()==rightbnd );
      CHECK( global_struct->last_index_r()==rightbnd );
    } else
      throw(fmt::format("{}: strange receiver {}",mycoord.as_string(),other.as_string()));
  }

  // initial vector is linear
  { data_pointer xdata;
    REQUIRE_NOTHROW( xdata = xt->get_data(mycoord) );
    for (int i=0; i<nlocal; i++)
      CHECK( xdata.at(i)==Approx(mytid*nlocal+i) );
  }

  // halo should be linear
  { data_pointer xdata; shared_ptr<object> mathalo; index_int hsize,hfirst;
    REQUIRE_NOTHROW( mathalo = matvec->get_beta_object(0) );
    REQUIRE_NOTHROW( xdata = mathalo->get_data(mycoord) );
    REQUIRE_NOTHROW( hsize = mathalo->volume(mycoord) );
    REQUIRE_NOTHROW( hfirst = mathalo->first_index_r(mycoord)[0] );
    for ( index_int i=0; i<hsize; i++) {
      INFO( fmt::format("{} halo @{}+{} = {}",mycoord.as_string(),hfirst,i,xdata.at(i)) );
      CHECK( xdata.at(i)==Approx(hfirst+i) );
    }
    // mvp only 1 at the ends
    data_pointer bdata;
    REQUIRE_NOTHROW( bdata = b0->get_data(mycoord) );
    for (int i=0; i<nlocal; i++) {
      INFO( "p=" << mytid << ", i=" << i );
      if (mytid==0 && i==0)
	CHECK( bdata.at(i)==Approx(-1.) );
      else if (mytid==ntids-1 && i==nlocal-1)
	CHECK( bdata.at(i)==Approx(ntids*nlocal) );
      else
	CHECK( bdata.at(i)==Approx(0.) );
    }
  }

  // norm is sqrt(2)
  { data_pointer ndata;
    REQUIRE_NOTHROW( ndata = rr0->get_data(mycoord) );
    CHECK( ndata.at(0)==Approx(sqrt( ntids*nlocal*ntids*nlocal + 1 )) );
  }

}

TEST_CASE( "compound kernel queue","[kernel][queue][50]" ) {

  // this test is based on the old grouping strategy
  architecture *aa;
  REQUIRE_NOTHROW( aa = env->make_architecture() );
  aa->set_collective_strategy_group();
  decomposition decomp = mpi_decomposition(aa);

  int nlocal = 50;
  auto block = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,nlocal*ntids) ),
    scalar = shared_ptr<distribution>( new mpi_replicated_distribution(decomp) );
  auto x = shared_ptr<object>( new mpi_object(block) ),
    y = shared_ptr<object>( new mpi_object(block) ),
    xy = shared_ptr<object>( new mpi_object(scalar) );
  auto
    makex = shared_ptr<kernel>( new mpi_origin_kernel(x) ),
    makey = shared_ptr<kernel>( new mpi_origin_kernel(y) ),
    prod = shared_ptr<kernel>( new mpi_innerproduct_kernel(x,y,xy) );
  algorithm *queue = new mpi_algorithm(decomp);
  int inprod_step;

  SECTION( "analyze in steps" ) {

    const char *path;
    SECTION( "kernels in logical order" ) {
      path = "in order";
      REQUIRE_NOTHROW( queue->add_kernel(makex) );
      REQUIRE_NOTHROW( queue->add_kernel(makey) );
      REQUIRE_NOTHROW( queue->add_kernel(prod) );
      inprod_step = 2;
    }
  
    SECTION( "kernels in wrong order" ) {
      path = "reversed";
      REQUIRE_NOTHROW( queue->add_kernel(prod) );
      REQUIRE_NOTHROW( queue->add_kernel(makex) );
      REQUIRE_NOTHROW( queue->add_kernel(makey) );
      inprod_step = 0;
    }
    INFO( "kernels were added: " << path );

    vector<shared_ptr<task>> tasks;
    vector<shared_ptr<kernel>> kernels;
    REQUIRE_NOTHROW( kernels = queue->get_kernels() );
    CHECK( kernels.size()==3 );
    for (int ik=0; ik<kernels.size(); ik++) {
      shared_ptr<kernel> k;
      REQUIRE_NOTHROW( k = kernels.at(ik) );
      REQUIRE_NOTHROW( k->analyze_dependencies() );
      REQUIRE_NOTHROW( tasks = k->get_tasks() );
      if (ik==inprod_step) 
	CHECK( tasks.size()==3 ); // this depends on how many tasks the reduction contributes
      else 
	CHECK( tasks.size()==1 );
      REQUIRE_NOTHROW( queue->add_kernel_tasks_to_queue(k) );
    }
    
    REQUIRE_NOTHROW( tasks = queue->get_tasks() );
    for (auto t : tasks ) {
      if (!t->has_type_origin()) {
	shared_ptr<object> in; int inn; int ostep;
	CHECK_NOTHROW( in = t->last_dependency().get_in_object() );
	CHECK_NOTHROW( inn = in->get_object_number() );
	CHECK( inn>=0 );
      }
    }
  }
  SECTION( "single analyze call" ) {
    REQUIRE_NOTHROW( queue->add_kernel(makex) );
    REQUIRE_NOTHROW( queue->add_kernel(makey) );
    REQUIRE_NOTHROW( queue->add_kernel(prod) );
    REQUIRE_NOTHROW( queue->analyze_dependencies() );
    inprod_step = 2;
  }
  vector<shared_ptr<task>> tsks;
  REQUIRE_NOTHROW( tsks = queue->get_tasks() );
  for (auto t : tsks ) {
    int ik;
    REQUIRE_NOTHROW( ik = t->get_step() );
    if (ik==inprod_step) {
      REQUIRE_NOTHROW( t->get_n_in_objects()==2 );
    } else {
      REQUIRE_NOTHROW( t->get_n_in_objects()==0 );
    }
  }
}

TEST_CASE( "r_norm1, different ways","[kernel][collective][55]" ) {

  INFO( "mytid: " << mytid );

  int
    P = env->get_architecture()->nprocs(), g=P-1;
  const char *mode;

  data_pointer data;
  shared_ptr<distribution> local_scalar = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,1,-1) );
  auto local_value = shared_ptr<object>( new mpi_object(local_scalar) );
  REQUIRE_NOTHROW( local_value->allocate() );
  REQUIRE_NOTHROW( data = local_value->get_data(mycoord) );
  data.at(0) = (double)mytid;

  shared_ptr<distribution> replicated;
  REQUIRE_NOTHROW( replicated = shared_ptr<distribution>( new mpi_replicated_distribution(decomp) ) );
  shared_ptr<object> global_sum;
  REQUIRE_NOTHROW( global_sum = shared_ptr<object>( new mpi_object(replicated) ) );

  SECTION( "spell out the way to the top" ) {
    mode = "spell it out";
    int
      groupsize = MIN(4,P), mygroupnum = mytid/groupsize, nfullgroups = ntids/groupsize,
      grouped_tids = nfullgroups*groupsize, // how many procs are in perfect groups?
      remainsize = P-grouped_tids, ngroups = nfullgroups+(remainsize>0);
    CHECK( remainsize<groupsize );
    shared_ptr<distribution> locally_grouped;
    { // every proc gets to know all the indices of its group
      parallel_structure groups;
      REQUIRE_NOTHROW( groups = parallel_structure(decomp) );
      for (int p=0; p<P; p++) {
	processor_coordinate pcoord;
	REQUIRE_NOTHROW( pcoord = decomp.coordinate_from_linear(p) );
	index_int groupnumber = p/groupsize,
	  f = groupsize*groupnumber,l=MIN(f+groupsize-1,g);
	REQUIRE( l>=f );
	shared_ptr<indexstruct> pstruct; shared_ptr<multi_indexstruct> mpstruct;
	REQUIRE_NOTHROW
	  ( pstruct = shared_ptr<indexstruct>( new contiguous_indexstruct(f,l) ) );
	REQUIRE_NOTHROW( mpstruct = shared_ptr<multi_indexstruct>
			 (new multi_indexstruct( pstruct )) );
	REQUIRE_NOTHROW( groups.set_processor_structure(pcoord,mpstruct ) );
      }
      REQUIRE_NOTHROW( groups.set_structure_type( groups.infer_distribution_type() ) );
      // this one works:
      //fmt::print("Locally grouped: each has its group <<{}>>\n",groups.as_string());
      REQUIRE_NOTHROW( locally_grouped = shared_ptr<distribution>( new mpi_distribution(groups) ) );
    }

    shared_ptr<distribution> partially_reduced;
    {
      parallel_structure
	partials = parallel_structure(decomp);
      for (int p=0; p<P; p++) {
	processor_coordinate pcoord;
	REQUIRE_NOTHROW( pcoord = decomp.coordinate_from_linear(p) );
	index_int groupnumber = p/groupsize;
	auto pstruct = shared_ptr<multi_indexstruct>
	  (new multi_indexstruct
	   ( shared_ptr<indexstruct>( new contiguous_indexstruct(groupnumber) ) ));
	REQUIRE_NOTHROW( partials.set_processor_structure(pcoord,pstruct) );
      }
      REQUIRE_NOTHROW( partials.set_structure_type( partials.infer_distribution_type() ) );
      REQUIRE_NOTHROW( partially_reduced = shared_ptr<distribution>( new mpi_distribution(partials) ) );
    }
    shared_ptr<object> partial_sums;
    REQUIRE_NOTHROW( partial_sums = shared_ptr<object>( new mpi_object(partially_reduced) ) );
    REQUIRE_NOTHROW( partial_sums->allocate() );
  
    // SECTION( "group and sum separate" ) {
    //   shared_ptr<object> local_groups;
    //   shared_ptr<kernel> partial_grouping,*local_summing_to_global;

    //   // one kernel for gathering the local values
    //   REQUIRE_NOTHROW( local_groups = new mpi_object(locally_grouped) );
    //   REQUIRE_NOTHROW( local_groups->allocate() );
    //   REQUIRE_NOTHROW( partial_grouping = shared_ptr<kernel>( new mpi_kernel(local_value,local_groups) ) );
    //   REQUIRE_NOTHROW( partial_grouping->set_localexecutefn( &veccopy ) );
    //   REQUIRE_NOTHROW( partial_grouping->set_explicit_beta_distribution(locally_grouped) );
    //   REQUIRE_NOTHROW( partial_grouping->analyze_dependencies() );
    //   REQUIRE_NOTHROW( partial_grouping->execute() );

    //   vector<shared_ptr<message>> *msgs; vector<shared_ptr<task>> tsks; data_pointer data;
    //   REQUIRE_NOTHROW( tsks = partial_grouping->get_tasks() );
    //   int nt = tsks->size(); CHECK( nt==1 );
    //   shared_ptr<task> t; REQUIRE_NOTHROW( t = tsks.at(0) );
    //   REQUIRE_NOTHROW( msgs = t->get_receive_messages() );
    //   if (mytid<grouped_tids) {
    // 	CHECK( msgs->size()==groupsize );
    //   } else {
    // 	CHECK( msgs->size()==remainsize );
    //   }
    //   REQUIRE_NOTHROW( msgs = t->get_send_messages() );
    //   CHECK( local_groups->first_index_r(mycoord).coord(0)==mygroupnum*groupsize );
    //   if (mytid<grouped_tids) {
    // 	CHECK( msgs->size()==groupsize );
    // 	CHECK( local_groups->volume(mycoord)==groupsize );
    //   } else {
    // 	CHECK( msgs->size()==remainsize );
    // 	CHECK( local_groups->volume(mycoord)==remainsize );
    //   }
    //   REQUIRE_NOTHROW( data = local_groups->get_data(mycoord) );
    //   for (index_int i=0; i<local_groups->volume(mycoord); i++) {
    // 	index_int ig = (local_groups->first_index_r(mycoord).coord(0)+i);
    // 	INFO( "ilocal=" << i << ", iglobal=" << ig );
    // 	CHECK( data.at(i)==ig );
    //   }

    //   REQUIRE_NOTHROW( local_summing_to_global = shared_ptr<kernel>( new mpi_kernel(local_groups,partial_sums) ) );
    //   //REQUIRE_NOTHROW( local_summing_to_global->set_type_local() );
    //   REQUIRE_NOTHROW( local_summing_to_global->set_explicit_beta_distribution
    // 		       (locally_grouped) );
    //   REQUIRE_NOTHROW( local_summing_to_global->set_localexecutefn( &summing ) );
    //   REQUIRE_NOTHROW( local_summing_to_global->analyze_dependencies() );
    //   REQUIRE_NOTHROW( local_summing_to_global->execute() );

    //   // duplicate code with previous section
    //   REQUIRE_NOTHROW( data = partial_sums->get_data(mycoord) );
    //   CHECK( partial_sums->volume(mycoord)==1 );
    //   index_int f = locally_grouped->first_index_r(mycoord).coord(0),
    // 	l = locally_grouped->last_index_r(mycoord).coord(0), s = (l+f)*(l-f+1)/2;
    //   CHECK( data.at(0)==s );
    // }

    SECTION( "group and sum in one" ) {
      shared_ptr<kernel> partial_summing;
      REQUIRE_NOTHROW( partial_summing = shared_ptr<kernel>( new mpi_kernel(local_value,partial_sums) ) );
      // depend on the numbers in your group
      REQUIRE_NOTHROW( partial_summing->set_explicit_beta_distribution(locally_grouped) );
      REQUIRE_NOTHROW( partial_summing->set_localexecutefn( &summing ) );
      REQUIRE_NOTHROW( partial_summing->analyze_dependencies() );

      vector<shared_ptr<message>> msgs; vector<shared_ptr<task>> tsks; data_pointer data;
      REQUIRE_NOTHROW( tsks = partial_summing->get_tasks() );
      int nt = tsks.size(); CHECK( nt==1 );
      shared_ptr<task> t; REQUIRE_NOTHROW( t = tsks.at(0) );
      //INFO( "partial summing task: " << t->as_string() );
      REQUIRE_NOTHROW( msgs = t->get_receive_messages() );
      if (mytid<grouped_tids) { // if I'm in a full group
	CHECK( msgs.size()==groupsize );
      } else {
	CHECK( msgs.size()==remainsize );
      }
      REQUIRE_NOTHROW( msgs = t->get_send_messages() );
      CHECK( partial_sums->volume(mycoord)==1 );
      if (mytid<grouped_tids) { // if I'm in a full group
	CHECK( msgs.size()==groupsize );
      } else {
	CHECK( msgs.size()==remainsize );
      }

      REQUIRE_NOTHROW( partial_summing->execute() );

      // duplicate code with previous section
      REQUIRE_NOTHROW( data = partial_sums->get_data(mycoord) );
      CHECK( partial_sums->volume(mycoord)==1 );
      index_int f = locally_grouped->first_index_r(mycoord).coord(0),
	l = locally_grouped->last_index_r(mycoord).coord(0), s = (l+f)*(l-f+1)/2;
      CHECK( data.at(0)==s );
    }

    shared_ptr<kernel> top_summing;
    REQUIRE_NOTHROW( top_summing = shared_ptr<kernel>( new mpi_kernel(partial_sums,global_sum) ) );
    parallel_structure top_beta(decomp);
    for (int p=0; p<P; p++)
      REQUIRE_NOTHROW
	( top_beta.set_processor_structure
	  (p, shared_ptr<indexstruct>( new contiguous_indexstruct(0,ngroups-1)) ) );
    REQUIRE_NOTHROW( top_beta.set_structure_type( top_beta.infer_distribution_type() ) );
    REQUIRE_NOTHROW( top_summing->set_explicit_beta_distribution
		     ( shared_ptr<distribution>( new mpi_distribution(top_beta) ) ) );
    REQUIRE_NOTHROW( top_summing->set_localexecutefn( &summing ) );
    REQUIRE_NOTHROW( top_summing->analyze_dependencies() );
    REQUIRE_NOTHROW( top_summing->execute() );
  }

  SECTION( "using the reduction kernel" ) {
    shared_ptr<kernel> sumkernel;
    SECTION( "send/recv strategy" ) {
      mode = "reduction kernel, send/recv";
      arch.set_collective_strategy_ptp();
      REQUIRE_NOTHROW( sumkernel = shared_ptr<kernel>( new mpi_reduction_kernel(local_value,global_sum) ) );
    }
    SECTION( "grouping strategy" ) {
      mode = "reduction kernel, grouping";
      arch.set_collective_strategy_group();
      REQUIRE_NOTHROW( sumkernel = shared_ptr<kernel>( new mpi_reduction_kernel(local_value,global_sum) ) );
    }
    REQUIRE_NOTHROW( sumkernel->analyze_dependencies() );
    REQUIRE_NOTHROW( sumkernel->execute() );
  }
    
  INFO( "mode: " << mode );
  data = global_sum->get_data(mycoord);
  CHECK( data.at(0)==(g*(g+1)/2) );
};

TEST_CASE( "explore the reduction kernel","[reduction][kernel][56]" ) {
  auto local_scalar = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,1,-1) ),
    global_scalar = shared_ptr<distribution>( new mpi_replicated_distribution(decomp) );
  CHECK( local_scalar->volume(mycoord)==1 );
  CHECK( local_scalar->first_index_r(mycoord).coord(0)==mytid );
  CHECK( global_scalar->volume(mycoord)==1 );
  CHECK( global_scalar->first_index_r(mycoord).coord(0)==0 );
  auto local_value = shared_ptr<object>( new mpi_object(local_scalar) ),
    global_value = shared_ptr<object>( new mpi_object(global_scalar) );
  REQUIRE_NOTHROW( local_value->allocate() );
  auto data = local_value->get_data(mycoord); data.at(0) = mytid;
  shared_ptr<kernel> reduction;

  int psqrt = sqrt(ntids);
  if (psqrt*psqrt<ntids) {
    printf("Test [56] needs square number of processors\n"); return; }

  printf("collective strategy disabled\n"); return;

  SECTION( "send/recv" ) {
    REQUIRE_NOTHROW( arch.set_collective_strategy_ptp() );
    REQUIRE_NOTHROW( reduction = shared_ptr<kernel>( new mpi_reduction_kernel(local_value,global_value) ) );
    REQUIRE_NOTHROW( reduction->analyze_dependencies() );
    //    REQUIRE_NOTHROW( reduction->gather_statistics() );
    CHECK( reduction->local_nmessages()==ntids+1 );
  }
  SECTION( "grouping" ) {
    REQUIRE_NOTHROW( arch.set_collective_strategy_group() );
    REQUIRE_NOTHROW( env->set_processor_grouping(psqrt) );
    REQUIRE_NOTHROW( reduction = shared_ptr<kernel>( new mpi_reduction_kernel(local_value,global_value) ) );
    REQUIRE_NOTHROW( reduction->analyze_dependencies() );
    //    REQUIRE_NOTHROW( reduction->gather_statistics() );
    CHECK( reduction->local_nmessages()==2*psqrt );
  }
}

shared_ptr<sparse_matrix> diffusion1d( shared_ptr<distribution> blocked,processor_coordinate mycoord)  { 
  index_int globalsize = domain_coordinate( blocked->global_size() ).at(0);
  auto A = shared_ptr<sparse_matrix>( new mpi_sparse_matrix(blocked) );
  for (int row=blocked->first_index_r(mycoord).coord(0); row<=blocked->last_index_r(mycoord).coord(0); row++) {
    int col;
    col = row;     A->add_element(row,col,2.);
    col = row+1; if (col<globalsize)
		   A->add_element(row,col,-1.);
    col = row-1; if (col>=0)
		   A->add_element(row,col,-1.);
  }
  return A;
}

TEST_CASE( "cg matvec kernels","[cg][kernel][sparse][60]" ) {

  INFO( "mytid: " << mytid );

  int nlocal = 1000;
  shared_ptr<distribution> blocked = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,nlocal,-1) );
  auto
    x = shared_ptr<object>( new mpi_object(blocked) ),
    xnew = shared_ptr<object>( new mpi_object(blocked) ),
    z = shared_ptr<object>( new mpi_object(blocked) ),
    r = shared_ptr<object>( new mpi_object(blocked) ),
    rnew = shared_ptr<object>( new mpi_object(blocked) ),
    p = shared_ptr<object>( new mpi_object(blocked) ),
    pold = shared_ptr<object>( new mpi_object(blocked) ),
    q = shared_ptr<object>( new mpi_object(blocked) ),
    qold = shared_ptr<object>( new mpi_object(blocked) );

  // scalars, all redundantly replicated
  shared_ptr<distribution> scalar = shared_ptr<distribution>( new mpi_replicated_distribution(decomp) );
  auto 
    rr  = shared_ptr<object>( new mpi_object(scalar) ),
    rrp = shared_ptr<object>( new mpi_object(scalar) ),
    rnorm = shared_ptr<object>( new mpi_object(scalar) ),
    pap = shared_ptr<object>( new mpi_object(scalar) ),
    alpha = shared_ptr<object>( new mpi_object(scalar) ),
    beta = shared_ptr<object>( new mpi_object(scalar) );
  double one = 1.;
  
  // the sparse matrix
  shared_ptr<sparse_matrix> A;
  REQUIRE_NOTHROW( A = diffusion1d(blocked,mycoord) );  
  REQUIRE_NOTHROW( r->allocate() );
  REQUIRE_NOTHROW( rr->allocate() );
  REQUIRE_NOTHROW( rrp->allocate() );
  REQUIRE_NOTHROW( z->allocate() );
  REQUIRE_NOTHROW( q->allocate() );
  REQUIRE_NOTHROW( rnorm->allocate() );
  REQUIRE_NOTHROW( alpha->allocate() );
  REQUIRE_NOTHROW( beta->allocate() );

  SECTION( "matvec" ) {
    shared_ptr<kernel> matvec;
    REQUIRE_NOTHROW( A = shared_ptr<sparse_matrix>( new mpi_sparse_matrix( blocked ) ) );

    int test;
    index_int
      my_first = blocked->first_index_r(mycoord).coord(0),
      my_last = blocked->last_index_r(mycoord).coord(0);
    SECTION( "diagonal matrix" ) {
      test = 1;
      for (index_int row=my_first; row<=my_last; row++) {
	REQUIRE_NOTHROW( A->add_element( row,row,2.0 ) );
      }
    }
    SECTION( "threepoint matrix" ) {
      test = 2;
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

    data_pointer pdata,qdata;
    REQUIRE_NOTHROW( pdata = p->get_data(mycoord) );
    for (index_int row=0; row<blocked->volume(mycoord); row++)
      pdata.at(row) = 3.;
    REQUIRE_NOTHROW( matvec = shared_ptr<kernel>( new mpi_spmvp_kernel( p,q,A ) ) );
    REQUIRE_NOTHROW( matvec->analyze_dependencies() );
    REQUIRE_NOTHROW( matvec->execute() );
    REQUIRE_NOTHROW( qdata = q->get_data(mycoord) );
    for (index_int row=my_first; row<my_last; row++) {
      index_int lrow = row-my_first;
      switch (test) {
      case 1: 
	CHECK( qdata.at(lrow)==Approx(6.) );
	break;
      case 2:
	if (row==0 || row==domain_coordinate( blocked->global_size() ).at(0)-1)
	  CHECK( qdata.at(lrow)==Approx(3.) );
	else
	  CHECK( qdata.at(lrow)==Approx(0.) );
	break;
      }
    }
  }

}

TEST_CASE( "cg norm kernels","[kernel][cg][norm][61]" ) {

  INFO( "mytid: " << mytid );

  int nlocal = 1000;
  shared_ptr<distribution> blocked = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,nlocal,-1) );
  auto
    x = shared_ptr<object>( new mpi_object(blocked) ), 
    xnew = shared_ptr<object>( new mpi_object(blocked) ),
    z = shared_ptr<object>( new mpi_object(blocked) ),
    r = shared_ptr<object>( new mpi_object(blocked) ), 
    rnew = shared_ptr<object>( new mpi_object(blocked) ),
    p = shared_ptr<object>( new mpi_object(blocked) ),
    pold = shared_ptr<object>( new mpi_object(blocked) ),
    q = shared_ptr<object>( new mpi_object(blocked) ), 
    qold = shared_ptr<object>( new mpi_object(blocked) );

  // scalars, all redundantly replicated
  shared_ptr<distribution> scalar = shared_ptr<distribution>( new mpi_replicated_distribution(decomp) );
  auto
    rr  = shared_ptr<object>( new mpi_object(scalar) ), 
    rrp = shared_ptr<object>( new mpi_object(scalar) ),
    rnorm = shared_ptr<object>( new mpi_object(scalar) ),
    pap = shared_ptr<object>( new mpi_object(scalar) ),
    alpha = shared_ptr<object>( new mpi_object(scalar) ),
    beta = shared_ptr<object>( new mpi_object(scalar) );
  double one = 1.;
  
  REQUIRE_NOTHROW( r->allocate() );
  REQUIRE_NOTHROW( rr->allocate() );
  REQUIRE_NOTHROW( rrp->allocate() );
  REQUIRE_NOTHROW( z->allocate() );
  REQUIRE_NOTHROW( q->allocate() );
  REQUIRE_NOTHROW( rnorm->allocate() );
  REQUIRE_NOTHROW( alpha->allocate() );
  REQUIRE_NOTHROW( beta->allocate() );

  SECTION( "r_norm squared" ) {
    shared_ptr<kernel> r_norm;
    auto 
      rdata = r->get_data(mycoord),
      rrdata = rnorm->get_data(mycoord);
    for (int i=0; i<nlocal; i++) {
      rdata.at(i) = 2.;
    }
    REQUIRE_NOTHROW( r_norm = shared_ptr<kernel>( new mpi_normsquared_kernel( r,rnorm ) ) );
    REQUIRE_NOTHROW( r_norm->analyze_dependencies() );
    REQUIRE_NOTHROW( r_norm->execute() );
    index_int g = r->global_volume();
    CHECK( g==nlocal*ntids );
    CHECK( rrdata.at(0)==Approx(4*g) );
  }
  
  SECTION( "r_norm1" ) {
    shared_ptr<kernel> r_norm;
    auto
      rdata = r->get_data(mycoord),
      rrdata = rnorm->get_data(mycoord);
    for (int i=0; i<nlocal; i++) {
      rdata.at(i) = 2.;
    }
    SECTION( "send/recv strategy" ) {
      arch.set_collective_strategy_ptp();
      REQUIRE_NOTHROW( r_norm = shared_ptr<kernel>( new mpi_norm_kernel( r,rnorm ) ) );
    }
    SECTION( "grouping strategy" ) {
      arch.set_collective_strategy_group();
      REQUIRE_NOTHROW( r_norm = shared_ptr<kernel>( new mpi_norm_kernel( r,rnorm ) ) );
    }
    REQUIRE_NOTHROW( r_norm->analyze_dependencies() );
    REQUIRE_NOTHROW( r_norm->execute() );
    index_int g = r->global_volume();
    CHECK( rrdata.at(0)==Approx(2*sqrt((double)g)) );
  }
  
  SECTION( "r_norm2" ) {
    shared_ptr<kernel> r_norm;
    auto
      rdata = r->get_data(mycoord),
      rrdata = rnorm->get_data(mycoord);
    for (int i=0; i<nlocal; i++) {
      rdata.at(i) = mytid*nlocal+i+1;
    }
    SECTION( "send/recv strategy" ) {
      arch.set_collective_strategy_ptp();
      REQUIRE_NOTHROW( r_norm = shared_ptr<kernel>( new mpi_norm_kernel( r,rnorm ) ) );
    }
    SECTION( "grouping strategy" ) {
      arch.set_collective_strategy_group();
      REQUIRE_NOTHROW( r_norm = shared_ptr<kernel>( new mpi_norm_kernel( r,rnorm ) ) );
    }
    REQUIRE_NOTHROW( r_norm->analyze_dependencies() );
    REQUIRE_NOTHROW( r_norm->execute() );
    index_int g = r->global_volume();
    CHECK( pow(rrdata.at(0),2)==Approx(g*(g+1)*(2*g+1)/6.) );
  }
  
}

TEST_CASE( "cg inner product kernels","[kernel][cg][inprod][62]" ) {

  INFO( "mytid: " << mytid );

  int nlocal = 1000;
  shared_ptr<distribution> blocked = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,nlocal,-1) );
  auto
    x = shared_ptr<object>( new mpi_object(blocked) ), 
    xnew = shared_ptr<object>( new mpi_object(blocked) ),
    z = shared_ptr<object>( new mpi_object(blocked) ),
    r = shared_ptr<object>( new mpi_object(blocked) ), 
    rnew = shared_ptr<object>( new mpi_object(blocked) ),
    p = shared_ptr<object>( new mpi_object(blocked) ), 
    pold = shared_ptr<object>( new mpi_object(blocked) ),
    q = shared_ptr<object>( new mpi_object(blocked) ), 
    qold = shared_ptr<object>( new mpi_object(blocked) );

  // scalars, all redundantly replicated
  shared_ptr<distribution> scalar = shared_ptr<distribution>( new mpi_replicated_distribution(decomp) );
  auto 
    rr  = shared_ptr<object>( new mpi_object(scalar) ), 
    rrp = shared_ptr<object>( new mpi_object(scalar) ),
    rnorm = shared_ptr<object>( new mpi_object(scalar) ),
    pap = shared_ptr<object>( new mpi_object(scalar) ),
    alpha = shared_ptr<object>( new mpi_object(scalar) ), 
    beta = shared_ptr<object>( new mpi_object(scalar) );
  double one = 1.;
  
  REQUIRE_NOTHROW( r->allocate() );
  REQUIRE_NOTHROW( rr->allocate() );
  REQUIRE_NOTHROW( rrp->allocate() );
  REQUIRE_NOTHROW( z->allocate() );
  REQUIRE_NOTHROW( q->allocate() );
  REQUIRE_NOTHROW( rnorm->allocate() );
  REQUIRE_NOTHROW( alpha->allocate() );
  REQUIRE_NOTHROW( beta->allocate() );

  SECTION( "rho_inprod" ) {
    const char *mode;
    shared_ptr<kernel> rho_inprod;
    auto
      rdata = r->get_data(mycoord),
      zdata = z->get_data(mycoord),
      rrdata = rr->get_data(mycoord);
    for (int i=0; i<nlocal; i++) {
      rdata.at(i) = 2.; zdata.at(i) = mytid*nlocal+i;
    }
    
    SECTION( "send/recv strategy" ) {
      mode = "reduction kernel uses send/recv";
      arch.set_collective_strategy_ptp();
      REQUIRE_NOTHROW( rho_inprod = shared_ptr<kernel>( new mpi_innerproduct_kernel( r,z,rr ) ) );
    }
    SECTION( "grouping strategy" ) {
      mode = "reduction kernel uses grouping";
      arch.set_collective_strategy_group();
      REQUIRE_NOTHROW( rho_inprod = shared_ptr<kernel>( new mpi_innerproduct_kernel( r,z,rr ) ) );
    }
    REQUIRE_NOTHROW( rho_inprod->analyze_dependencies() );
    REQUIRE_NOTHROW( rho_inprod->execute() );
    // {
    //   shared_ptr<kernel> prekernel;
    //   REQUIRE_NOTHROW( prekernel = rho_inprod->get_prekernel() );
    //   CHECK( prekernel->get_n_in_objects()==2 );
    // }
    index_int g = r->global_volume();
    CHECK( rrdata.at(0)==Approx(g*(g-1)) );
  }
  
}

TEST_CASE( "cg vector kernels","[kernel][cg][axpy][63]" ) {

  INFO( "mytid: " << mytid );

  int nlocal = 1000;
  shared_ptr<distribution> blocked = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,nlocal,-1) );
  auto
    x = shared_ptr<object>( new mpi_object(blocked) ), 
    xnew = shared_ptr<object>( new mpi_object(blocked) ),
    z = shared_ptr<object>( new mpi_object(blocked) ),
    r = shared_ptr<object>( new mpi_object(blocked) ), 
    rnew = shared_ptr<object>( new mpi_object(blocked) ),
    p = shared_ptr<object>( new mpi_object(blocked) ), 
    pold = shared_ptr<object>( new mpi_object(blocked) ),
    q = shared_ptr<object>( new mpi_object(blocked) ), 
    qold = shared_ptr<object>( new mpi_object(blocked) );

  // scalars, all redundantly replicated
  shared_ptr<distribution> scalar = shared_ptr<distribution>( new mpi_replicated_distribution(decomp) );
  auto
    rr  = shared_ptr<object>( new mpi_object(scalar) ), 
    rrp = shared_ptr<object>( new mpi_object(scalar) ),
    rnorm = shared_ptr<object>( new mpi_object(scalar) ),
    pap = shared_ptr<object>( new mpi_object(scalar) ),
    alpha = shared_ptr<object>( new mpi_object(scalar) ), 
    beta = shared_ptr<object>( new mpi_object(scalar) );
  double one = 1.;
  
  REQUIRE_NOTHROW( r->allocate() );
  REQUIRE_NOTHROW( rr->allocate() );
  REQUIRE_NOTHROW( rrp->allocate() );
  REQUIRE_NOTHROW( z->allocate() );
  REQUIRE_NOTHROW( q->allocate() );
  REQUIRE_NOTHROW( rnorm->allocate() );
  REQUIRE_NOTHROW( alpha->allocate() );
  REQUIRE_NOTHROW( beta->allocate() );

  SECTION( "copy" ) {
    shared_ptr<kernel> rrcopy;
    int n;
    data_pointer rdata,sdata;
    
    SECTION( "scalar" ) {
      n = 1;
      rdata = rr->get_data(mycoord); sdata = rrp->get_data(mycoord);
      for (int i=0; i<n; i++)
	rdata.at(i) = 2.*(mytid*n+i);
      REQUIRE_NOTHROW( rrcopy = shared_ptr<kernel>( new mpi_copy_kernel( rr,rrp ) ) );
    }
    SECTION( "vector" ) {
      n = nlocal;
      rdata = r->get_data(mycoord); sdata = z->get_data(mycoord);
      for (int i=0; i<n; i++)
	rdata.at(i) = 2.*(mytid*n+i);
      REQUIRE_NOTHROW( rrcopy = shared_ptr<kernel>( new mpi_copy_kernel( r,z ) ) );
    }
    
    REQUIRE_NOTHROW( rrcopy->analyze_dependencies() );
    REQUIRE_NOTHROW( rrcopy->execute() );
    for (int i=0; i<n; i++)
      CHECK( sdata.at(i)==Approx(2.*(mytid*n+i)) );
  }

  SECTION( "add" ) {
    shared_ptr<kernel> sum,makex,makez;
    REQUIRE_NOTHROW( makex = shared_ptr<kernel>( new mpi_origin_kernel(x) ) );
    REQUIRE_NOTHROW( makez = shared_ptr<kernel>( new mpi_origin_kernel(z) ) );
    REQUIRE_NOTHROW( sum = shared_ptr<kernel>( new mpi_sum_kernel(x,z,xnew) ) );
    algorithm *queue = new mpi_algorithm(decomp);
    SECTION("the logical way") {
      REQUIRE_NOTHROW( queue->add_kernel(makex) );
      REQUIRE_NOTHROW( queue->add_kernel(makez) );
      REQUIRE_NOTHROW( queue->add_kernel(sum) );
    }
    SECTION("to be contrary") {
      REQUIRE_NOTHROW( queue->add_kernel(sum) );
      REQUIRE_NOTHROW( queue->add_kernel(makex) );
      REQUIRE_NOTHROW( queue->add_kernel(makez) );
    }
    {
      auto xdata = x->get_data(mycoord),
	zdata = z->get_data(mycoord);
      for (index_int i=0; i<nlocal; i++) { xdata.at(i) = 1.; zdata.at(i) = 2.; }
    }
    REQUIRE_NOTHROW( queue->analyze_dependencies() );
    REQUIRE_NOTHROW( queue->execute() );
    {
      auto newdata = xnew->get_data(mycoord);
      for (index_int i=0; i<nlocal; i++)
	CHECK( newdata.at(i)==Approx(3.) );
    }
  }
  
}

TEST_CASE( "cg scalar kernels","[kernel][cg][scalar][64]" ) {

  INFO( "mytid: " << mytid );

  int nlocal = 1000;
  shared_ptr<distribution> blocked = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,nlocal,-1) );
  auto
    x = shared_ptr<object>( new mpi_object(blocked) ), 
    xnew = shared_ptr<object>( new mpi_object(blocked) ),
    z = shared_ptr<object>( new mpi_object(blocked) ),
    r = shared_ptr<object>( new mpi_object(blocked) ), 
    rnew = shared_ptr<object>( new mpi_object(blocked) ),
    p = shared_ptr<object>( new mpi_object(blocked) ), 
    pold = shared_ptr<object>( new mpi_object(blocked) ),
    q = shared_ptr<object>( new mpi_object(blocked) ), 
    qold = shared_ptr<object>( new mpi_object(blocked) );

  // scalars, all redundantly replicated
  shared_ptr<distribution> scalar = shared_ptr<distribution>( new mpi_replicated_distribution(decomp) );
  auto
    rr  = shared_ptr<object>( new mpi_object(scalar) ), 
    rrp = shared_ptr<object>( new mpi_object(scalar) ),
    rnorm = shared_ptr<object>( new mpi_object(scalar) ),
    pap = shared_ptr<object>( new mpi_object(scalar) ),
    alpha = shared_ptr<object>( new mpi_object(scalar) ), 
    beta = shared_ptr<object>( new mpi_object(scalar) );
  double one = 1.;
  
  REQUIRE_NOTHROW( r->allocate() );
  REQUIRE_NOTHROW( rr->allocate() );
  REQUIRE_NOTHROW( rrp->allocate() );
  REQUIRE_NOTHROW( z->allocate() );
  REQUIRE_NOTHROW( q->allocate() );
  REQUIRE_NOTHROW( rnorm->allocate() );
  REQUIRE_NOTHROW( alpha->allocate() );
  REQUIRE_NOTHROW( beta->allocate() );

  SECTION( "scalar kernel error catching" ) {
    shared_ptr<kernel> beta_calc;
    shared_ptr<object> rr;
    REQUIRE_NOTHROW( rr = shared_ptr<object>(make_shared<mpi_object>(blocked)) );
    // VLE why are these supposed to throw?
    // REQUIRE_THROWS
    //   ( beta_calc = shared_ptr<kernel>(make_shared<mpi_scalar_kernel>( rr,"/",rrp,beta ) ) );
    // REQUIRE_THROWS
    // ( beta_calc = shared_ptr<kernel>(make_shared<mpi_scalar_kernel>( rrp,"/",rr,beta ) ) );
    // REQUIRE_THROWS
    // ( beta_calc = shared_ptr<kernel>(make_shared<mpi_scalar_kernel>( beta,"/",rrp,rr ) ) );
  }

  SECTION( "beta=rr/rrp" ) {
    shared_ptr<kernel> beta_calc;
    CHECK_NOTHROW( rr->get_data(mycoord).at(0) = 5. );
    CHECK_NOTHROW( rrp->get_data(mycoord).at(0) = 4. );
    REQUIRE_NOTHROW( beta_calc = shared_ptr<kernel>
		     (make_shared<mpi_scalar_kernel>( rr,"/",rrp,beta ) ) );
    REQUIRE_NOTHROW( beta_calc->analyze_dependencies() );
    REQUIRE_NOTHROW( beta_calc->execute(true) );
    data_pointer bd;
    REQUIRE_NOTHROW( bd = beta->get_data(mycoord) );
    REQUIRE( bd.size()==1 );
    CHECK( bd.at(0)==Approx(1.25) );
  }

}

TEST_CASE( "cg update kernels","[kernel][cg][update][65]" ) {

  INFO( "mytid: " << mytid );

  int nlocal = 1000;
  shared_ptr<distribution> blocked = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,nlocal,-1) );
  auto
    x = shared_ptr<object>( new mpi_object(blocked) ), 
    xnew = shared_ptr<object>( new mpi_object(blocked) ),
    z = shared_ptr<object>( new mpi_object(blocked) ),
    r = shared_ptr<object>( new mpi_object(blocked) ), 
    rnew = shared_ptr<object>( new mpi_object(blocked) ),
    p = shared_ptr<object>( new mpi_object(blocked) ), 
    pold = shared_ptr<object>( new mpi_object(blocked) ),
    q = shared_ptr<object>( new mpi_object(blocked) ), 
    qold = shared_ptr<object>( new mpi_object(blocked) );

  // scalars, all redundantly replicated
  shared_ptr<distribution> scalar = shared_ptr<distribution>( new mpi_replicated_distribution(decomp) );
  auto
    rr  = shared_ptr<object>( new mpi_object(scalar) ), 
    rrp = shared_ptr<object>( new mpi_object(scalar) ),
    rnorm = shared_ptr<object>( new mpi_object(scalar) ),
    pap = shared_ptr<object>( new mpi_object(scalar) ),
    alpha = shared_ptr<object>( new mpi_object(scalar) ), 
    beta = shared_ptr<object>( new mpi_object(scalar) );
  double one = 1.;
  
  REQUIRE_NOTHROW( r->allocate() );
  REQUIRE_NOTHROW( rr->allocate() );
  REQUIRE_NOTHROW( rrp->allocate() );
  REQUIRE_NOTHROW( z->allocate() );
  REQUIRE_NOTHROW( q->allocate() );
  REQUIRE_NOTHROW( rnorm->allocate() );
  REQUIRE_NOTHROW( alpha->allocate() );
  REQUIRE_NOTHROW( beta->allocate() );

  {
    double threeval = 3.;
    auto three = shared_ptr<object>( new mpi_object(scalar) );
    REQUIRE_NOTHROW( three->allocate() );
    REQUIRE_NOTHROW( three->set_value(threeval) );
    {
      data_pointer threedata;
      REQUIRE_NOTHROW( threedata = three->get_data(mycoord) );
      CHECK( threedata.at(0)==Approx(threeval) );
    }
    shared_ptr<kernel> pupdate;
    auto
      bdata = beta->get_data(mycoord),
      zdata = z->get_data(mycoord),
      odata = pold->get_data(mycoord),
      pdata = p->get_data(mycoord);
    bdata.at(0) = 2.;
    for (int i=0; i<nlocal; i++) { // 3*2 , 2*7 : ++ = 20, -+ = 8, +- = -8, -- = -20
      zdata.at(i) = 2.; odata.at(i) = 7.;
    }
    SECTION( "pp test s1" ) {
      three = shared_ptr<object>( new mpi_object(blocked) );
      REQUIRE_THROWS( pupdate = shared_ptr<kernel>( new mpi_axbyz_kernel
							  ( '+',three,z, '+',beta,pold, p ) ) );
    }
    SECTION( "pp test s2" ) {
      beta = shared_ptr<object>( new mpi_object(blocked) );
      REQUIRE_THROWS( pupdate = shared_ptr<kernel>( new mpi_axbyz_kernel
							  ( '+',three,z, '+',beta,pold, p ) ) );
    }
    SECTION( "pp" ) {
      REQUIRE_NOTHROW( pupdate = shared_ptr<kernel>( new mpi_axbyz_kernel
							   ( '+',three,z, '+',beta,pold, p ) ) );
      REQUIRE_NOTHROW( pupdate->analyze_dependencies() );
      vector<dependency> deps;
      REQUIRE_NOTHROW( deps = pupdate->get_dependencies() );
      CHECK( deps.size()==4 );
      shared_ptr<task> tsk; REQUIRE_NOTHROW( tsk = pupdate->get_tasks().at(0) );
      auto msgs = tsk->get_receive_messages(); CHECK( msgs.size()==4 );
      for ( auto msg : msgs ) {
	INFO( "message: " << msg->as_string() );
	shared_ptr<multi_indexstruct> global,local;
	REQUIRE_NOTHROW( global = msg->get_global_struct() );
	REQUIRE_NOTHROW( local = msg->get_local_struct() );
	index_int siz; REQUIRE_NOTHROW( siz = global->volume() );
	CHECK( siz==local->volume() );
	if (siz==1) {
	} else {
	  CHECK( siz==nlocal );
	}
      }
      REQUIRE_NOTHROW( pupdate->execute() );
      for (int i=0; i<nlocal; i++)
	CHECK( pdata.at(i)==Approx(20.) );
    }
    SECTION( "mp" ) {
      REQUIRE_NOTHROW( pupdate = shared_ptr<kernel>( new mpi_axbyz_kernel
							   ( '-',three,z, '+',beta,pold, p ) ) );
      REQUIRE_NOTHROW( pupdate->analyze_dependencies() );
      REQUIRE_NOTHROW( pupdate->execute() );
      for (int i=0; i<nlocal; i++)
	CHECK( pdata.at(i)==Approx(8.) );
    }
    SECTION( "pm" ) {
      REQUIRE_NOTHROW( pupdate = shared_ptr<kernel>( new mpi_axbyz_kernel
							   ( '+',three,z, '-',beta,pold, p ) ) );
      REQUIRE_NOTHROW( pupdate->analyze_dependencies() );
      REQUIRE_NOTHROW( pupdate->execute() );
      for (int i=0; i<nlocal; i++)
	CHECK( pdata.at(i)==Approx(-8.) );
    }
    SECTION( "mm" ) {
      REQUIRE_NOTHROW( pupdate = shared_ptr<kernel>( new mpi_axbyz_kernel
							   ( '-',three,z, '-',beta,pold, p ) ) );
      REQUIRE_NOTHROW( pupdate->analyze_dependencies() );
      REQUIRE_NOTHROW( pupdate->execute() );
      for (int i=0; i<nlocal; i++)
	CHECK( pdata.at(i)==Approx(-20.) );
    }
  }
}

TEST_CASE( "cg preconditioner kernels","[kernel][cg][precon][66]" ) {

  INFO( "mytid: " << mytid );

  int nlocal = 1000;
  shared_ptr<distribution> blocked = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,nlocal,-1) );
  auto
    x = shared_ptr<object>( new mpi_object(blocked) ), 
    xnew = shared_ptr<object>( new mpi_object(blocked) ),
    z = shared_ptr<object>( new mpi_object(blocked) ),
    r = shared_ptr<object>( new mpi_object(blocked) ), 
    rnew = shared_ptr<object>( new mpi_object(blocked) ),
    p = shared_ptr<object>( new mpi_object(blocked) ), 
    pold = shared_ptr<object>( new mpi_object(blocked) ),
    q = shared_ptr<object>( new mpi_object(blocked) ), 
    qold = shared_ptr<object>( new mpi_object(blocked) );

  // scalars, all redundantly replicated
  shared_ptr<distribution> scalar = shared_ptr<distribution>( new mpi_replicated_distribution(decomp) );
  auto
    rr  = shared_ptr<object>( new mpi_object(scalar) ), 
    rrp = shared_ptr<object>( new mpi_object(scalar) ),
    rnorm = shared_ptr<object>( new mpi_object(scalar) ),
    pap = shared_ptr<object>( new mpi_object(scalar) ),
    alpha = shared_ptr<object>( new mpi_object(scalar) ), 
    beta = shared_ptr<object>( new mpi_object(scalar) );
  double one = 1.;
  
  // the sparse matrix
  shared_ptr<sparse_matrix> A;
  REQUIRE_NOTHROW( A = diffusion1d(blocked,mycoord) );  
  REQUIRE_NOTHROW( r->allocate() );
  REQUIRE_NOTHROW( rr->allocate() );
  REQUIRE_NOTHROW( rrp->allocate() );
  REQUIRE_NOTHROW( z->allocate() );
  REQUIRE_NOTHROW( q->allocate() );
  REQUIRE_NOTHROW( rnorm->allocate() );
  REQUIRE_NOTHROW( alpha->allocate() );
  REQUIRE_NOTHROW( beta->allocate() );

  SECTION( "precon" ) {
    shared_ptr<kernel> precon;
    REQUIRE_NOTHROW( precon = shared_ptr<kernel>( new mpi_preconditioning_kernel( r,z ) ) );
  }
}

TEST_CASE( "neuron kernel","[kernel][sparse][DAG][70]" ) {

  INFO( "mytid=" << mytid );
  int nlocal = 10, g = ntids*nlocal;
  shared_ptr<distribution> blocked =
    shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,g) );
  auto
    a = shared_ptr<object>( new mpi_object(blocked) ),
    b = shared_ptr<object>( new mpi_object(blocked) ),
    c1 = shared_ptr<object>( new mpi_object(blocked) ), 
    c2 = shared_ptr<object>( new mpi_object(blocked) ),
    d = shared_ptr<object>( new mpi_object(blocked) );

  shared_ptr<sparse_matrix>
    Anarrow = shared_ptr<sparse_matrix>( new mpi_sparse_matrix( blocked ) ),
    Awide   = shared_ptr<sparse_matrix>( new mpi_sparse_matrix( blocked ) );
  {
    index_int
      globalsize = domain_coordinate( blocked->global_size() ).at(0),
      my_first = blocked->first_index_r(mycoord).coord(0),
      my_last = blocked->last_index_r(mycoord).coord(0);
    for (index_int row=my_first; row<=my_last; row++) {
      int col;
      // A narrow is tridiagonal
      col = row;     Anarrow->add_element(row,col,1.);
      col = row+1; if (col<globalsize) Anarrow->add_element(row,col,1.);
      col = row-1; if (col>=0)         Anarrow->add_element(row,col,1.);
      // A wide is distance 3 tridiagonal
      col = row;     Awide->add_element(row,col,1.);
      col = row+3; if (col<globalsize) Awide->add_element(row,col,1.);
      col = row-3; if (col>=0)         Awide->add_element(row,col,1.);
    }
  }
  shared_ptr<kernel>
    make_input = shared_ptr<kernel>( new mpi_origin_kernel(a) ),
    fast_mult1 = shared_ptr<kernel>( new mpi_spmvp_kernel(a,b,Anarrow) ),
    fast_mult2 = shared_ptr<kernel>( new mpi_spmvp_kernel(b,c1,Anarrow) ),
    slow_mult  = shared_ptr<kernel>( new mpi_spmvp_kernel(a,c2,Awide) ),
    assemble   = shared_ptr<kernel>( new mpi_sum_kernel(c1,c2,d) );

  algorithm *queue = new mpi_algorithm(decomp);
  CHECK_NOTHROW( queue->add_kernel(make_input) );
  CHECK_NOTHROW( queue->add_kernel(fast_mult1) );
  CHECK_NOTHROW( queue->add_kernel(fast_mult2) );
  CHECK_NOTHROW( queue->add_kernel(slow_mult) );
  CHECK_NOTHROW( queue->add_kernel(assemble) );

  SECTION( "kernel analysis" ) {
    CHECK_NOTHROW( make_input->analyze_dependencies() );
    CHECK_NOTHROW( fast_mult1->analyze_dependencies() );
    CHECK_NOTHROW( fast_mult2->analyze_dependencies() );
    CHECK_NOTHROW( slow_mult->analyze_dependencies() );
    CHECK_NOTHROW( assemble->analyze_dependencies() );
    return;
  }
  SECTION( "queue analysis" ) {
    CHECK_NOTHROW( queue->analyze_dependencies() );
  }
  
  SECTION( "strictly local" ) {
    int set = nlocal/2;
    {
      // check that we have enough space
      CHECK( a->volume(mycoord)==nlocal );
      CHECK( (set-2)>0 );
      CHECK( (set+2)<(nlocal-1) );
      // set input vector to delta halfway each subdomain
      auto adata = a->get_data(mycoord);
      for (index_int i=0; i<nlocal; i++)
	adata.at(i) = 0.;
      adata.at(set) = 1.;
    }
    CHECK_NOTHROW( queue->execute() );
    { // result of one narrow multiply
      INFO( "Anarrow" );
      auto data = b->get_data(mycoord); 
      for (index_int i=0; i<nlocal; i++) {
	INFO( "b i=" << i );
	if (i<set-1 || i>set+1)
	  CHECK( data.at(i)==Approx(0.) );
	else
	  CHECK( data.at(i)!=Approx(0.) );
      }
    }
    { // two narrow multiplies in a row
      INFO( "Anarrow^2" );
      auto data = c1->get_data(mycoord);
      for (index_int i=0; i<nlocal; i++) {
	INFO( "c1 i=" << i );
	if (i<set-2 || i>set+2)
	  CHECK( data.at(i)==Approx(0.) );
	else
	  CHECK( data.at(i)!=Approx(0.) );
      }
    }
    { // result of one wide multiply
      INFO( "Awide" );
      auto data = c2->get_data(mycoord); 
      for (index_int i=0; i<nlocal; i++) {
	INFO( "b i=" << i );
	if (i==set-3 || i==set+3 || i==set)
	  CHECK( data.at(i)!=Approx(0.) );
	else
	  CHECK( data.at(i)==Approx(0.) );
      }
    }
    { // adding it together
      auto data = d->get_data(mycoord);
      for (index_int i=0; i<nlocal; i++) {
	INFO( "d i=" << i );
	if (i<set-3 || i>set+3)
	  CHECK( data.at(i)==Approx(0.) );
	else
	  CHECK( data.at(i)!=Approx(0.) );
      }
    }
  }
  SECTION( "spilling" ) {
    {
      auto adata = a->get_data(mycoord);
      for (index_int i=0; i<nlocal; i++)
  	adata.at(i) = 0.;
      if (mytid%2==1)
  	adata.at(0) = 1.;
    }
    CHECK_NOTHROW( queue->execute() );
    {
      auto data = d->get_data(mycoord);
      if (mytid%2==0) { // crud at the top
  	for (index_int i=0; i<nlocal; i++) {
  	  INFO( "i=" << i );
  	  if (i<nlocal-3)
  	    CHECK( data.at(i)==Approx(0.) );
  	  else
  	    CHECK( data.at(i)!=Approx(0.) );
  	}
      } else { // crud at the bottom
  	for (index_int i=0; i<nlocal; i++) {
  	  INFO( "i=" << i );
  	  if (i>3)
  	    CHECK( data.at(i)==Approx(0.) );
  	  else
  	    CHECK( data.at(i)!=Approx(0.) );
  	}
      }
    }
  }
}

TEST_CASE( "Bilinear laplace","[distribution][multi][81]" ) {

#include "laplace_functions.h"

  if (ntids<4) { printf("need at least 4 procs for grid\n"); return; }

  // two-d decomposition
  int laplace_dim = 2;
  processor_coordinate layout = arch.get_proc_layout(laplace_dim);
  decomp = mpi_decomposition(arch,layout);
  processor_coordinate mycoord = decomp.coordinate_from_linear(mytid);

  shared_ptr<distribution> nodes_dist;
  shared_ptr<object> nodes_in,nodes_out;
  shared_ptr<kernel> bilinear_op;

  // number of elements (in each direction) is an input parameter
  index_int local_nnodes = 2;

  /* Create distributions */
  //SECTION( "set from global size" ) {
  nodes_dist = shared_ptr<distribution>
    ( make_shared<mpi_block_distribution>
      (decomp,domain_coordinate
       (vector<index_int>{local_nnodes*layout[0],local_nnodes*layout[1]})));
  //}
  
  // SECTION( "set from local size" ) {
  //   nodes_dist = shared_ptr<distribution>( make_shared<mpi_block_distribution>
  //     (decomp,vector<index_int>{local_nnodes,local_nnodes},-1);
  // }

  INFO( fmt::format("proc: {} out of decomposition: {}",
		    mycoord.as_string(),layout.as_string()) );

  /* Create the objects */
  nodes_in = shared_ptr<object>( new mpi_object(nodes_dist) ); nodes_in->set_name("nodes in");
  nodes_out = shared_ptr<object>( new mpi_object(nodes_dist) ); nodes_out->set_name("nodes out");

  stencil_operator *bilinear_stencil = new stencil_operator(2);
  bilinear_stencil->add( 0, 0);
  bilinear_stencil->add( 0,+1);
  bilinear_stencil->add( 0,-1);
  bilinear_stencil->add(-1, 0);
  bilinear_stencil->add(-1,+1);
  bilinear_stencil->add(-1,-1);
  bilinear_stencil->add(+1, 0);
  bilinear_stencil->add(+1,+1);
  bilinear_stencil->add(+1,-1);

bilinear_op = shared_ptr<kernel>( new mpi_kernel(nodes_in,nodes_out) );
  REQUIRE_NOTHROW( bilinear_op->add_sigma_stencil(bilinear_stencil) );
  bilinear_op->set_localexecutefn( &laplace_bilinear_fn );

  algorithm *bilinear = new mpi_algorithm(decomp);
bilinear->add_kernel( shared_ptr<kernel>( new mpi_setconstant_kernel(nodes_in,1.) ) );
  bilinear->add_kernel( bilinear_op );
  REQUIRE_NOTHROW( bilinear->analyze_dependencies() );

  bool
    pfirst_i = mycoord.coord(0)==0, pfirst_j = mycoord.coord(1)==0,
    plast_i = mycoord.coord(0)==layout.coord(0)-1,
    plast_j = mycoord.coord(1)==layout.coord(1)-1;
  INFO( fmt::format("Proc {} is first:{},{} last:{},{}",
		    mycoord.as_string(),pfirst_i,pfirst_j,plast_i,plast_j) );

  shared_ptr<task> ltask;
  REQUIRE_NOTHROW( ltask = bilinear_op->get_tasks().at(0) );
  shared_ptr<object> halo;
  REQUIRE_NOTHROW( halo = bilinear_op->get_beta_object(0) );
  auto pstruct = nodes_out->get_processor_structure(mycoord);
  auto hstruct = halo->get_processor_structure(mycoord);
  auto
    pfirst = pstruct->first_index_r(), plast = pstruct->last_index_r(),
    hfirst = hstruct->first_index_r(), hlast = hstruct->last_index_r();
  INFO( fmt::format("Halo (multi: {}) has endpoints {}--{}",
		    hstruct->is_multi(),hfirst.as_string(),hlast.as_string() ) );
  CHECK( hfirst[0]==pfirst[0]-!pfirst_i );
  CHECK( hfirst[1]==pfirst[1]-!pfirst_j );
  CHECK( hlast[0]==plast[0]+!plast_i );
  CHECK( hlast[1]==plast[1]+!plast_j );
  vector<shared_ptr<message>> msgs;
  REQUIRE_NOTHROW( msgs = ltask->get_receive_messages() );
  CHECK( msgs.size()==
	 9- 3*pfirst_i - 3*pfirst_j - 3*plast_i - 3*plast_j
	 + (pfirst_i && pfirst_j)
	 + (pfirst_i && plast_j)
	 + (plast_i && pfirst_j)
	 + (plast_i && plast_j)
	 );

  REQUIRE_NOTHROW( bilinear->execute() );

  {
    data_pointer data;
    REQUIRE_NOTHROW( data = nodes_in->get_data(mycoord) );
    int
      ilo = mycoord.coord(0), // ( mycoord.coord(0)==0 ? 0 : 1 ),
      jlo = mycoord.coord(1), // ( mycoord.coord(1)==0 ? 0 : 1 ),
      ihi = local_nnodes, // - ( mycoord.coord(0)==layout.coord(0)-1 ? 0 : 1 ),
      jhi = local_nnodes; // - ( mycoord.coord(1)==layout.coord(1)-1 ? 0 : 1 );
    for (int i=ilo; i<ihi; i++) {
      for (int j=jlo; j<jhi; j++) {
	INFO( i << "," << j );
	CHECK( data.at( i*local_nnodes + j )==Approx(1.) );
      }
    }
  }

  {
    data_pointer data;
    REQUIRE_NOTHROW( data = nodes_out->get_data(mycoord) );
    int
      ilo = ( mycoord.coord(0)==0 ? 1 : 0 ),
      jlo = ( mycoord.coord(1)==0 ? 1 : 0 ),
      ihi = local_nnodes - ( mycoord.coord(0)==layout.coord(0)-1 ? 1 : 0 ),
      jhi = local_nnodes - ( mycoord.coord(1)==layout.coord(1)-1 ? 1 : 0 );
    for (int i=ilo; i<ihi; i++) {
      for (int j=jlo; j<jhi; j++) {
	INFO( fmt::format("local i,j={},{} is global {},{}",i,j,i+pfirst[0],j+pfirst[1]) );
	CHECK( data.at( i*local_nnodes + j )==Approx(0.) );
      }
    }
  }
}

#if 0
TEST_CASE( "Masked distribution on output","[distribution][mask][111]" ) {
  if( ntids<2 ) { printf("masking requires two procs\n"); return; }

  INFO( " mytid = " << mytid );
  index_int localsize = 5; int alive;
  processor_mask *mask;

  alive = 1; processor_coordinate alive_proc = processor_coordinate1d(alive);
  REQUIRE_NOTHROW( mask = new processor_mask(decomp) );
  REQUIRE_NOTHROW( mask->add(alive_proc) );


  shared_ptr<distribution> block, masked_block;
  REQUIRE_NOTHROW( block = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,localsize,-1) ) );
  REQUIRE_NOTHROW( masked_block = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,localsize,-1) ) );
  REQUIRE_NOTHROW( masked_block->add_mask(mask) );
  CHECK( masked_block->lives_on(alive_proc) );

  shared_ptr<object> whole_vector, masked_vector;
  REQUIRE_NOTHROW( whole_vector  = shared_ptr<object>( new mpi_object(block) ) );
  whole_vector->set_name("whole vector");
  REQUIRE_NOTHROW( masked_vector = shared_ptr<object>( new mpi_object(masked_block) ) );
  masked_vector->set_name("masked vector");

  data_pointer data;
  REQUIRE_NOTHROW( whole_vector->allocate() );
  REQUIRE_NOTHROW( masked_vector->allocate() );
  CHECK_NOTHROW( data = whole_vector->get_data(mycoord) );
  CHECK( block->lives_on(mycoord) );
  if (mytid==alive) {
    CHECK( masked_block->lives_on(mycoord) );
    CHECK( masked_vector->lives_on(mycoord) );
    REQUIRE_NOTHROW( data = masked_vector->get_data(mycoord) );
  } else {
    CHECK( !masked_block->lives_on(mycoord) );
    CHECK( !masked_vector->lives_on(mycoord) );
    REQUIRE_THROWS( data = masked_vector->get_data(mycoord) );
  }

  {
    data_pointer indata,outdata;
    REQUIRE_NOTHROW( indata = whole_vector->get_data(mycoord) );
    for (index_int i=0; i<localsize; i++) indata.at(i) = 1.;
    if (masked_vector->lives_on(mycoord)) {
      REQUIRE_NOTHROW( outdata = masked_vector->get_data(mycoord) );
      CHECK( outdata!=nullptr );
      for (index_int i=0; i<localsize; i++) outdata.at(i) = 2.;
    } else {
      REQUIRE_THROWS( outdata = masked_vector->get_data(mycoord) );
    }
  }
  auto copy = shared_ptr<kernel>( new mpi_kernel(whole_vector,masked_vector) );
  copy->set_last_dependency().set_type_local();
  copy->set_localexecutefn( &veccopy );
  REQUIRE_NOTHROW( copy->analyze_dependencies() );

  
  REQUIRE_NOTHROW( copy->execute() );

  if (masked_vector->lives_on(mycoord)) { data_pointer outdata;
    REQUIRE_NOTHROW( outdata = masked_vector->get_data(mycoord) );
    for (index_int i=0; i<localsize; i++) 
      CHECK( outdata.at(i) == 1. );
  } // output vector otherwise not defined
}

TEST_CASE( "Masked distribution on input","[distribution][mask][112]" ) {

  if( ntids<2 ) { printf("masking requires two procs\n"); return; }
  INFO( "mytid: " << mytid );

  index_int localsize = 5;
  processor_mask *mask;

  REQUIRE_NOTHROW( mask = new processor_mask(decomp) );
  processor_coordinate p0 = processor_coordinate1d(0);
  REQUIRE_NOTHROW( mask->add(p0) );

  auto block = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,localsize,-1) ),
    masked_block = shared_ptr<distribution>( new mpi_distribution( *block ) );
  REQUIRE_NOTHROW( masked_block->add_mask(mask) );

  auto
    whole_vector = shared_ptr<object>( new mpi_object(block) ),
    masked_vector = shared_ptr<object>( new mpi_object(masked_block) );
  data_pointer data;
  REQUIRE_NOTHROW( whole_vector->allocate() );
  REQUIRE_NOTHROW( masked_vector->allocate() );
  CHECK_NOTHROW( data = whole_vector->get_data(mycoord) );
  CHECK( block->lives_on(mycoord) );
  if (mytid==0) {
    CHECK( masked_block->lives_on(mycoord) );
    REQUIRE_NOTHROW( data = masked_vector->get_data(mycoord) );
  } else {
    CHECK( !masked_block->lives_on(mycoord) );
    REQUIRE_THROWS( data = masked_vector->get_data(mycoord) );
  }

  // set the whole output to 1
  {
    data_pointer outdata;
    REQUIRE_NOTHROW( outdata = whole_vector->get_data(mycoord) );
    for (index_int i=0; i<localsize; i++) outdata.at(i) = 1.;
  }
  // set input to 2, only on the mask
  if (masked_vector->get_distribution()->lives_on(mycoord)) {
    data_pointer indata;
    REQUIRE_NOTHROW( indata = masked_vector->get_data(mycoord) );
    CHECK( indata!=nullptr );
    for (index_int i=0; i<localsize; i++) indata.at(i) = 2.;
  } else {
    REQUIRE_THROWS( masked_vector->get_data(mycoord) );
  }
  shared_ptr<kernel> copy;
  REQUIRE_NOTHROW( copy = shared_ptr<kernel>( new mpi_copy_kernel(masked_vector,whole_vector) ) );

  {
    dependency dep;
    REQUIRE_NOTHROW( dep = copy->last_dependency() );
    REQUIRE_NOTHROW( copy->analyze_dependencies() );
    CHECK( !dep.get_beta_object()->has_mask() );
    REQUIRE_NOTHROW( copy->execute() );

    data_pointer outdata;
    REQUIRE_NOTHROW( outdata = whole_vector->get_data(mycoord) );
    if (mytid==0) { // output has copied value
      for (index_int i=0; i<localsize; i++) 
	CHECK( outdata.at(i) == 2. );
    } else { // output has original value
      for (index_int i=0; i<localsize; i++) 
	CHECK( outdata.at(i) == 1. );
    }
  }
}

TEST_CASE( "Two masks","[distribution][mask][113]" ) {

  if( ntids<4 ) { printf("test 72 needs at least 4 procs\n"); return; }

  INFO( "mytid=" << mytid );
  index_int localsize = 5;
  processor_mask *mask1,*mask2;

  mask1 = new processor_mask(decomp);
  mask2 = new processor_mask(decomp);
  for (int tid=0; tid<ntids; tid+=2) {
    processor_coordinate ptid = processor_coordinate1d(tid);
    REQUIRE_NOTHROW( mask1->add(ptid) );
  }
  for (int tid=0; tid<ntids; tid+=4) {
    processor_coordinate ptid = processor_coordinate1d(tid);
    REQUIRE_NOTHROW( mask2->add(ptid) );
  }
  
  shared_ptr<object>
    whole_vector,masked_vector1,masked_vector2;
  {
    auto block = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,localsize,-1) );
    whole_vector = shared_ptr<object>( new mpi_object(block) );
    REQUIRE_NOTHROW( whole_vector->allocate() );
    data_pointer data;
    REQUIRE_NOTHROW( data = whole_vector->get_data(mycoord) );
    for (index_int i=0; i<localsize; i++)
      data.at(i) = 1;
  }
  
  {
    auto masked_block1 = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,localsize,-1) );
    REQUIRE_NOTHROW( masked_block1->add_mask(mask1) );
    REQUIRE_NOTHROW( masked_vector1 = shared_ptr<object>( new mpi_object(masked_block1) ) );
    REQUIRE_NOTHROW( masked_vector1->allocate() );
    if (masked_vector1->get_distribution()->lives_on(mycoord)) {
      data_pointer data;
      REQUIRE_NOTHROW( data = masked_vector1->get_data(mycoord) );
      for (index_int i=0; i<localsize; i++)
	data.at(i) = 2;
    }
  }
  {
    auto masked_block2 = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,localsize,-1) );
    REQUIRE_NOTHROW( masked_block2->add_mask(mask2) );
    REQUIRE_NOTHROW( masked_vector2 = shared_ptr<object>( new mpi_object(masked_block2) ) );
    REQUIRE_NOTHROW( masked_vector2->allocate() );
    if (masked_vector2->get_distribution()->lives_on(mycoord)) {
      data_pointer data;
      REQUIRE_NOTHROW( data = masked_vector2->get_data(mycoord) );
      for (index_int i=0; i<localsize; i++)
	data.at(i) = 4;
    }
  }

  auto copy1 = shared_ptr<kernel>( new mpi_kernel(whole_vector,masked_vector1) );
  copy1->set_last_dependency().add_sigma_operator( ioperator("none") );
  copy1->set_localexecutefn( &veccopy );
  REQUIRE_NOTHROW( copy1->analyze_dependencies() );
  REQUIRE_NOTHROW( copy1->execute() );

  {
    data_pointer data1;
    if (masked_vector1->get_distribution()->lives_on(mycoord)) {
      REQUIRE_NOTHROW( data1 = masked_vector1->get_data(mycoord) );
      for (index_int i=0; i<localsize; i++) 
	CHECK( data1[i] == 1. );
    } else
      REQUIRE_THROWS( data1 = masked_vector1->get_data(mycoord) );
  }

  auto copy2 = shared_ptr<kernel>( new mpi_kernel(masked_vector1,masked_vector2) );
  copy2->set_last_dependency().add_sigma_operator( ioperator("none") );
  copy2->set_localexecutefn( &veccopy );
  REQUIRE_NOTHROW( copy2->analyze_dependencies() );
  REQUIRE_NOTHROW( copy2->execute() );

  {
    data_pointer data2;
    if (masked_vector2->get_distribution()->lives_on(mycoord)) {
      REQUIRE_NOTHROW( data2 = masked_vector2->get_data(mycoord) );
      for (index_int i=0; i<localsize; i++) 
	CHECK( data2[i] == 1. );
    } else
      REQUIRE_THROWS( data2 = masked_vector2->get_data(mycoord) );
  }
}

TEST_CASE( "mask on replicated scalars","[distribution][mask][replicated][114]" ) {

  if (ntids<2) { printf("test 73 needs at least 2\n"); return; }
  INFO( "mytid=" << mytid );

  processor_mask *mask = new processor_mask(decomp);
  mask->add(new processor_coordinate1d(0));
  shared_ptr<distribution> scalar = shared_ptr<distribution>( new mpi_replicated_distribution(decomp) );
  shared_ptr<distribution> scalars = shared_ptr<distribution>( new mpi_replicated_distribution(decomp) );
  // mpi_distribution( *scalar ); copying doesn't work!?
  REQUIRE_NOTHROW( scalar->add_mask(mask) );
  CHECK( !scalars->has_mask() );

  shared_ptr<object> one,ones;
  REQUIRE_NOTHROW( one = shared_ptr<object>( new mpi_object(scalar) ) );
  if (mytid==0) {
    CHECK( one->get_distribution()->lives_on(mycoord) );
    CHECK( one->first_index_r(mycoord).coord(0)==0 );
  } else {
    CHECK( !one->get_distribution()->lives_on(mycoord) );
    REQUIRE_THROWS( one->first_index_r(mycoord).coord(0) );
  }
  one->allocate();
  double oneval = 1.; REQUIRE_NOTHROW( one->set_value(&oneval) );
  data_pointer in_data;
  if (one->get_distribution()->lives_on(mycoord)) {
    REQUIRE_NOTHROW( in_data = one->get_data(mycoord) );
    CHECK( in_data.at(0)==Approx(1.) );
  } else 
    REQUIRE_THROWS( in_data = one->get_data(mycoord) );

  REQUIRE_NOTHROW( ones = shared_ptr<object>( new mpi_object(scalars) ) );
  CHECK( ones->first_index_r(mycoord).coord(0)==0 );
  ones->allocate();

  shared_ptr<kernel> bcast; shared_ptr<task> tsk;
  REQUIRE_NOTHROW( bcast = shared_ptr<kernel>( new mpi_bcast_kernel(one,ones) ) );
  REQUIRE_NOTHROW( bcast->set_comm_trace_level(comm_trace_level::EXEC) );
  CHECK( ones->get_distribution()->lives_on(mycoord) );
  //REQUIRE_NOTHROW( bcast->get_dependencies().at(0)->ensure_beta_distribution(ones) );
  shared_ptr<distribution> beta;
  REQUIRE_NOTHROW( beta = bcast->get_dependencies().at(0)->get_beta_object()->get_distribution() );
  CHECK( beta->volume(mycoord)==1 );
  CHECK( beta->first_index_r(mycoord).coord(0)==0 );

  REQUIRE_NOTHROW( bcast->analyze_dependencies() );
  dependency dep; REQUIRE_NOTHROW( dep = bcast->last_dependency() );
  shared_ptr<object> halo; REQUIRE_NOTHROW( halo = dep.get_beta_object() );
  CHECK( halo->get_distribution()->lives_on(mycoord) );

  REQUIRE_NOTHROW( tsk = bcast->get_tasks().at(0) );
  vector<shared_ptr<message>> msgs;
  REQUIRE_NOTHROW( msgs = tsk->get_receive_messages() );
  CHECK( msgs.size()==1 );
  CHECK( msgs.at(0)->get_receiver()->coord(0)==mytid );
  CHECK( msgs.at(0)->get_sender().coord(0)==0 );
  CHECK( msgs.at(0)->get_global_struct()->local_size()==1 );
  CHECK( msgs.at(0)->get_local_struct()->local_size()==1 );

  REQUIRE_NOTHROW( msgs = tsk->get_send_messages() );
  if (one->get_distribution()->lives_on(mycoord)) {
    CHECK( msgs.size()==ntids );
    //for (auto m=msgs->begin(); m!=msgs->end(); ++m) {
    for ( auto m : msgs ) {
      CHECK( m->get_global_struct()->local_size()==1 );
      CHECK( m->get_local_struct()->local_size()==1 );
    }
  } else
    CHECK( msgs.size()==0 );

  REQUIRE_NOTHROW( bcast->execute() );
  data_pointer halo_data;
  REQUIRE_NOTHROW( halo_data = halo->get_data(mycoord) );
  CHECK( halo_data.at(0)==Approx(1.) );
  data_pointer out_data;
  REQUIRE_NOTHROW( out_data = ones->get_data(mycoord) );
  CHECK( out_data.at(0)==Approx(1.) );
}
#endif
