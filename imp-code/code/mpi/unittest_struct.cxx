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
 **** unit tests for communication structure
 ****
 ****************************************************************/

#include <stdlib.h>
#include <math.h>

#include "catch2/catch_all.hpp"

#include "mpi_base.h"
#include "mpi_static_vars.h"
#include "unittest_functions.h"
#include "imp_functions.h"
#include "mpi_ops.h"

using fmt::format;
using fmt::print;
using fmt::memory_buffer;
using fmt::format_to;
using fmt::to_string;

using std::make_shared;
using std::shared_ptr;
using std::vector;

TEST_CASE( "Environment is proper","[environment][init][0]" ) {
  return;
  
  int tmp; processor_coordinate tmpcoord;
  CHECK_NOTHROW( tmp = env->get_architecture()->nprocs() );
  CHECK_NOTHROW( tmp = env->get_architecture()->mytid() );
  CHECK_NOTHROW( tmpcoord = env->get_architecture()->get_proc_endpoint(mytid) );

  // is the enviroment non-empty?
  CHECK( env->get_architecture()->nprocs() > 0 );

  //  check that we instantiate processes
  CHECK( ntids>0 );
  CHECK( ntids==env->get_architecture()->nprocs() );
  CHECK( mytid==env->get_architecture()->mytid() );
  // mpi is not thread-based so we can not query this.
  //  CHECK_THROWS( env->get_architecture()->nthreads_per_node()==1 );

  {  // for completeness
    int *tids = new int[ntids];
    printf("procid=%d\n",mytid);
    //    MPI_Allgather(&mytid,1,MPI_INT,tids,1,MPI_INT,comm);
    for (int p=0; p<ntids; p++) 
      REQUIRE(tids[p]==p);
  }
  return;

  {
    const char* path;
    shared_ptr<distribution> d;
    index_int nlocal = 10, nglobal = nlocal*ntids;
    SECTION( "from global" ) { path = "from global";
      REQUIRE_NOTHROW( d = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,-1,nglobal) ) );
    }
    SECTION( "from local" ) { path = "from local";
      REQUIRE_NOTHROW( d = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,nlocal,-1) ) );
    }
    INFO( "construction: " << path );
    CHECK( d->global_volume()==nglobal );
    int nt,o;
    REQUIRE_NOTHROW( nt = d->domains_volume() );
    REQUIRE_NOTHROW( o = d->get_over_factor() );
    CHECK( nt==ntids*o );
    shared_ptr<object> v;
    
    //REQUIRE_NOTHROW( v = new mpi_object(d) );
    REQUIRE_NOTHROW( v = d->new_object(d) );
    CHECK( v->get_cookie()==entity_cookie::OBJECT );
    REQUIRE_NOTHROW( nt = v->domains_volume() );
    CHECK( nt == ntids*o );
  }
};

TEST_CASE( "entity stuff","[entity][02]" ) {
  int n_obj;
  {
    auto summary = env->mode_summarize_entities();
    n_obj = std::get<RESULT_OBJECT>(*summary);
  }
  auto d =   shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,100,-1) );
  return;
  shared_ptr<object> o1,o2;
  REQUIRE_NOTHROW( o1 = d->new_object(d) );
  REQUIRE_NOTHROW( o2 = d->new_object(d) );
  {
    auto summary = env->mode_summarize_entities();
    n_obj = std::get<RESULT_OBJECT>(*summary) - n_obj;
  }
  CHECK( n_obj==2 );
}

TEST_CASE( "Test parallel indexstructs","[index][structure][03]" ) {
  index_int localsize = 20, globalsize = localsize*ntids;
  auto range =
    shared_ptr<indexstruct>( new contiguous_indexstruct(0,globalsize-1) );
  CHECK( decomp.get_dimensionality()==1 );
  int shift=0,extend=0; const char *path;
  SECTION( "plain and simple" ) { shift = 0; path = "none";
  };
  SECTION( "shift right" ) { shift = 5; path = "shift 5";
    REQUIRE_NOTHROW( range = range->operate( shift_operator(shift) ) );
  }
  SECTION( "shift left" ) { shift = -5; path = "shift -5";
    REQUIRE_NOTHROW( range = range->operate( shift_operator(shift) ) );
  }
  SECTION( "extend" ) { extend = 2; path = "extend 2";
  }
  INFO( "test: " << path << " giving range: " << range->as_string() );

  parallel_indexstruct *parallel;
  REQUIRE_NOTHROW( parallel = new parallel_indexstruct(ntids) );
  REQUIRE_NOTHROW( parallel->create_from_indexstruct(range) );

  INFO( "parallel structure: " << parallel->as_string() );
  if (extend>0 && ntids>1) { int nd;
    REQUIRE_NOTHROW( nd = decomp.domains_volume() );
    CHECK( nd==ntids );
    shared_ptr<indexstruct>
      laststruct,xtndstruct;
    REQUIRE_NOTHROW( laststruct = parallel->get_processor_structure(nd-1) );
    REQUIRE_NOTHROW( xtndstruct = laststruct->operate(shift_operator(extend)) );
    CHECK( xtndstruct->first_index()==laststruct->first_index()+extend );
    CHECK( xtndstruct->last_index()==laststruct->last_index()+extend );
    REQUIRE_NOTHROW( parallel->set_processor_structure( nd-1, xtndstruct ) );
  }
  CHECK( parallel->local_size(mytid)==localsize );
  shared_ptr<indexstruct> enclosing;
  REQUIRE_NOTHROW( enclosing = parallel->get_enclosing_structure() );
  INFO( "enclosing struct: " << enclosing->as_string() );
  if (extend==0)
    CHECK( range->equals(enclosing) );
  else if (ntids>1) {
    CHECK( parallel->outer_size()==globalsize+extend );
    CHECK( enclosing->local_size()==globalsize+extend );
  }
}

TEST_CASE( "MPI data is local","[data][04]" ) {
  if (arch.get_over_factor()>1) return;
  index_int nlocal=20,nglobal = nlocal*ntids;
  parallel_indexstruct *pidx;
  REQUIRE_NOTHROW( pidx = new parallel_indexstruct(ntids) );
  REQUIRE_NOTHROW( pidx->create_from_global_size(nglobal) );
  CHECK( pidx->local_size(mytid)==nlocal );
  parallel_structure pstr;
  REQUIRE_NOTHROW( pstr = parallel_structure(decomp) );
  REQUIRE_NOTHROW( pstr.create_from_global_size(nglobal) );
  CHECK( pstr.get_dimension_structure(0)->local_size(mytid)==nlocal );

  shared_ptr<distribution> block;
  REQUIRE_NOTHROW( block =   shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,nlocal,-1) ) );
  shared_ptr<object> obj;
  REQUIRE_NOTHROW( obj = block->new_object(block) ); //new mpi_object(block) );
  return;
  REQUIRE_NOTHROW( obj->allocate() );
  data_pointer data;
  REQUIRE_NOTHROW( data = obj->get_data(mycoord) );
  index_int s; REQUIRE_NOTHROW( s = obj->local_allocation() );
  CHECK( s==nlocal );
  for (int i=0; i<s; i++)
    data.at(i) = 3.14;
  processor_coordinate nextp;
  REQUIRE_NOTHROW( nextp = decomp.coordinate_from_linear( (mytid+1)%ntids ) );
  INFO( "next processor: " << nextp.as_string() );
  if (ntids>1)
    REQUIRE_THROWS( obj->get_data(nextp) );
}

TEST_CASE( "MPI ortho data is local","[data][ortho][04k]" ) {
  if (arch.get_over_factor()>1) {
    print("Can not do [04k] with arch over_factor {}\n",arch.get_over_factor());
    return;
  }
  int k=3;
  index_int nlocal=20,nglobal = nlocal*ntids;

  shared_ptr<distribution> block;
  REQUIRE_NOTHROW
    ( block = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,k,nlocal,-1) ) );
  shared_ptr<object> obj;
  REQUIRE_NOTHROW( obj = block->new_object(block) );
  REQUIRE_NOTHROW( obj->allocate() );
  data_pointer data;
  REQUIRE_NOTHROW( data = obj->get_data(mycoord) );
  index_int local_alloc;
  REQUIRE_NOTHROW( local_alloc = obj->local_allocation() );
  REQUIRE( local_alloc==nlocal*k );
  int local_index{0};
  for (int i=0; i<nlocal; i++) {
    for (int j=0; j<k; j++) {
      CHECK_NOTHROW( data.at(local_index) = i+(1./(j+1)) );
      local_index++;
    }
  }
  CHECK( local_index==local_alloc );
}

TEST_CASE( "sanity check on object construction","[mpi][object][distribution][05]" ) {

  int nlocal = 10,gsize = nlocal*ntids;
  shared_ptr<distribution> d1;
  REQUIRE_NOTHROW( d1 =   shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,-1,gsize) ) );
  CHECK( d1->get_dimensionality()==1 );
  CHECK( d1->has_type(distribution_type::CONTIGUOUS) );
  shared_ptr<object> o1,r1;
  CHECK_NOTHROW( r1 = d1->new_object(d1) ); //new mpi_object(d1) );
  CHECK_NOTHROW( o1 = d1->new_object(d1) ); //new mpi_object(d1) );

  int *senders = new int[ntids], receives;
  REQUIRE_NOTHROW( receives = d1->reduce_scatter(senders,mytid) );
}

TEST_CASE( "make sure that beta construction does not touch dists","[distribution][beta][06]" ) {
  INFO( "mytid=" << mytid );
  // create distributions and objects for threepoint combination
  ioperator no_op("none"), right_shift(">=1"), left_shift("<=1");

  int nlocal = 10,gsize = nlocal*ntids, dim=1;
  shared_ptr<distribution> d1;
  REQUIRE_NOTHROW( d1 =   shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,-1,gsize) ) );
  domain_coordinate myfirst,mylast;
  REQUIRE_NOTHROW( myfirst = d1->first_index_r(mycoord) );
  REQUIRE_NOTHROW( mylast = d1->last_index_r(mycoord) );

  signature_function *sigma_opers = new signature_function();
  shared_ptr<distribution> beta_dist;
  sigma_opers->add_sigma_operator( no_op );
  sigma_opers->add_sigma_operator( left_shift );
  sigma_opers->add_sigma_operator( right_shift );

  parallel_structure preds;
  shared_ptr<multi_indexstruct> enclosing;
  REQUIRE_NOTHROW( enclosing = d1->get_enclosing_structure() );
  REQUIRE_NOTHROW( preds = sigma_opers->derive_beta_structure(d1,enclosing) );
  CHECK_NOTHROW( beta_dist = shared_ptr<distribution>( new mpi_distribution(preds) ) );

  CHECK( d1->first_index_r(mycoord)==myfirst );
  CHECK( d1->last_index_r(mycoord)==mylast );
}

TEST_CASE( "checking on index offset calculation","[index][07]" ) {
  int dim;
  SECTION( "1" ) { dim = 1; }
  SECTION( "2" ) { dim = 2; }
  SECTION( "3" ) { dim = 3; }
  INFO( "mytid=" << mytid << "; dim=" << dim );

  processor_coordinate layout;
  REQUIRE_NOTHROW( layout = arch.get_proc_layout(dim) );
  INFO( "layout: " << layout.as_string() );
  CHECK( layout.volume()==ntids );
  decomposition decomp;
  REQUIRE_NOTHROW( decomp = mpi_decomposition(arch,layout) );
  CHECK( decomp.get_dimensionality()==dim );
  processor_coordinate mycoord;
  REQUIRE_NOTHROW( mycoord = decomp.coordinate_from_linear(mytid) );
  CHECK( mycoord.get_dimensionality()==dim );
  INFO( "mycoord=" << mycoord.as_string() );

  vector<index_int> domain;
  index_int gsize = 1;
  for (int id=0; id<dim; id++) {
    int pi = layout.coord(id);
    CHECK( pi>=1 );
    index_int dsize = 10*pi;
    domain.push_back(dsize);
    gsize *= dsize;
  }
  domain_coordinate gsize_vector( domain );
  shared_ptr<distribution> d; shared_ptr<object> v;
  REQUIRE_NOTHROW
    ( d = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,gsize_vector) ) );
  CHECK( d->get_dimensionality()==dim );
  REQUIRE_NOTHROW( v = d->new_object(d) ); //new mpi_object(d) );
  CHECK( v->global_volume()==gsize );

  shared_ptr<kernel> make;
  REQUIRE_NOTHROW( make = shared_ptr<kernel>( new mpi_origin_kernel(v,std::string("vconstantp")) ) );
  REQUIRE_NOTHROW( make->set_localexecutefn(&vecsetconstantp) );
  REQUIRE_NOTHROW( make->analyze_dependencies(true) );
  REQUIRE_NOTHROW( make->execute(true) );

  data_pointer data;
  REQUIRE_NOTHROW( data = v->get_data(mycoord) );
  index_int n = d->volume(mycoord);
  for (int i=0; i<n; i++) {
    INFO( "index in data: " << i );
    CHECK( data.at(i)==Approx( (double)mytid ) );
  }
}

TEST_CASE( "support functions","[function][11]" ) {
  if (ntids<2) {
    printf("At least two procs for [01]\n"); return; }
  index_int localsize = 10, dim=1;
  shared_ptr<distribution> blocked;
  REQUIRE_NOTHROW( blocked =   shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,localsize,-1) ) );
  domain_coordinate f;
  REQUIRE_NOTHROW( f = blocked->global_first_index() );
  REQUIRE_NOTHROW( blocked->set_name("blocked01") );
  CHECK( blocked->get_dimension_structure(0)->get_type()!=distribution_type::UNDEFINED );
  CHECK( blocked->get_dimension_structure(0)->get_type()==distribution_type::CONTIGUOUS );
  CHECK( blocked->get_type()!=distribution_type::UNDEFINED );
  CHECK( blocked->get_type()==distribution_type::CONTIGUOUS );

  shared_ptr<object> v;
  REQUIRE_NOTHROW( v = blocked->new_object(blocked) );
  REQUIRE_NOTHROW( v->set_name("object01") );
  CHECK( !v->get_distribution()->get_processor_structure(mycoord)->is_empty() );

  // shared_ptr<kernel> set_delta;
  // REQUIRE_NOTHROW( set_delta = shared_ptr<kernel>( new mpi_kernel(v) ) );
  kernel set_delta;
  REQUIRE_NOTHROW( set_delta = mpi_kernel(v) );
  CHECK( set_delta.has_type_origin() );
  auto localstruct = set_delta.get_out_distribution()->get_processor_structure(mycoord);
  CHECK( !localstruct->is_empty() );
  CHECK( localstruct->volume()==localsize );

  //index_int delta;
  int p; domain_coordinate *delta;
  SECTION( "p=0" ) {
    delta = new domain_coordinate( vector<index_int>{0} );
    p = 0;
  }
  SECTION( "p=1" ) {
    delta = new domain_coordinate( vector<index_int>{localsize} );
    p = 1;
  }
  REQUIRE_NOTHROW
    ( set_delta.set_localexecutefn
      ( [delta] ( kernel_function_args ) -> void {
	vecdelta(kernel_function_call,*delta); } ) );
  REQUIRE_NOTHROW( set_delta.analyze_dependencies() );
  vector<shared_ptr<task>> tsks;
  REQUIRE_NOTHROW( tsks = set_delta.get_tasks() );
  CHECK( tsks.size()==1 );

  REQUIRE_NOTHROW( set_delta.execute() );

  data_pointer data;
  REQUIRE_NOTHROW( data = v->get_data(mycoord) );
  if (p==mytid)
    CHECK( data.at(0)==Approx(1.) );
  else
    CHECK( data.at(0)==Approx(0.) );
}

TEST_CASE( "Analyze single interval","[index][structure][12]" ) {
  int localsize = 5; int dim=1;
  auto dist = shared_ptr<distribution>( new mpi_distribution(decomp) );
  CHECK_NOTHROW( dist->create_from_global_size(localsize*ntids) );
  REQUIRE_NOTHROW( dist->memoize() );
  vector<shared_ptr<message>> msgs;
  shared_ptr<message> m; shared_ptr<multi_indexstruct> s;

  SECTION( "find an exact processor interval" ) {
    shared_ptr<multi_indexstruct> segment;
    REQUIRE_NOTHROW( segment = shared_ptr<multi_indexstruct>
    		     ( new multi_indexstruct( shared_ptr<indexstruct>
    					      {new contiguous_indexstruct(0,localsize-1)}
    					      ) ) );
    CHECK_NOTHROW( msgs = dist->messages_for_segment
		   ( 0,mycoord,self_treatment::INCLUDE,segment,segment ) );
    CHECK( msgs.size()==1 );
    CHECK_NOTHROW( m = msgs.at(0) );
    CHECK( m->get_sender()==processor_coordinate( vector<int>{0} ) );
    CHECK( m->get_receiver()==mycoord );
    REQUIRE_NOTHROW( s = m->get_global_struct() );
    REQUIRE( s!=nullptr );
    CHECK( s->first_index_r()==domain_coordinate( vector<index_int>{0} ) );
    CHECK( s->last_index_r()==domain_coordinate( vector<index_int>{localsize-1} ) );
  }

  SECTION( "find a sub interval" ) {
    shared_ptr<multi_indexstruct> segment;
    REQUIRE_NOTHROW( segment = shared_ptr<multi_indexstruct>
		     ( new multi_indexstruct( shared_ptr<indexstruct>
					{new contiguous_indexstruct(1,localsize-2)}
					      ) ) );
    CHECK_NOTHROW( msgs = dist->messages_for_segment
		   ( 0,mycoord,self_treatment::INCLUDE,segment,segment ) );
    CHECK( msgs.size()==1 );
    m = msgs[0];
    CHECK( m->get_sender().coord(0)==0 );
    CHECK( m->get_receiver().coord(0)==mytid );
    s = m->get_global_struct();
    REQUIRE( s!=nullptr );
    CHECK( s->first_index(0)==1 );
    CHECK( s->last_index(0)==localsize-2 );
  }

  SECTION( "something with nontrivial intersection" ) {
    INFO( "mytid=" << mytid );
    if (ntids>1) {
      shared_ptr<multi_indexstruct> segment;
      REQUIRE_NOTHROW( segment = // one element beyond the 1st proc
		       shared_ptr<multi_indexstruct>
		       ( new multi_indexstruct( shared_ptr<indexstruct>
					      {new contiguous_indexstruct(0,localsize)} )
			 ) );
      // VLE get this working again in multi mode.
      // first manually 
      // for (int ip=0; ip<ntids; ip++) {
      // 	int p = (mytid+ip)%ntids; // start with self first
      // 	INFO( "intersect with p=" << p );
      // 	shared_ptr<multi_indexstruct> intersect;
      // 	REQUIRE_NOTHROW
      // 	  ( intersect = segment->intersect( pstruct->get_processor_structure(p) ) );
      // 	if (p==0) {
      // 	  CHECK( !intersect->is_empty() );
      // 	} else if (p==1) {
      // 	  CHECK( !intersect->is_empty() );
      // 	} else {
      // 	  CHECK( intersect->is_empty() );
      // 	}
      // }

      // now with the system call
      CHECK_NOTHROW( msgs = dist->messages_for_segment
		     ( 0,mycoord,self_treatment::INCLUDE,segment,segment ) );
      CHECK( msgs.size()==2 );
      for (int im=0; im<2; im++) { // we don't insist on ordering
	shared_ptr<multi_indexstruct> range;
  	m = msgs[im];
  	CHECK( m->get_receiver().coord(0)==mytid );
  	CHECK( ( (m->get_sender().coord(0)==0) || (m->get_sender().coord(0)==1) ) );
  	range = m->get_global_struct();
  	if (m->get_sender().coord(0)==0) {
  	  CHECK( range->first_index(0)==0 );
  	  CHECK( range->last_index(0)==localsize-1 );
  	} else {
  	  CHECK( range->first_index(0)==localsize );
  	  CHECK( range->last_index(0)==localsize );
  	}
      }
    }
  }
}

TEST_CASE( "Analyze one dependency","[operate][dependence][14]") {

  INFO( "mytid=" << mytid );

  int localsize = 100, dim=1;
  shared_ptr<distribution> alpha = 
    shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,-1,localsize*ntids) );
  ioperator shiftop;
  shared_ptr<multi_indexstruct> alpha_block,segment;
  vector<shared_ptr<message>> msgs; shared_ptr<message> m;
  auto
    my_first = alpha->first_index_r(mycoord),
    my_last = alpha->last_index_r(mycoord);
  CHECK( my_first==mycoord_coord*localsize );
  CHECK( my_last==(mycoord_coord+1)*localsize-1 );

  // right bump
  shiftop = ioperator(">=1");
  CHECK( shiftop.is_right_shift_op() );
  CHECK( !shiftop.is_modulo_op() );
  alpha_block = alpha->get_processor_structure(mycoord); // should be from alpha/gamma
  CHECK( alpha_block->first_index_r()==my_first );

  {
    REQUIRE_NOTHROW( segment = alpha_block->operate
		     (shiftop,alpha->get_enclosing_structure()) );
    INFO( "messages for segment " << segment->as_string()
	  << " based on gamma block " << alpha_block->as_string() );
    CHECK( segment->first_index_r()==my_first+1 );
    CHECK_NOTHROW( msgs = alpha->messages_for_segment
		   ( 0,mycoord,self_treatment::INCLUDE,segment,segment ) );
    if (mytid<ntids-1) {
      CHECK( msgs.size()==2 );
    } else {
      CHECK( msgs.size()==1 );
    }
    m = msgs[0]; 
    CHECK( m->get_sender()==mycoord );
    CHECK( m->get_global_struct()->local_size_r().coord(0)==localsize-1 );
    CHECK( m->get_global_struct()->first_index_r()==my_first+1 );
    if (mytid<ntids-1) {
      m = msgs[1];
      CHECK( m->get_sender().coord(0)==mytid+1 );
      CHECK( m->get_global_struct()->volume()==1 );
      CHECK( m->get_global_struct()->first_index_r()==my_last+1 );
    }
  }

  // left bump
  shiftop = ioperator("<=1");
  CHECK( shiftop.is_left_shift_op() );
  CHECK( !shiftop.is_modulo_op() );
  alpha_block = alpha->get_processor_structure(mycoord);
  CHECK( alpha_block->local_size_r().coord(0)==localsize );
  REQUIRE_NOTHROW( segment = alpha_block->operate
		   (shiftop,alpha->get_enclosing_structure()) );
  CHECK_NOTHROW( msgs = alpha->messages_for_segment
		 ( 0,mycoord,self_treatment::INCLUDE,segment,segment ) );
  if (mytid>0) {
    CHECK( msgs.size()==2 );
  } else {
    CHECK( msgs.size()==1 );
  }
  if (mytid>0) {
    for (int im=0; im<2; im++) {
      m = msgs[im];
      if ( m->get_sender().coord(0)==mytid-1 ) {
	CHECK( m->get_global_struct()->local_size_r().coord(0)==1 );
      } else if ( m->get_sender().coord(0)==mytid ) {
	CHECK( m->get_global_struct()->local_size_r().coord(0)==localsize-1 );
      } else {
	CHECK( 1==0 );
      }
    }
  } else {
    m = msgs[0];
    CHECK( m->get_sender().coord(0)==mytid );
    CHECK( m->get_global_struct()->local_size_r().coord(0)==localsize-1 );
  }
}

TEST_CASE( "Analyze one cyclic dependency","[operate][dependence][cyclic][15]") {

  if (ntids<2) return;
  INFO( "mytid=" << mytid );

  int nlocal = 100,gsize = ntids*nlocal, dim=1;
  shared_ptr<distribution> alpha = 
    shared_ptr<distribution>( new mpi_cyclic_distribution(decomp,nlocal) );
  ioperator shiftop;
  vector<shared_ptr<message>> msgs; shared_ptr<message> m;
  auto
    my_first = alpha->first_index_r(mycoord),
    my_last = alpha->last_index_r(mycoord);
  shared_ptr<multi_indexstruct> alpha_block, segment;
  processor_coordinate sender; index_int ls;

  CHECK( my_first==mycoord_coord );
  CHECK( my_last==domain_coordinate_allones(dim)*(gsize-1-(ntids-1-mytid)) );

  SECTION( "right bump" ) { // VLE make alpha_block multi, and lose segment
    shiftop = ioperator(">=1");
    CHECK( shiftop.is_right_shift_op() );
    CHECK( !shiftop.is_modulo_op() );
    REQUIRE_NOTHROW( alpha_block = alpha->get_processor_structure(mycoord) );
    CHECK( alpha_block->is_strided() );
    CHECK( alpha_block->first_index_r()==my_first );
    CHECK( alpha_block->get_component(0)->stride()==ntids );
    REQUIRE_NOTHROW( segment = alpha_block->operate(shiftop,alpha->get_enclosing_structure()) );
    CHECK( segment->is_strided() );
    CHECK( segment->first_index_r()==my_first+1 );
    CHECK( segment->get_component(0)->stride()==ntids );
    CHECK_NOTHROW( msgs = alpha->messages_for_segment
		   ( 0,mycoord,/* skip self: */self_treatment::EXCLUDE,
		     segment,segment ) );
    CHECK( msgs.size()==1 );

    m = msgs.at(0);
    REQUIRE_NOTHROW( sender = m->get_sender() );
    if (mytid<ntids-1) {
      CHECK( sender==(mycoord+1) );
      REQUIRE_NOTHROW( ls = m->get_global_struct()->volume() );
      CHECK( ls==nlocal );
    } else {
      CHECK( sender==processor_coordinate_zero(dim) );
      REQUIRE_NOTHROW( ls = m->get_global_struct()->volume() );
      CHECK( ls==(nlocal-1) );
    }
    CHECK( m->get_global_struct()->first_index_r()==my_first+1 );
  }

  SECTION( "left bump" ) {
    shiftop = ioperator("<=1");
    CHECK( shiftop.is_left_shift_op() );
    CHECK( !shiftop.is_modulo_op() );
    shared_ptr<multi_indexstruct>
      alpha_block = alpha->get_processor_structure(mycoord);
    CHECK( alpha_block->volume()==nlocal );
    REQUIRE_NOTHROW( segment = alpha_block->operate(shiftop,alpha->get_enclosing_structure()) );
    CHECK_NOTHROW( msgs = alpha->messages_for_segment
		   ( 0,mycoord,self_treatment::INCLUDE,segment,segment ) );
    CHECK( msgs.size()==1 );
    REQUIRE_NOTHROW( m = msgs.at(0) );
    CHECK( m->get_sender().coord(0)==MOD(mytid-1,ntids) );
    if (mytid==0)
      CHECK( m->get_global_struct()->local_size_r().coord(0)==(nlocal-1) );
    else
      CHECK( m->get_global_struct()->local_size_r().coord(0)==nlocal );
    // if (mytid>0) {
    //   CHECK( msgs.size()==2 );
    // } else {
    // }
    // if (mytid>0) {
    //   for (int im=0; im<2; im++) {
    // 	m = msgs[im];
    // 	if ( m->get_sender().coord(0)==mytid-1 ) {
    // 	  CHECK( m->get_global_struct()->local_size_r().coord(0)==1 );
    // 	} else if ( m->get_sender().coord(0)==mytid ) {
    // 	  CHECK( m->get_global_struct()->local_size_r().coord(0)==(nlocal-1) );
    // 	} else {
    // 	  CHECK( 1==0 );
    // 	}
    //   }
    // } else {
    //   m = msgs[0];
    //   CHECK( m->get_sender().coord(0)==mytid );
    //   CHECK( m->get_global_struct()->local_size_r().coord(0)==(nlocal-1) );
    // }
  }
}

TEST_CASE( "Analyze one dependency modulo","[operate][dependence][modulo][16]") {

  INFO( "mytid=" << mytid );

  int localsize = 100, dim=1;
  shared_ptr<distribution> alpha = 
    shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,-1,localsize*ntids) );
  ioperator shiftop;
  vector<shared_ptr<message>> msgs; shared_ptr<message> m;
  auto
    my_first = alpha->first_index_r(mycoord),
    my_last = alpha->last_index_r(mycoord);
  index_int gsize = alpha->global_size().coord(0);
  shared_ptr<multi_indexstruct> segment,halo,alpha_block;

  SECTION( "right modulo" ) {
    shiftop = ioperator(">>1");
    CHECK( shiftop.is_right_shift_op() );
    CHECK( shiftop.is_modulo_op() );

    // my block in my alpha distribution
    REQUIRE_NOTHROW( alpha_block = alpha->get_processor_structure(mycoord) );
    CHECK( alpha_block->local_size_r().coord(0)==localsize );
    CHECK( alpha_block->first_index_r()==my_first );
    CHECK( alpha_block->last_index_r()==my_last );
    CHECK( (alpha_block->last_index_r()-alpha_block->first_index_r()+1)==
	   domain_coordinate_allones(dim)*localsize );

    // operated alpha block is shifted, can stick out beyond
    REQUIRE_NOTHROW( segment = alpha_block->operate(shiftop) );
    REQUIRE_NOTHROW( halo = alpha_block->struct_union(segment) );
    CHECK( segment->first_index_r()==my_first+1 );
    CHECK( segment->last_index_r()==my_last+1 );
    if (mytid==ntids-1) {
      CHECK( segment->last_index_r()>alpha_block->last_index_r() );
    }

    // the halo is the union my the alpha block and the operated alpha block
    CHECK( halo->first_index_r()==my_first );
    CHECK( halo->last_index_r()==my_last+1 );

    // it takes two messages to construct the halo
    INFO( "finding messages for segment " << segment->as_string()
	  << " on halo " << halo->as_string() );
    CHECK_NOTHROW( msgs = alpha->messages_for_segment
		   ( 0,mycoord,self_treatment::INCLUDE,segment,halo ) );
    CHECK( msgs.size()==2 );
    CHECK( ntids>1 ); // this test does not work for 1 task
    for (int imsg=0; imsg<2; imsg++) {
      auto m = msgs[imsg];
      shared_ptr<multi_indexstruct> glob,loc;
      REQUIRE_NOTHROW( glob = m->get_global_struct() );
      REQUIRE_NOTHROW( loc = m->get_local_struct() );
      int sender,receiver;
      REQUIRE_NOTHROW( receiver = m->get_receiver().coord(0) );
      REQUIRE_NOTHROW( sender = m->get_sender().coord(0) );
      CHECK( receiver==mytid );
      if (sender==mytid) {
	CHECK( glob->local_size_r().coord(0)==localsize-1 );
	CHECK( glob->first_index_r()==my_first+1 );
	CHECK( loc->first_index_r()==domain_coordinate_allones(dim) );
      } else {
	CHECK( sender==(mytid+1)%ntids );
	CHECK( glob->local_size_r().coord(0)==1 );
	CHECK( glob->first_index_r()==(my_last+1)%gsize );
	CHECK( loc->first_index_r()==domain_coordinate_allones(dim)*localsize );
      }
    }
  }

  SECTION( "left modulo" ) {
    shiftop = ioperator("<<1");
    CHECK( shiftop.is_left_shift_op() );
    CHECK( shiftop.is_modulo_op() );
    REQUIRE_NOTHROW( alpha_block = alpha->get_processor_structure(mycoord) );
    CHECK( alpha_block->local_size_r().coord(0)==localsize );
    REQUIRE_NOTHROW( segment = alpha_block->operate(shiftop) );
    REQUIRE_NOTHROW( halo = alpha_block->struct_union(segment) );
    CHECK( halo->first_index_r()==my_first-1 );
    CHECK( halo->last_index_r()==my_last );
    CHECK_NOTHROW( msgs = alpha->messages_for_segment
		   ( 0,mycoord,self_treatment::INCLUDE,segment,halo ) );
    CHECK( msgs.size()==2 );
    for (int imsg=0; imsg<2; imsg++) {
      auto m = msgs[imsg];
      int sender,receiver;
      REQUIRE_NOTHROW( receiver = m->get_receiver().coord(0) );
      REQUIRE_NOTHROW( sender = m->get_sender().coord(0) );
      shared_ptr<multi_indexstruct> glob,loc;
      REQUIRE_NOTHROW( glob = m->get_global_struct() );
      REQUIRE_NOTHROW( loc = m->get_local_struct() );
      if (sender==mytid) {
	CHECK( glob->local_size_r().coord(0)==localsize-1 );
	CHECK( glob->first_index_r()==my_first );
	CHECK( loc->first_index_r()==domain_coordinate_allones(dim) );
      } else {
	CHECK( sender==(mytid-1+ntids)%ntids );
	CHECK( glob->local_size_r().coord(0)==1 );
	CHECK( glob->first_index_r()==(my_first-1+gsize)%gsize );
	CHECK( loc->first_index_r()==domain_coordinate_allones(dim)*0 );
      }
    }
  }
}

TEST_CASE( "Analyze processor dependencies, right modulo, include self message",
	   "[index][message][modulo][17]") {
  INFO( "mytid=" << mytid );

  vector<shared_ptr<message>> msgs;
  shared_ptr<message> m,m0,m1;
  int localsize = 10, gsize = localsize*ntids, dim=1;
  auto
    no_op       = ioperator("none"),
    right_shift = ioperator(">>1"),
    left_shift  = ioperator("<<1");
  shared_ptr<distribution> d1 = 
    shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,-1,gsize) );
  int myfirst = mytid*localsize,mylast = (mytid+1)*localsize-1;
  shared_ptr<multi_indexstruct> betablock,h;

  // send from self to self: should deliver only one message
  h = shared_ptr<multi_indexstruct>
    ( new multi_indexstruct
      ( shared_ptr<indexstruct>( new contiguous_indexstruct(myfirst,mylast) ) ) );
  REQUIRE_NOTHROW
    ( betablock = d1->get_processor_structure(mycoord)->operate(no_op) );
  CHECK_NOTHROW( msgs = d1->messages_for_segment
		 ( 0,mycoord,self_treatment::INCLUDE,betablock,h ) );
  CHECK( msgs.size()==1 );
  m0 = msgs[0];
  REQUIRE( m0!=nullptr );
  CHECK( m0->get_receiver().coord(0)==mytid );
  CHECK( m0->get_sender().coord(0)==mytid );
  auto
    global_struct = m0->get_global_struct(),
    local_struct = m0->get_local_struct();
  CHECK( global_struct->first_index_r()==myfirst );
  CHECK( global_struct->last_index_r()==mylast );
  CHECK( local_struct->first_index_r()==domain_coordinate_allones(dim)*0 );
  CHECK( local_struct->last_index_r()==domain_coordinate_allones(dim)*localsize-1 );

  // something non-trivial
  h = shared_ptr<multi_indexstruct>
    (new multi_indexstruct
     ( shared_ptr<indexstruct>( new contiguous_indexstruct(myfirst,mylast+1) ) ) );
  REQUIRE_NOTHROW
    ( betablock = d1->get_processor_structure(mycoord)->operate(right_shift) );
  CHECK_NOTHROW( msgs = d1->messages_for_segment
		 ( 0,mycoord,self_treatment::INCLUDE,betablock,h ) );
  // there are two messages because of the module shift
  CHECK( msgs.size()==2 );
  for (int imsg=0; imsg<2; imsg++) {
    auto m = msgs[imsg];
    REQUIRE( m!=nullptr );
    CHECK( m->get_receiver().coord(0)==mytid );
    int s = m->get_sender().coord(0);
    if (mytid==ntids-1) {
      CHECK( ( s==0 || s==mytid ) );
    } else {
      CHECK( ( s==mytid || s==mytid+1 ) );
    }
    auto 
      src = m->get_global_struct(), tar = m->get_local_struct();
    if (s==mytid) {
      // first message is shift of my local data
      CHECK( src->first_index_r()==myfirst+1 );
      CHECK( src->last_index_r()==mylast );
      CHECK( tar->first_index_r()==domain_coordinate_allones(dim) );
      CHECK( tar->last_index_r()==domain_coordinate_allones(dim)*localsize-1 );
    } else {
      // second msg gets one element from my right neighbour
      CHECK( src->first_index_r()==(mylast+1)%gsize );
      CHECK( src->last_index_r()==(mylast+1)%gsize );
      CHECK( tar->first_index_r()==domain_coordinate_allones(dim)*localsize );
      CHECK( tar->last_index_r()==domain_coordinate_allones(dim)*localsize );
    }
  }
}

TEST_CASE( "Analyze processor dependencies, modulo, skip self",
	   "[index][operate][halo][message][modulo][18]") {
  int localsize = 10, gsize = localsize*ntids, dim=1;
  auto
    no_op       = ioperator("none"),
    right_shift = ioperator(">>1"),
    left_shift  = ioperator("<<1");
  shared_ptr<distribution> d1 = 
    shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,-1,gsize) );
  int
    myfirst = mytid*localsize,mylast = (mytid+1)*localsize-1;
  shared_ptr<indexstruct> h;
  vector<shared_ptr<message>> msgs;
  shared_ptr<message> m,m0;
  int s0;
  shared_ptr<multi_indexstruct> src0,tar0,betablock;

  INFO( "mytid=" << mytid );

  SECTION( "send from self to self: should deliver only one message" ) {
    h = shared_ptr<indexstruct>( new contiguous_indexstruct(0,gsize) );
    betablock = d1->get_processor_structure(mycoord)->operate(no_op);
    CHECK_NOTHROW( msgs = d1->messages_for_segment
		   ( 0,mycoord,self_treatment::EXCLUDE,betablock,
		     make_shared<multi_indexstruct>(h) ) );
    //shared_ptr<multi_indexstruct>( new multi_indexstruct(h)) ) );
    CHECK( msgs.size()==0 ); // skip self: nothing remains
    h = shared_ptr<indexstruct>( new contiguous_indexstruct(0,gsize) );
    betablock = d1->get_processor_structure(mycoord)->operate(no_op);
    CHECK_NOTHROW( msgs = d1->messages_for_segment
		   ( 0,mycoord,self_treatment::INCLUDE,
		     betablock,make_shared<multi_indexstruct>(h) ) );
    REQUIRE( msgs.size()==1 ); // this is the real case
    m0 = msgs[0];
    auto
      global_struct = m0->get_global_struct(),
      local_struct = m0->get_local_struct();
    auto
      f = global_struct->first_index_r(),
      l = global_struct->last_index_r();
    CHECK( f==myfirst );
    CHECK( l==mylast );
    CHECK( m0->get_receiver().coord(0)==mytid );
    CHECK( m0->get_sender().coord(0)==mytid );
  }

  SECTION( "something non-trivial: right shift" ) {
    h = shared_ptr<indexstruct>( new contiguous_indexstruct(myfirst,mylast+1) );
    betablock = d1->get_processor_structure(mycoord)->operate(right_shift);
    CHECK_NOTHROW( msgs = d1->messages_for_segment
		   ( 0,mycoord,self_treatment::EXCLUDE,
		     betablock,make_shared<multi_indexstruct>(h) ) );
    //shared_ptr<multi_indexstruct>( new multi_indexstruct(h)) ) );
    // there is one message for everyone, since we're skipping self
    REQUIRE( msgs.size()==1 );
    m0 = msgs[0];
    CHECK( m0->get_sender().coord(0)==(mytid+1)%ntids );
    CHECK( m0->get_receiver().coord(0)==mytid );
    src0 = m0->get_global_struct(); tar0 = m0->get_local_struct();
    // msg gets one element from my right neighbour, global coordinates, wrapped
    CHECK( src0->first_index_r()==(mylast+1)%gsize );
    CHECK( src0->last_index_r()==(mylast+1)%gsize );
    // winds up in the last location of the halo regardless
    CHECK( tar0->first_index_r()==domain_coordinate_allones(dim)*localsize );
    CHECK( tar0->last_index_r()==domain_coordinate_allones(dim)*localsize );
  }

  SECTION( "something non-trivial: left shift" ) {
    h = shared_ptr<indexstruct>( new contiguous_indexstruct(myfirst-1,mylast) );
    betablock = d1->get_processor_structure(mycoord)->operate(left_shift);
    CHECK_NOTHROW( msgs = d1->messages_for_segment
		   ( 0,mycoord,self_treatment::EXCLUDE,
		     betablock,make_shared<multi_indexstruct>(h) ) );
    //shared_ptr<multi_indexstruct>( new multi_indexstruct(h)) ) );
    // there is one message for everyone
    REQUIRE( msgs.size()==1 );
    m0 = msgs[0];
    CHECK( m0->get_sender().coord(0)==MOD(mytid-1,ntids) );
    CHECK( m0->get_receiver().coord(0)==mytid );
    src0 = m0->get_global_struct(); tar0 = m0->get_local_struct();
    // msg gets one element from my left neighbour
    CHECK( src0->first_index_r()[0]==MOD(myfirst-1,gsize) );
    CHECK( src0->last_index_r()==src0->first_index_r() );
    // winds up in the first location of the halo regardless
    CHECK( tar0->first_index_r()==domain_coordinate_allones(dim)*0 );
    CHECK( tar0->last_index_r()==domain_coordinate_allones(dim)*0 );
  }
}

TEST_CASE( "Analyze processor dependencies, left bump","[index][message][19]") {
  vector<shared_ptr<message>> msgs;
  shared_ptr<message> m0,m1;
  auto
    no_op       = ioperator("none"),
    right_shift = ioperator(">=1"),
    left_shift  = ioperator("<=1");
  int localsize = 10, gsize = localsize*ntids, dim=1;
  shared_ptr<distribution> d1 = 
    shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,-1,gsize) );
  auto
    myfirst = mycoord_coord*localsize,
    mylast = (mycoord_coord+1)*localsize-1;

  shared_ptr<multi_indexstruct> betablock,h;
  { // halo sticks out left and right, when those points exist
    auto f=myfirst,l=mylast;
    if (mytid>0)
      f = f-1;
    if (mytid<ntids-1)
      l = l+1;
    h = shared_ptr<multi_indexstruct>( new contiguous_multi_indexstruct( f,l ) );
  }

  if (mytid==0 && mytid==ntids-1) {
    CHECK( h->local_size_r().coord(0)==localsize );
  } else if (mytid==0 || mytid==ntids-1) {
    CHECK( h->local_size_r().coord(0)==localsize+1 );
  } else {
    CHECK( h->local_size_r().coord(0)==localsize+2 );
  }
  REQUIRE_NOTHROW( betablock = d1->get_processor_structure(mycoord)->operate
		   (left_shift,d1->get_enclosing_structure()) );
  CHECK_NOTHROW( msgs = d1->messages_for_segment
		 ( 0,mycoord,self_treatment::INCLUDE,
		   betablock,h ) );

  // there are two messages because of the module shift
  if (mytid==0) {
    CHECK( msgs.size()==1 );
  } else {
    CHECK( msgs.size()==2 );
    m1 = msgs[1];
  }
  m0 = msgs[0]; 
  CHECK( m0->get_receiver().coord(0)==mytid );

  int s0,s1;
  s0 = m0->get_sender().coord(0);

  if (mytid==0) {
    shared_ptr<multi_indexstruct> src0,tar0,src1,tar1;
    src0 = m0->get_global_struct();
    tar0 = m0->get_local_struct();

    CHECK( s0==mytid );
    CHECK( src0->first_index_r()==myfirst );
    CHECK( src0->last_index_r()==mylast-1 );
    CHECK( tar0->first_index_r()==domain_coordinate_allones(dim)*0 ); // isn't this supposed to start at 1?
    CHECK( tar0->last_index_r()==domain_coordinate_allones(dim)*localsize-2 );
  } else { // for everyone else there is a point coming from left
    for (int im=0; im<2; im++) {
      auto m = msgs[im];
      auto
	src = m->get_global_struct(),
	tar = m->get_local_struct();

      CHECK( m->get_receiver().coord(0)==mytid );
      if ( m->get_sender().coord(0)==mytid-1 ) {
	// single element from left neighbour
	CHECK( src->first_index_r()==myfirst-1 );
	CHECK( src->last_index_r()==myfirst-1 );
	CHECK( tar->first_index_r()==domain_coordinate_allones(dim)*0 );
	CHECK( tar->last_index_r()==domain_coordinate_allones(dim)*0 );
      } else if (m->get_sender().coord(0)==mytid ) {
	// then shift of local
	auto len = (mylast-1)-myfirst+1;
	CHECK( src->first_index_r()==myfirst );
	CHECK( src->last_index_r()==myfirst+len-1 );
	CHECK( tar->first_index_r()==domain_coordinate_allones(dim) );
	CHECK( tar->last_index_r()==domain_coordinate_allones(dim)+len-1 );
      } else {
	CHECK( 1==0 );
      }      
    }
  }
}

TEST_CASE( "messages from parallel indexstruct","[indexstruct][message][20]" ) {
  parallel_structure pstr;
  REQUIRE_NOTHROW( pstr = parallel_structure( decomp ) );

  int base = 0; const char *construct;
  SECTION( "ordinary zero-based" ) { construct = "global";
    REQUIRE_NOTHROW( pstr.create_from_global_size( ntids ) );
  }
  SECTION( "from contiguous" ) { construct = "zero-based";
    REQUIRE_NOTHROW( pstr.create_from_indexstruct
		     ( shared_ptr<indexstruct>( new contiguous_indexstruct(0,ntids-1) ) ) );
  }
  SECTION( "from shifted contiguous" ) { construct = "one-based";
    base = 1;
    REQUIRE_NOTHROW( pstr.create_from_indexstruct
		     ( shared_ptr<indexstruct>( new contiguous_indexstruct(1,ntids) ) ) );
  }
  INFO( "pstr constructed: " << construct << ": " << pstr.as_string() );

  shared_ptr<distribution> dist = shared_ptr<distribution>( new mpi_distribution(/*decomp,*/pstr) );
  {
    shared_ptr<multi_indexstruct> localstruct =
      shared_ptr<multi_indexstruct>
      ( new multi_indexstruct
	( shared_ptr<indexstruct>( new contiguous_indexstruct(mytid+base) ) ) );
    INFO( "local struct: " << dist->get_processor_structure(mycoord)->as_string() << "\n" <<
	  "should be: " << localstruct->as_string() );
    CHECK( pstr.get_processor_structure(mycoord)->equals(localstruct) );
    CHECK( dist->get_processor_structure(mycoord)->equals(localstruct) );
  }
  shared_ptr<multi_indexstruct> beta;
  vector<shared_ptr<message>> msgs;

  REQUIRE_NOTHROW
    ( beta = shared_ptr<multi_indexstruct>
      ( new multi_indexstruct
	( shared_ptr<indexstruct>( new contiguous_indexstruct( mytid+base,mytid+base ) ) ) ));
  REQUIRE_NOTHROW
    ( msgs = dist->messages_for_segment
      ( 0,mycoord,self_treatment::INCLUDE,beta,beta ) );
  CHECK( msgs.size()==1 );
  
  index_int *idx = new index_int[1]; idx[0] = mytid+base;
  REQUIRE_NOTHROW
    ( beta = shared_ptr<multi_indexstruct>
      ( new multi_indexstruct
	( shared_ptr<indexstruct>( new indexed_indexstruct( 1,idx ) ) ) ));
  REQUIRE_NOTHROW
    ( msgs = dist->messages_for_segment
      ( 0,mycoord,self_treatment::INCLUDE,beta,beta ) );
  CHECK( msgs.size()==1 );
}

TEST_CASE( "Task dependencies threepoint bump","[task][message][object][21]" ) {

  INFO( "mytid: " << mytid );
  // create distributions and objects for threepoint combination
  ioperator no_op("none"), right_shift(">=1"), left_shift("<=1");

  shared_ptr<distribution> d1; index_int gsize = 10*ntids;
  REQUIRE_NOTHROW
    ( d1 = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,gsize) ) );
  index_int v;
  REQUIRE_NOTHROW( v = d1->get_dimension_structure(0)->pidx_domains_volume() );
  CHECK( v==ntids );

  shared_ptr<object> o1,r1;
  CHECK_NOTHROW( o1 = d1->new_object(d1) ); //new mpi_object(d1) );
  CHECK_NOTHROW( r1 = d1->new_object(d1) ); //new mpi_object(d1) );

  // declare the beta vectors
  signature_function sigma_opers;
  CHECK_NOTHROW( sigma_opers.add_sigma_operator( no_op ) );
  CHECK_NOTHROW( sigma_opers.add_sigma_operator( left_shift ) );
  CHECK_NOTHROW( sigma_opers.add_sigma_operator( right_shift ) );
  CHECK( sigma_opers.get_cookie()==entity_cookie::SIGNATURE );
  CHECK( d1->get_cookie()==entity_cookie::DISTRIBUTION );
  CHECK( r1->get_cookie()==entity_cookie::OBJECT );

  kernel combine;
  try {
    parallel_structure preds;
    shared_ptr<multi_indexstruct> enclosing;
    REQUIRE_NOTHROW( enclosing = d1->get_enclosing_structure() );
    {
      shared_ptr<multi_indexstruct> bigstruct;
      REQUIRE_NOTHROW( bigstruct = shared_ptr<multi_indexstruct>
		       ( new multi_indexstruct
			 ( shared_ptr<indexstruct>
			   ( new contiguous_indexstruct(0,gsize-1) ) ) ) );
      INFO( fmt::format("Enclosing is <<{}>>\nshould be <<{}>>",
			enclosing->as_string(),bigstruct->as_string()) );
      CHECK( enclosing->equals(bigstruct) );
    }
    REQUIRE_NOTHROW( preds = sigma_opers.derive_beta_structure(d1,enclosing) );
    CHECK( !preds.has_type(distribution_type::UNDEFINED) );
    shared_ptr<distribution> beta_dist;
    CHECK_NOTHROW( beta_dist = shared_ptr<distribution>( new mpi_distribution(preds) ) );
    REQUIRE_NOTHROW( beta_dist->get_visibility(mycoord) );

    CHECK_NOTHROW( combine = mpi_kernel(o1,r1) );
    CHECK( combine.get_cookie()==entity_cookie::KERNEL );
    { // sanity check
      CHECK( o1->get_dimensionality()==1 );
      CHECK( d1->get_dimension_structure(0)->pidx_domains_volume()==ntids );
    }
    CHECK_NOTHROW( combine.set_explicit_beta_distribution( beta_dist ) );
    shared_ptr<distribution> halo_dist;
    REQUIRE_NOTHROW( halo_dist= combine.set_last_dependency().find_beta_distribution(r1) );
    REQUIRE_NOTHROW( combine.set_last_dependency().create_beta_vector(halo_dist) );
  } catch (std::string c) { fmt::print("[21] kernel stuff failed: {}\n",c); return;
  } catch (...) { fmt::print("[21] kernel stuff failed\n"); return; }
  
  return;
  // check on whether we record stuff....
  int n_entities;
  REQUIRE_NOTHROW( n_entities = env->n_entities() );
  int n_task;
  {
    result_tuple *summary;
    REQUIRE_NOTHROW( summary = env->mode_summarize_entities() );
    n_task = std::get<RESULT_TASK>(*summary);
  }
  REQUIRE_NOTHROW( combine.split_to_tasks() );
  {
    result_tuple *summary;
    REQUIRE_NOTHROW( summary = env->mode_summarize_entities() );
    n_task = std::get<RESULT_TASK>(*summary) - n_task;
  }
  CHECK( n_task==ntids );
  CHECK( env->n_entities()==n_entities+1 );

  // back to business .....
  shared_ptr<task> combine_task;
  CHECK( combine.get_tasks().size()>0 );
  REQUIRE_NOTHROW( combine_task = combine.get_tasks().at(0) );
  REQUIRE_NOTHROW( combine_task->derive_receive_messages() );
  { // check that the created object has proper lambdas installed
    shared_ptr<object> halo;
    REQUIRE_NOTHROW( halo = combine_task->get_beta_object(0) );
    REQUIRE_NOTHROW( halo->get_visibility(mycoord) );
  }

  CHECK( combine.get_cookie()==entity_cookie::KERNEL );
  CHECK( combine_task->get_cookie()==entity_cookie::TASK );
  result_tuple *results; int ntasks;
  // REQUIRE_NOTHROW( results = env->mode_summarize_entities() );
  // REQUIRE_NOTHROW( ntasks = std::get<RESULT_TASK>(*results) );
  // CHECK( ntasks > 0 );

}

TEST_CASE( "Task dependencies threepoint modulo","[task][message][22][modulo]" ) {

  INFO( "mytid=" << mytid );
  if (ntids<=2) { printf("Test 22 needs more than 2 procs\n"); return; }

  // create distributions and objects for threepoint combination
  ioperator no_op("none"), right_shift(">>1"), left_shift("<<1");
  int nlocal = 10,gsize = nlocal*ntids;
  shared_ptr<distribution> d1 = 
    shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,-1,gsize) );
  auto
    my_first = d1->first_index_r(mycoord),
    my_last = d1->last_index_r(mycoord);
  index_int nglobal = d1->global_size().coord(0);
  shared_ptr<object> o1,r1;
  CHECK_NOTHROW( r1 = d1->new_object(d1) ); //new mpi_object(d1) );
  CHECK_NOTHROW( o1 = d1->new_object(d1) ); //new mpi_object(d1) );

  // declare the beta vectors
  signature_function sigma_opers;
  sigma_opers.add_sigma_operator( no_op );
  sigma_opers.add_sigma_operator( left_shift );
  sigma_opers.add_sigma_operator( right_shift );

  parallel_structure preds;
  shared_ptr<multi_indexstruct> enclosing;
  REQUIRE_NOTHROW( enclosing = r1->get_enclosing_structure() );
  REQUIRE_NOTHROW( preds = sigma_opers.derive_beta_structure(d1,enclosing) );
  shared_ptr<distribution> beta_dist;
  CHECK_NOTHROW( beta_dist = shared_ptr<distribution>( new mpi_distribution(preds) ) );
  INFO( "derived beta: " << beta_dist->as_string() );
  CHECK( beta_dist->global_volume()==nglobal+2 ); // outer_size
  CHECK( beta_dist->global_volume()==nglobal+2 );
  CHECK( beta_dist->global_first_index().coord(0)==-1 );
  CHECK( beta_dist->global_last_index().coord(0)==nglobal );

  kernel combine;
  CHECK_NOTHROW( combine = mpi_kernel(o1,r1) );
  CHECK_NOTHROW( combine.set_explicit_beta_distribution( beta_dist ) );
  shared_ptr<distribution> halo_dist;
  REQUIRE_NOTHROW( halo_dist= combine.set_last_dependency().find_beta_distribution(r1) );
  REQUIRE_NOTHROW( combine.create_beta_vector(halo_dist) );

  REQUIRE_NOTHROW( combine.split_to_tasks() );
  shared_ptr<task> combine_task;
  REQUIRE_NOTHROW( combine_task = combine.get_tasks().at(0) );
  REQUIRE_NOTHROW( combine_task->derive_receive_messages() );

  // one message from self, two neighbours
  CHECK( combine_task->get_receive_messages().size()==3 );
  for ( auto m : combine_task->get_receive_messages() ) {
    auto src = m->get_global_struct(),tar = m->get_local_struct();
    domain_coordinate f = src->first_index_r(), l = src->last_index_r();
    auto
      sf = src->first_index_r(),sl = src->last_index_r(), // global coordinates
      tf = tar->first_index_r(),tl = tar->last_index_r(), // local to halo
      slen = sl-sf+1,tlen = tl-tf+1;
    if (m->get_sender().coord(0)==MOD(mytid+1,ntids)) { // From The Right
      CHECK( sf==MOD(my_last+1,nglobal) ); // stick out right, modulo
      CHECK( slen==1 );
      CHECK( tf==nlocal+1 );
      CHECK( tlen==1 );
    } else if (m->get_sender().coord(0)==MOD(mytid-1,ntids)) { // From The Left
      CHECK( sf==MOD(my_first-1,nglobal) ); // stick out left, modulo
      CHECK( slen==1 );
      CHECK( tf==0 );
      CHECK( tlen==1 );
    } else {
      CHECK( m->get_sender().coord(0)==mytid );
      CHECK( sf==my_first ); 
      CHECK( slen==nlocal );
      CHECK( tf==1 );
      CHECK( tlen==nlocal );
    }
    CHECK( d1->contains_element(m->get_sender(),f) );
    CHECK( d1->contains_element(m->get_sender(),l) );
  }
}

TEST_CASE( "Task send structure threepoint bump","[task][message][object][23]" ) {

  INFO( "mytid=" << mytid );
  // create distributions and objects for threepoint combination
  ioperator no_op("none"), right_shift(">=1"), left_shift("<=1");

  int nlocal = 10,gsize = nlocal*ntids;
  shared_ptr<distribution> d1;
  REQUIRE_NOTHROW
    ( d1 = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,-1,gsize) ) );
  int
    myfirst = d1->first_index_r(mycoord).coord(0),
    mylast = d1->last_index_r(mycoord).coord(0);
  shared_ptr<object> o1,r1;
  REQUIRE_NOTHROW( r1 = d1->new_object(d1) ); //new mpi_object(d1) );
  REQUIRE_NOTHROW( o1 = d1->new_object(d1) ); //new mpi_object(d1) );
  CHECK( r1->get_enclosing_structure()->volume()==gsize );
  CHECK( o1->get_enclosing_structure()->volume()==gsize );

  shared_ptr<multi_indexstruct> enclosing;
  signature_function sigma_opers;
  parallel_structure preds;

  REQUIRE_NOTHROW( enclosing = o1->get_enclosing_structure() );
  SECTION( "no-op beta" ) {
    sigma_opers.add_sigma_operator( no_op );
    REQUIRE_NOTHROW( preds = sigma_opers.derive_beta_structure(d1,enclosing) );
    CHECK( !preds.has_type(distribution_type::UNDEFINED) );

    shared_ptr<distribution> beta_dist;
    CHECK_NOTHROW( beta_dist =
		   //make_shared<distribution>( new mpi_distribution(preds) )
		   shared_ptr<distribution>( make_shared<mpi_distribution>(preds) )
		   );
    beta_dist->set_name("beta from preds");
    CHECK( !beta_dist->has_type(distribution_type::UNDEFINED) );

    auto originate = shared_ptr<kernel>( new mpi_kernel(o1,r1) );
    CHECK_NOTHROW( originate->set_explicit_beta_distribution(beta_dist) );
    CHECK( originate->last_dependency().has_type_explicit_beta() );
    SECTION( "kernel analysis in pieces" ) {
      CHECK( !originate->kernel_has_tasks() );
      CHECK( !originate->get_has_been_analyzed() );
      REQUIRE_NOTHROW( originate->split_to_tasks() );
      shared_ptr<object> outobject;
      REQUIRE_NOTHROW( outobject = originate->get_out_object() );
      REQUIRE_NOTHROW( originate->endow_beta_objects(outobject) );

      vector< shared_ptr<task> > tasks;
      REQUIRE_NOTHROW( tasks = originate->get_tasks() );
      for ( auto &t : tasks ) {
	vector<shared_ptr<message>> rmsgs;
	REQUIRE_NOTHROW( rmsgs = t->derive_receive_messages() );
	for ( auto m : rmsgs ) {
	  REQUIRE_NOTHROW( m->get_in_object() );
	}
	REQUIRE_NOTHROW( t->set_receive_messages(rmsgs) );
	// VLE we're spelling out: REQUIRE_NOTHROW( t->analyze_dependencies() );
      }

      // shared_ptr<distribution> halo_dist;
      // REQUIRE_NOTHROW( halo_dist= originate->set_last_dependency().find_beta_distribution(r1) );
      // REQUIRE_NOTHROW( originate->create_beta_vector(halo_dist) );
    }
    SECTION( "kernel analysis as call" ) {
      REQUIRE_NOTHROW( originate->analyze_dependencies() );
    }
    // sanity check on tasks
    {
      shared_ptr<task> originate_task;
      REQUIRE_NOTHROW( originate_task = originate->get_tasks().at(0) );
      int kout,tout;
      REQUIRE_NOTHROW( kout = originate->get_out_object()->get_object_number() );
      REQUIRE_NOTHROW( tout = originate_task->get_out_object()->get_object_number() );
      REQUIRE( tout==kout );
    }
  }

  SECTION( "more stuff in the beta" ) {
    shared_ptr<distribution> beta_dist;
    sigma_opers.add_sigma_operator( no_op );
    sigma_opers.add_sigma_operator( left_shift );
    sigma_opers.add_sigma_operator( right_shift );
    REQUIRE_NOTHROW( preds = sigma_opers.derive_beta_structure(d1,enclosing) );
    CHECK( !preds.get_processor_structure(mycoord)->is_empty() );
    CHECK_NOTHROW( beta_dist = shared_ptr<distribution>( new mpi_distribution(preds) ) );
    {
      index_int bfirst,blast;
      REQUIRE_NOTHROW( bfirst = beta_dist->first_index_r(mycoord).coord(0) );
      REQUIRE_NOTHROW( blast = beta_dist->last_index_r(mycoord).coord(0) );
      if (mytid==0)
	CHECK( bfirst==myfirst );
      else
	CHECK( bfirst==myfirst-1 );
      if (mytid==ntids-1)
	CHECK( blast==mylast );
      else
	CHECK( blast==mylast+1 );
    }
    {
      domain_coordinate
	myfirst = d1->first_index_r(mycoord),
	mylast = d1->last_index_r(mycoord);
      domain_coordinate bfirst,blast;
      REQUIRE_NOTHROW( bfirst = beta_dist->first_index_r(mycoord) );
      REQUIRE_NOTHROW( blast = beta_dist->last_index_r(mycoord) );
      if (mytid==0)
	CHECK( bfirst==myfirst );
      else
	CHECK( bfirst==myfirst-1 );
      if (mytid==ntids-1)
	CHECK( blast==mylast );
      else
	CHECK( blast==mylast+1 );
    }

    kernel combine = mpi_kernel(o1,r1);
    CHECK_NOTHROW( combine.set_last_dependency().set_explicit_beta_distribution(beta_dist) );
    //REQUIRE_NOTHROW( combine.create_beta_vector(r1->get_distribution()) );
    REQUIRE_NOTHROW( combine.analyze_dependencies() );

    shared_ptr<task> combine_task;
    REQUIRE_NOTHROW( combine_task = combine.get_tasks().at(0) );
    vector<shared_ptr<message>> msgs;
    REQUIRE_NOTHROW( msgs = combine_task->get_receive_messages() );
    if (ntids==1) {
      CHECK( msgs.size()==1 );
    } else if (mytid==0 || mytid==ntids-1) {
      CHECK( msgs.size()==2 );
    } else {
      CHECK( msgs.size()==3 );
    }
    
    int nsends;
    REQUIRE_NOTHROW( nsends = combine_task->get_nsends() );
    if (ntids==1) {
      CHECK( nsends==1 );
    } else if (mytid==0 || mytid==ntids-1) {
      CHECK( nsends==2 );
    } else {
      CHECK( nsends==3 );
    }
  
    for ( auto m : combine_task->get_receive_messages() ) {
      auto src = m->get_global_struct(),tar = m->get_local_struct();
      domain_coordinate
  	sf = src->first_index_r(), sl = src->last_index_r(),
  	tf = tar->first_index_r(), tl = tar->last_index_r();
      CHECK( d1->contains_element(m->get_sender(),sf) );
      CHECK( d1->contains_element(m->get_sender(),sl) );
      // src coordinates are global
      if (m->get_sender().coord(0)==mytid-1) {
  	CHECK( sf==d1->first_index_r(mycoord)-1 );
  	CHECK( sl==sf );
  	CHECK( src->local_size_r().coord(0)==1 );
      } else if (m->get_sender().coord(0)==mytid+1) {
  	CHECK( sf==d1->last_index_r(mycoord)+1 );
  	CHECK( sl==sf );
  	CHECK( src->local_size_r().coord(0)==1 );
      } else {
  	CHECK( m->get_sender().coord(0)==mytid );
  	CHECK( sf==d1->first_index_r(mycoord) );
  	CHECK( sl==d1->last_index_r(mycoord) );
  	CHECK( src->local_size_r().coord(0)==nlocal );
      }
      // tar coordinates are wrt the halo
      int hasleft = (mytid>0);
      if (m->get_sender().coord(0)==mytid-1) {
  	CHECK( tf==domain_coordinate(vector<index_int>{0}) );
  	CHECK( tl==tf );
  	CHECK( tar->local_size_r().coord(0)==1 );
      } else if (m->get_sender().coord(0)==mytid+1) {
  	CHECK( tf==domain_coordinate(vector<index_int>{nlocal+hasleft}) );
  	CHECK( tl==tf );
  	CHECK( tar->local_size_r().coord(0)==1 );
      } else {
  	CHECK( tf==domain_coordinate(vector<index_int>{hasleft}) );
  	CHECK( tl==domain_coordinate(vector<index_int>{hasleft+nlocal-1}) );
  	CHECK( tar->local_size_r().coord(0)==nlocal );
      }
    }
  }

}

TEST_CASE( "Task send structure threepoint modulo","[task][message][object][modulo][24]" ) {

  // create distributions and objects for threepoint combination

  ioperator no_op("none"), right_shift(">>1"), left_shift("<<1");

  shared_ptr<distribution> d1 = 
    shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,-1,10*ntids) );
  // a couple of tests that we will apply to the beta distribution
  CHECK_NOTHROW( d1->get_dimension_structure(0)->get_processor_structure(0) );
  CHECK_NOTHROW( d1->get_processor_structure(mycoord) );

  shared_ptr<object> o1,r1;
  CHECK_NOTHROW( r1 = d1->new_object(d1) ); //new mpi_object(d1) );
  CHECK_NOTHROW( o1 = d1->new_object(d1) ); //new mpi_object(d1) );

  kernel combine = mpi_kernel(o1,r1);
  REQUIRE_NOTHROW( combine.split_to_tasks() );
  shared_ptr<task> combine_task;
  REQUIRE_NOTHROW( combine_task = combine.get_tasks().at(0) );

  // SECTION( "spell out the beta construction" ) {

  //   // declare the beta vectors
  //   signature_function sigma_opers;
  //   CHECK_NOTHROW( sigma_opers.add_sigma_operator( no_op ) );
  //   CHECK_NOTHROW( sigma_opers.add_sigma_operator( left_shift ) );
  //   CHECK_NOTHROW( sigma_opers.add_sigma_operator( right_shift ) );

  //   parallel_structure preds;
  //   shared_ptr<multi_indexstruct> enclosing;
  //   REQUIRE_NOTHROW( enclosing = o1->get_enclosing_structure() );
  //   REQUIRE_NOTHROW( preds = sigma_opers.derive_beta_structure(d1,enclosing) );
  //   shared_ptr<distribution> beta_dist;
  //   CHECK_NOTHROW( beta_dist = shared_ptr<distribution>( new mpi_distribution(/*decomp,*/preds) ) );
  //   // check wellformedness of the beta distribution and beta opers
  //   CHECK( beta_dist->domains_volume()==ntids );
  //   CHECK_NOTHROW( beta_dist->get_dimension_structure(0)->get_processor_structure(0) );
  //   CHECK_NOTHROW( beta_dist->get_processor_structure(mycoord) );
  //   // set the beta definition in the task
  //   CHECK_NOTHROW( combine.set_beta_distribution( 0,beta_dist ) );
  // }
  SECTION( "in one fell swoop" ) {
    REQUIRE_NOTHROW( combine_task->set_last_dependency() );
    auto dep = combine_task->set_last_dependency();
    // declare the beta 
    CHECK_NOTHROW( dep.add_sigma_operator( no_op ) );
    CHECK_NOTHROW( dep.add_sigma_operator( left_shift ) );
    CHECK_NOTHROW( dep.add_sigma_operator( right_shift ) );

    CHECK_NOTHROW( combine.endow_beta_objects(combine.get_out_object()) );
  }

  // see what is coming on
  REQUIRE_NOTHROW( combine_task->derive_receive_messages() );
  
  // and for the mirror, what is going out
  CHECK_NOTHROW( combine_task->derive_send_messages() );
  vector<shared_ptr<message>> msgs;
  REQUIRE_NOTHROW( msgs = combine_task->get_send_messages() );
  CHECK( msgs.size()==3 );

  index_int sf,sl;
  //for (vector<shared_ptr<message>>::iterator m=msgs.begin(); m!=msgs.end(); ++m) {
  for ( auto m : msgs ) {
    auto glb = m->get_global_struct();
    sf = glb->first_index_r().coord(0);
    sl = glb->last_index_r().coord(0);
    CHECK( sf>=d1->first_index_r(mycoord).coord(0)-1 );
    CHECK( sl<=d1->last_index_r(mycoord).coord(0)+1 );
    auto loc = m->get_local_struct();
    sf = loc->first_index_r().coord(0);
    sl = loc->last_index_r().coord(0);
    CHECK( sf>=0 );
    CHECK( sl<d1->volume( m->get_sender() ) );
  }
}

TEST_CASE( "Halo object for task modulo","[task][halo][modulo][25]" ) {

  // create distributions and objects for threepoint combination
  ioperator no_op("none"), right_shift(">>1"), left_shift("<<1");
  shared_ptr<distribution> d1 = 
    shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,-1,10*ntids) );
  shared_ptr<object> o1,r1;
  std::string r1name,o1name;
  r1name.assign("result_vector_1"); o1name.assign("origin_vector_1");
  CHECK_NOTHROW( r1 = d1->new_object(d1) ); //new mpi_object(d1) ); // result vector
  r1->set_name( r1name );
  CHECK( r1->get_name().compare(r1name)==0 );

  CHECK_NOTHROW( o1 = d1->new_object(d1) ); //new mpi_object(d1) ); // origin vector
  o1->set_name( o1name );

  parallel_structure preds;
  shared_ptr<object> halo;
  kernel combine = mpi_kernel(o1,r1);
  REQUIRE_NOTHROW( combine.split_to_tasks() );
  shared_ptr<task> combine_task;
  REQUIRE_NOTHROW( combine_task = combine.get_tasks().at(0) );

  REQUIRE_NOTHROW( combine.set_last_dependency() );
  auto dep = combine.set_last_dependency();

  SECTION( "no-op" ) {
    CHECK_NOTHROW( dep.add_sigma_operator( no_op ) );
    CHECK_NOTHROW( combine.endow_beta_objects(combine_task->get_out_object()) );

    halo = combine.get_beta_object(0);
    CHECK( r1->get_name().compare(r1name)==0 );
    CHECK( combine.get_out_object()->get_name().compare(r1name)==0 );
    CHECK( combine.get_out_object()->volume(mycoord)==10 );
    CHECK( halo!=nullptr );
    CHECK( halo->volume(mycoord)==10 );
  }
  SECTION( "no-op and left" ) {
    CHECK_NOTHROW( dep.add_sigma_operator( no_op ) );
    CHECK_NOTHROW( dep.add_sigma_operator( left_shift ) );
    CHECK_NOTHROW( combine.endow_beta_objects(combine_task->get_out_object()) );

    halo = combine_task->get_beta_object(0);
    //    CHECK( combine_task->has_outvector() );
    CHECK( combine.get_out_object()->volume(mycoord)==10 );
    CHECK( halo!=nullptr );
    CHECK( halo->volume(mycoord)==11 );
  }
  SECTION( "no-op and right" ) {
    CHECK_NOTHROW( dep.add_sigma_operator( no_op ) );
    CHECK_NOTHROW( dep.add_sigma_operator( right_shift ) );
    CHECK_NOTHROW( combine.endow_beta_objects(combine_task->get_out_object()) );

    halo = combine.get_beta_object(0);
    CHECK( combine.get_out_object()->volume(mycoord)==10 );
    CHECK( halo!=nullptr );
    CHECK( halo->volume(mycoord)==11 );
  }
  SECTION( "no-op and left and right" ) {
    CHECK_NOTHROW( dep.add_sigma_operator( no_op ) );
    CHECK_NOTHROW( dep.add_sigma_operator( left_shift ) );
    CHECK_NOTHROW( dep.add_sigma_operator( right_shift ) );
    CHECK_NOTHROW( combine.endow_beta_objects(combine_task->get_out_object()) );

    halo = combine.get_beta_object(0);
    CHECK( combine.get_out_object()->volume(mycoord)==10 );
    CHECK( halo!=nullptr );
    CHECK( halo->volume(mycoord)==12 );
  }
}

TEST_CASE( "Halo object for task bump","[task][halo][26]" ) {

  INFO( "mytid=" << mytid );

  // create distributions and objects for threepoint combination
  ioperator no_op("none"), right_shift(">=1"), left_shift("<=1");
  shared_ptr<distribution> d1 = 
    shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,-1,10*ntids) );
  shared_ptr<object> o1,r1;
  CHECK_NOTHROW( r1 = d1->new_object(d1) ); //new mpi_object(d1) ); // result vector
  CHECK_NOTHROW( o1 = d1->new_object(d1) ); // origin vector

  parallel_structure preds;
  kernel combine = mpi_kernel(o1,r1);
  REQUIRE_NOTHROW( combine.split_to_tasks() );
  shared_ptr<task> combine_task;
  REQUIRE_NOTHROW( combine_task = combine.get_tasks().at(0) );

  REQUIRE_NOTHROW( combine.set_last_dependency() );
  auto &dep = combine.set_last_dependency();

  SECTION( "no-op" ) {
    CHECK_NOTHROW( dep.add_sigma_operator( no_op ) );
    REQUIRE( !combine.last_dependency().has_type_uninitialized() );
    CHECK_NOTHROW( combine.endow_beta_objects(combine_task->get_out_object()) );

    CHECK( combine.get_out_object()->volume(mycoord)==10 );
    CHECK( combine.get_beta_object(0)!=nullptr );
    CHECK( combine.get_beta_object(0)->volume(mycoord)==10 );
  }
  SECTION( "no-op and left" ) {
    CHECK_NOTHROW( dep.add_sigma_operator( no_op ) );
    CHECK_NOTHROW( dep.add_sigma_operator( left_shift ) );
    CHECK_NOTHROW( combine.endow_beta_objects(combine_task->get_out_object()) );

    CHECK( combine.get_out_object()->volume(mycoord)==10 );
    CHECK( combine.get_beta_object(0)!=nullptr );
    if (arch.is_first_proc(mytid)) {
      CHECK( combine.get_beta_object(0)->volume(mycoord)==10 );
    } else {
      CHECK( combine.get_beta_object(0)->volume(mycoord)==11 );
    }
  }
  SECTION( "no-op and right" ) {
    CHECK_NOTHROW( dep.add_sigma_operator( no_op ) );
    CHECK_NOTHROW( dep.add_sigma_operator( right_shift ) );
    CHECK_NOTHROW( combine.endow_beta_objects(combine_task->get_out_object()) );

    CHECK( combine.get_out_object()->volume(mycoord)==10 );
    CHECK( combine.get_beta_object(0)!=nullptr );
    if (arch.is_last_proc(mytid)) {
      CHECK( combine.get_beta_object(0)->volume(mycoord)==10 );
    } else {
      CHECK( combine.get_beta_object(0)->volume(mycoord)==11 );
    }
  }
  SECTION( "no-op and left and right" ) {
    CHECK_NOTHROW( dep.add_sigma_operator( no_op ) );
    CHECK_NOTHROW( dep.add_sigma_operator( left_shift ) );
    CHECK_NOTHROW( dep.add_sigma_operator( right_shift ) );
    CHECK_NOTHROW( combine.endow_beta_objects(combine_task->get_out_object()) );

    CHECK( combine.get_out_object()->volume(mycoord)==10 );
    CHECK( combine.get_beta_object(0)!=nullptr );
    if (ntids==1) {
      CHECK( combine.get_beta_object(0)->volume(mycoord)==10 );
    } else if (arch.is_first_proc(mytid) || arch.is_last_proc(mytid)) {
      CHECK( combine.get_beta_object(0)->volume(mycoord)==11 );
    } else {
      CHECK( combine.get_beta_object(0)->volume(mycoord)==12 );
    }
  }
}

TEST_CASE( "Halo object for kernel modulo","[kernel][halo][modulo][27]" ) {

  // create distributions and objects for threepoint combination
  ioperator no_op("none"), right_shift(">>1"), left_shift("<<1");
  shared_ptr<distribution> d1 = 
    shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,-1,10*ntids) );
  shared_ptr<object> o1,r1;
  CHECK_NOTHROW( r1 = d1->new_object(d1) ); //new mpi_object(d1) ); // result vector
  CHECK_NOTHROW( o1 = d1->new_object(d1) ); // origin vector
  
  kernel combine;
  //mpi_task *tsk;
  shared_ptr<task> tsk;
  CHECK_NOTHROW( combine = mpi_kernel(o1,r1) );
  CHECK( combine.get_n_in_objects()==1 );

  SECTION( "no-op" ) {
    CHECK_NOTHROW( combine.add_sigma_operator( no_op ) );
    REQUIRE_NOTHROW( combine.set_last_dependency().endow_beta_object(o1) );
    CHECK_NOTHROW( combine.split_to_tasks() );
    CHECK( combine.get_tasks().size()==1 );
    CHECK_NOTHROW( tsk = combine.get_tasks().at(0) );
    CHECK( tsk->get_beta_object(0)!=nullptr );
    CHECK( tsk->get_beta_object(0)->volume(mycoord)==10 );
  }

  SECTION( "no-op and left" ) {
    CHECK_NOTHROW( combine.add_sigma_operator( no_op ) );
    CHECK_NOTHROW( combine.add_sigma_operator( left_shift ) );
    REQUIRE_NOTHROW( combine.set_last_dependency().endow_beta_object(o1) );
    CHECK_NOTHROW( combine.split_to_tasks() );
    CHECK( combine.get_tasks().size()==1 );
    CHECK_NOTHROW( tsk = combine.get_tasks().at(0) );
    CHECK( tsk->get_beta_object(0)!=nullptr );
    CHECK( tsk->get_beta_object(0)->volume(mycoord)==11 );
  }

  SECTION( "no-op and right" ) {
    CHECK_NOTHROW( combine.add_sigma_operator( no_op ) );
    CHECK_NOTHROW( combine.add_sigma_operator( right_shift ) );
    REQUIRE_NOTHROW( combine.set_last_dependency().endow_beta_object(o1) );
    CHECK_NOTHROW( combine.split_to_tasks() );
    CHECK( combine.get_tasks().size()==1 );
    CHECK_NOTHROW( tsk = combine.get_tasks().at(0) );
    CHECK( tsk->get_beta_object(0)!=nullptr );
    CHECK( tsk->get_beta_object(0)->volume(mycoord)==11 );
  }

  SECTION( "no-op and left and right" ) {
    CHECK_NOTHROW( combine.add_sigma_operator( no_op ) );
    CHECK_NOTHROW( combine.add_sigma_operator( left_shift ) );
    CHECK_NOTHROW( combine.add_sigma_operator( right_shift ) );
    REQUIRE_NOTHROW( combine.set_last_dependency().endow_beta_object(o1) );
    CHECK_NOTHROW( combine.split_to_tasks() );
    CHECK( combine.get_tasks().size()==1 );
    CHECK_NOTHROW( tsk = combine.get_tasks().at(0) );
    CHECK( tsk->get_beta_object(0)!=nullptr );
    CHECK( tsk->get_beta_object(0)->volume(mycoord)==12 );
  }
}

TEST_CASE( "Halo object for kernel bump","[kernel][halo][28]" ) {

  INFO( "mytid=" << mytid );

  // create distributions and objects for threepoint combination
  ioperator no_op("none"), right_shift(">=1"), left_shift("<=1");
  shared_ptr<distribution> d1 = 
    shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,-1,10*ntids) );
  shared_ptr<object> o1,r1;
  CHECK_NOTHROW( r1 = d1->new_object(d1) ); //new mpi_object(d1) ); // result vector
  CHECK_NOTHROW( o1 = d1->new_object(d1) ); // origin vector

  kernel combine;
  CHECK_NOTHROW( combine = mpi_kernel(o1,r1) );
  /* mpi_task* */ shared_ptr<task> tsk;

  SECTION( "no-op" ) {
    CHECK_NOTHROW( combine.add_sigma_operator( no_op ) );
    REQUIRE_NOTHROW( combine.set_last_dependency().endow_beta_object(o1) );
    CHECK_NOTHROW( combine.split_to_tasks() );
    CHECK( combine.get_tasks().size()==1 );
    CHECK_NOTHROW( tsk = combine.get_tasks().at(0) );
    CHECK( tsk->get_beta_object(0)!=nullptr );
    CHECK( tsk->get_beta_object(0)->volume(mycoord)==10 );
  }

  SECTION( "no-op and left" ) {
    CHECK_NOTHROW( combine.add_sigma_operator( no_op ) );
    CHECK_NOTHROW( combine.add_sigma_operator( left_shift ) );
    REQUIRE_NOTHROW( combine.set_last_dependency().endow_beta_object(o1) );
    CHECK_NOTHROW( combine.split_to_tasks() );
    CHECK( combine.get_tasks().size()==1 );
    CHECK_NOTHROW( tsk = combine.get_tasks().at(0) );
    REQUIRE( tsk->get_beta_object(0)!=nullptr );
    if (arch.is_first_proc(mytid)) {
      CHECK( tsk->get_beta_object(0)->volume(mycoord)==10 );
    } else {
      CHECK( tsk->get_beta_object(0)->volume(mycoord)==11 );
    }
  }

  SECTION( "no-op and right" ) {
    CHECK_NOTHROW( combine.add_sigma_operator( no_op ) );
    CHECK_NOTHROW( combine.add_sigma_operator( right_shift ) );
    REQUIRE_NOTHROW( combine.set_last_dependency().endow_beta_object(o1) );
    CHECK_NOTHROW( combine.split_to_tasks() );
    CHECK( combine.get_tasks().size()==1 );
    CHECK_NOTHROW( tsk = combine.get_tasks().at(0) );
    REQUIRE( tsk->get_beta_object(0)!=nullptr );
    if (arch.is_last_proc(mytid)) {
      CHECK( tsk->get_beta_object(0)->volume(mycoord)==10 );
    } else {
      CHECK( tsk->get_beta_object(0)->volume(mycoord)==11 );
    }
  }

  SECTION( "no-op and left and right" ) {
    CHECK_NOTHROW( combine.add_sigma_operator( no_op ) );
    CHECK_NOTHROW( combine.add_sigma_operator( left_shift ) );
    CHECK_NOTHROW( combine.add_sigma_operator( right_shift ) );
    REQUIRE_NOTHROW( combine.set_last_dependency().endow_beta_object(o1) );
    CHECK_NOTHROW( combine.split_to_tasks() );
    CHECK( combine.get_tasks().size()==1 );
    CHECK_NOTHROW( tsk = combine.get_tasks().at(0) );
    REQUIRE( tsk->get_beta_object(0)!=nullptr );
    if (ntids==1) {
      CHECK( tsk->get_beta_object(0)->volume(mycoord)==10 );
    } else if (arch.is_last_proc(mytid) || arch.is_first_proc(mytid)) {
      CHECK( tsk->get_beta_object(0)->volume(mycoord)==11 );
    } else {
      CHECK( tsk->get_beta_object(0)->volume(mycoord)==12 );
    }
  }
}

TEST_CASE( "Task execute on local data","[mpi][object][task][kernel][execute][29]" ) {

  int s = 10;
  auto block = 
    shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,s*ntids) );
  CHECK( block->volume(mycoord)==s );
  auto v = block->new_object(block);
  v->allocate();
  shared_ptr<kernel> k = shared_ptr<kernel>( new mpi_kernel(v) );
  REQUIRE_NOTHROW( k->analyze_dependencies() ); //split_to_tasks() );
  shared_ptr<task> t;
  REQUIRE_NOTHROW( t = k->get_tasks().at(0) );

  index_int vol;
  REQUIRE_NOTHROW( vol = t->get_out_object()->volume(mycoord) );
  CHECK( vol==s );
  t->set_localexecutefn(  &vector_gen );
  REQUIRE_NOTHROW( k->execute() );
  CHECK( k->get_all_msgs_completed() );
  //REQUIRE_NOTHROW( t->local_execute(v) );
  data_pointer data;
  REQUIRE_NOTHROW( data = v->get_data(mycoord) );
  for (int i=0; i<s; i++) {
    CHECK( data.at(i)==mytid+.5 );
  }
}

TEST_CASE( "Task execute on external data","[mpi][object][task][kernel][execute][external][30]" ) {
  
  // same test as before but now with externally allocated data
  int s = 10;
  auto xdata = data_allocate(s);
    //data_pointer( new vector<double>(s) );
  auto block = 
    shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,s*ntids) );
  auto xvector = block->new_object_from_data(xdata);
  shared_ptr<kernel> k = shared_ptr<kernel>( new mpi_kernel(xvector) );
  REQUIRE_NOTHROW( k->split_to_tasks());
  shared_ptr<task> xt;
  REQUIRE_NOTHROW( xt = k->get_tasks().at(0) );

  xt->set_localexecutefn(  &vector_gen );
  xt->local_execute(xvector); // no input because this is a generating kernel
  for (int i=0; i<s; i++) {
    CHECK( xdata->at(i)==mytid+.5 );
  }
}

TEST_CASE( "Execute local to local task","[task][halo][execute][31]" ) {

  INFO( "mytid=" << mytid );

  int nlocal=15, dim=1;
  ioperator no_op("none");
  auto block = 
    shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,-1,nlocal*ntids) );
  CHECK( block->get_type()==distribution_type::CONTIGUOUS );
  auto
    myfirst = block->first_index_r(mycoord),
    mylast = block->last_index_r(mycoord);
  auto 
    xvector = block->new_object(block),
    yvector = block->new_object(block);
  shared_ptr<distribution> xdist,ydist;
  REQUIRE_NOTHROW( xdist = xvector->get_distribution() );
  REQUIRE_NOTHROW( ydist = yvector->get_distribution() );
  {
    REQUIRE_NOTHROW( xvector->allocate() );
    data_pointer xdata;
    REQUIRE_NOTHROW( xdata = xvector->get_data(mycoord) );
    for (int i=0; i<nlocal; i++)
      xdata.at(i) = 1.5+i;
  }

  CHECK( xdist->get_type()==distribution_type::CONTIGUOUS );
  CHECK( ydist->get_type()==distribution_type::CONTIGUOUS );

  vector<mpi_task*> *tasks;
  data_pointer halo_data,ydata;
  shared_ptr<object> halo;
  shared_ptr<distribution> halo_dist;
  vector<shared_ptr<message>> msgs; shared_ptr<message> msg;

  SECTION( "analysis of the copy kernel" ) {
    shared_ptr<kernel> copy;
    shared_ptr<task> copy_task;
    const char *path;

    REQUIRE_NOTHROW( copy = shared_ptr<kernel>( new mpi_kernel(xvector,yvector) ) );
    CHECK
      ( copy->get_out_object()->get_distribution()->has_type(distribution_type::CONTIGUOUS) );
    copy->set_name("31copy");
    copy->add_sigma_operator( no_op );
    REQUIRE( !copy->last_dependency().has_type_uninitialized() );
    copy->set_localexecutefn( &veccopy );
    CHECK( !copy->has_type_origin() );

    REQUIRE_NOTHROW( copy->analyze_dependencies() );
    CHECK_NOTHROW( copy_task = copy->get_tasks().at(0) );

    // check that everything is in place
    CHECK( !copy_task->has_type_origin() );
    CHECK( copy_task->get_n_in_objects()>0 );
    REQUIRE_NOTHROW( halo = copy_task->get_beta_object(0) );
    REQUIRE( halo!=nullptr );
    REQUIRE_NOTHROW( halo_dist = halo->get_distribution() );
    REQUIRE( halo_dist!=nullptr );

    REQUIRE_NOTHROW( halo_dist->get_visibility(mycoord) ); // test for correct lambda
    CHECK( halo_dist->first_index_r(mycoord)==myfirst );
    CHECK( halo_dist->last_index_r(mycoord)==mylast );
    CHECK( halo_dist->volume(mycoord)==nlocal );
    CHECK( copy_task->get_receive_messages().size()==1 );
    vector<shared_ptr<message>> msgs;
    REQUIRE_NOTHROW( msgs = copy_task->get_send_messages() );
    CHECK( msgs.size()==1 );

    shared_ptr<multi_indexstruct> global_struct,local_struct;

    // check recv messages
    REQUIRE_NOTHROW( msgs = copy_task->get_receive_messages() );
    REQUIRE( msgs.size()==1 );
    REQUIRE_NOTHROW( msg = msgs.at(0) );
    CHECK_NOTHROW( global_struct = msg->get_global_struct() );
    CHECK_NOTHROW( local_struct = msg->get_local_struct() );
    CHECK( msg->get_sender()==mycoord );
    CHECK( msg->get_receiver()==mycoord );
    CHECK( global_struct->first_index_r()==myfirst ); // global
    CHECK( global_struct->last_index_r()==mylast );
    CHECK( local_struct->first_index_r()==domain_coordinate_allones(dim)*0 ); // local wrt halo
    CHECK( local_struct->last_index_r()==domain_coordinate_allones(dim)*nlocal-1 );
    
    // check send messages
    REQUIRE_NOTHROW( msgs = copy_task->get_send_messages() );
    REQUIRE( msgs.size()==1 );
    REQUIRE_NOTHROW( msg = msgs.at(0) );
    CHECK_NOTHROW( global_struct = msg->get_global_struct() );
    CHECK_NOTHROW( local_struct = msg->get_local_struct() );
    CHECK( msg->get_sender().coord(0)==mytid );
    CHECK( msg->get_receiver().coord(0)==mytid );
    CHECK( global_struct->first_index_r()==myfirst ); // global
    CHECK( global_struct->last_index_r()==mylast );
    CHECK( local_struct->first_index_r()==domain_coordinate_allones(dim)*0 ); // local wrt local
    CHECK( local_struct->last_index_r()==domain_coordinate_allones(dim)*nlocal-1 );
    
    // execute!
    CHECK_NOTHROW( copy_task->execute() );
    CHECK_NOTHROW( halo_data = halo->get_data(mycoord) );
    for (int i=0; i<nlocal; i++)
      CHECK( halo_data.at(i) == Approx(1.5+i) );
    CHECK_NOTHROW( ydata = yvector->get_data(mycoord) );
    for (int i=0; i<nlocal; i++)
      CHECK( ydata.at(i) == Approx(1.5+i) );
  }

  SECTION( "let's try the same with scaling" ) {
    shared_ptr<kernel> scale;

    REQUIRE_NOTHROW( scale = shared_ptr<kernel>( new mpi_kernel(xvector,yvector) ) );

    scale->set_name("31scale");
    scale->add_sigma_operator( no_op );
    scale->set_localexecutefn(  &vecscalebytwo );
    CHECK( !scale->has_type_origin() );
  
    REQUIRE_NOTHROW( scale->analyze_dependencies() );

    shared_ptr<task> scale_task;
    CHECK_NOTHROW( scale_task = scale->get_tasks().at(0) );

    CHECK_NOTHROW( scale_task->execute() );
    REQUIRE_NOTHROW( halo = scale_task->get_beta_object(0) );
    CHECK_NOTHROW( halo_data = halo->get_data(mycoord) );
    for (int i=0; i<nlocal; i++) {
      CHECK( halo_data.at(i) == Approx(1.5+i) );
    }
    CHECK_NOTHROW( ydata = yvector->get_data(mycoord) );
    for (int i=0; i<nlocal; i++) {
      CHECK( ydata.at(i) == Approx( 2*(1.5+i)) );
    }
  }
}

TEST_CASE( "Execute local to local task, k>1","[task][halo][execute][ortho][31k]" ) {

  INFO( "mytid=" << mytid );

  int nlocal=15,k=2,dim=1;
  ioperator no_op("none");
  shared_ptr<distribution> block;
  REQUIRE_NOTHROW( block =   shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,k,-1,nlocal*ntids) ) );
  auto
    myfirst = block->first_index_r(mycoord),
    mylast = block->last_index_r(mycoord);
  auto 
    xvector = block->new_object(block),
    yvector = block->new_object(block);
  shared_ptr<distribution> xdist,ydist;
  REQUIRE_NOTHROW( xdist = xvector->get_distribution() );
  REQUIRE_NOTHROW( ydist = yvector->get_distribution() );
  CHECK( xdist->get_orthogonal_dimension()==k );
  CHECK( ydist->get_orthogonal_dimension()==k );
  {
    REQUIRE_NOTHROW( xvector->allocate() );
    data_pointer xdata;
    REQUIRE_NOTHROW( xdata = xvector->get_data(mycoord) );
    for (int i=0; i<k*nlocal; i++)
      xdata.at(i) = 1.5+i;
  }
  auto
    copy = shared_ptr<kernel>( new mpi_kernel(xvector,yvector) ),
    scale = shared_ptr<kernel>( new mpi_kernel(xvector,yvector) );

  copy->set_name("31copy");
  copy->add_sigma_operator( no_op );
  copy->set_localexecutefn( &veccopy );
  CHECK( !copy->has_type_origin() );

  scale->set_name("31scale");
  scale->add_sigma_operator( no_op );
  scale->set_localexecutefn(  &vecscalebytwo );
  
  vector<mpi_task*> *tasks;
  data_pointer halo_data,ydata;
  shared_ptr<object> halo;
  shared_ptr<distribution> halo_dist;
  vector<shared_ptr<message>> msgs; shared_ptr<message> msg;

  auto localpt = domain_coordinate( vector<index_int>{nlocal} );

  SECTION( "analysis of the copy kernel" ) {
    shared_ptr<task> copy_task;
    const char *path;
    
    REQUIRE_NOTHROW( copy->analyze_dependencies() );
    CHECK_NOTHROW( copy_task = copy->get_tasks().at(0) );

    // check that everything is in place
    CHECK( !copy_task->has_type_origin() );
    CHECK( copy_task->get_n_in_objects()>0 );
    REQUIRE_NOTHROW( halo = copy_task->get_beta_object(0) );
    REQUIRE( halo!=nullptr );
    REQUIRE_NOTHROW( halo_dist = halo->get_distribution() );
    REQUIRE( halo_dist!=nullptr );
    INFO( fmt::format("Checking halo, on p={} : {}",
		      mytid,halo_dist->get_processor_structure(mycoord)->as_string()) );
    REQUIRE_NOTHROW( halo_dist->get_visibility(mycoord) ); // test for correct lambda
    CHECK( halo_dist->first_index_r(mycoord)==myfirst );
    CHECK( halo_dist->last_index_r(mycoord)==mylast );
    CHECK( halo_dist->volume(mycoord)==nlocal );
    CHECK( halo_dist->get_orthogonal_dimension()==k );
    CHECK( copy_task->get_receive_messages().size()==1 );
    vector<shared_ptr<message>> msgs;
    REQUIRE_NOTHROW( msgs = copy_task->get_send_messages() );
    CHECK( msgs.size()==1 );

    shared_ptr<multi_indexstruct> global_struct,local_struct;

    // check recv messages
    REQUIRE_NOTHROW( msgs = copy_task->get_receive_messages() );
    REQUIRE( msgs.size()==1 );
    REQUIRE_NOTHROW( msg = msgs.at(0) );
    CHECK_NOTHROW( global_struct = msg->get_global_struct() );
    CHECK_NOTHROW( local_struct = msg->get_local_struct() );
    CHECK( msg->get_sender()==mycoord );
    CHECK( msg->get_receiver()==mycoord );
    CHECK( global_struct->first_index_r()==myfirst ); // global
    CHECK( global_struct->last_index_r()==mylast );
    CHECK( local_struct->first_index_r()==domain_coordinate_zero(dim) ); // local wrt halo
    CHECK( local_struct->last_index_r()==localpt-1 );
    
    // check send messages
    REQUIRE_NOTHROW( msgs = copy_task->get_send_messages() );
    REQUIRE( msgs.size()==1 );
    REQUIRE_NOTHROW( msg = msgs.at(0) );
    CHECK_NOTHROW( global_struct = msg->get_global_struct() );
    CHECK_NOTHROW( local_struct = msg->get_local_struct() );
    CHECK( msg->get_sender()==mycoord );
    CHECK( msg->get_receiver()==mycoord );
    CHECK( global_struct->first_index_r()==myfirst ); // global
    CHECK( global_struct->last_index_r()==mylast );
    CHECK( local_struct->first_index_r()==domain_coordinate_zero(dim) ); // local wrt local
    CHECK( local_struct->last_index_r()==localpt-1 );
    
    // execute!
    CHECK_NOTHROW( copy_task->execute() );
    CHECK_NOTHROW( halo_data = halo->get_data(mycoord) );
    for (int i=0; i<k*nlocal; i++)
      CHECK( halo_data.at(i) == Approx(1.5+i) );
    CHECK_NOTHROW( ydata = yvector->get_data(mycoord) );
    for (int i=0; i<k*nlocal; i++)
      CHECK( ydata.at(i) == Approx(1.5+i) );
  }

  SECTION( "let's try the same with scaling" ) {

    REQUIRE_NOTHROW( scale->analyze_dependencies() );

    shared_ptr<task> scale_task;
    CHECK_NOTHROW( scale_task = scale->get_tasks().at(0) );

    CHECK_NOTHROW( scale_task->execute() );
    REQUIRE_NOTHROW( halo = scale_task->get_beta_object(0) );
    CHECK_NOTHROW( halo_data = halo->get_data(mycoord) );
    for (int i=0; i<nlocal; i++) {
      CHECK( halo_data.at(i) == Approx(1.5+i) );
    }
    CHECK_NOTHROW( ydata = yvector->get_data(mycoord) );
    for (int i=0; i<nlocal; i++) {
      CHECK( ydata.at(i) == Approx( 2*(1.5+i)) );
    }
  }
}

TEST_CASE( "Execute shift left task","[task][halo][execute][32]" ) {

  INFO( "mytid=" << mytid );

  int nlocal=15, dim=1;
  auto block = 
    shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,-1,nlocal*ntids) );
  auto
    myfirst = block->first_index_r(mycoord),
    mylast = block->last_index_r(mycoord);
  CHECK( myfirst==mycoord_coord*nlocal );
  CHECK( mylast==(mycoord_coord+1)*nlocal-1 );
  auto xdata = data_allocate(nlocal); //shared_ptr<double>( new double[nlocal] );
  auto 
    xvector = block->new_object_from_data(xdata),
    yvector = block->new_object(block);
  for (int i=0; i<nlocal; i++)
    xdata->at(i) = pointfunc33(i,myfirst[0]);
  auto shift = shared_ptr<kernel>( new mpi_kernel(xvector,yvector) );
  shift->set_name("32shift");
  shift->add_sigma_operator( ioperator("none") );
  shift->add_sigma_operator( ioperator(">=1") );
  shift->set_localexecutefn(  &vecshiftleftbump );
  
  shared_ptr<task> shift_task;
  data_pointer halo_data,ydata;

  CHECK_NOTHROW( shift->analyze_dependencies() );
  CHECK_NOTHROW( shift->split_to_tasks() );
  CHECK_NOTHROW( shift_task = shift->get_tasks().at(0) );

  // check that everything is in place
  REQUIRE( shift->get_n_in_objects()>0 );
  REQUIRE( shift_task->get_n_in_objects()>0 );
  REQUIRE( shift_task->get_beta_object(0)!=nullptr );
  if (mytid<ntids-1) {
    CHECK( shift_task->get_beta_object(0)->volume(mycoord)==nlocal+1 );
  } else {
    CHECK( shift_task->get_beta_object(0)->volume(mycoord)==nlocal );
  }
  // check recv messagegs
  shared_ptr<multi_indexstruct> global_struct,local_struct;
  if (mytid<ntids-1) {
    CHECK( shift_task->get_receive_messages().size()==2 );
  } else {
    CHECK( shift_task->get_receive_messages().size()==1 );
  }

  auto localpt = domain_coordinate( vector<index_int>{nlocal} );
  {
    vector<shared_ptr<message>> msgs;
    REQUIRE_NOTHROW( msgs = shift_task->get_receive_messages() );
    int nmsgs = msgs.size();
    if (mytid<ntids-1) 
      REQUIRE( nmsgs==2 );
    else
      REQUIRE( nmsgs==1 );
    for (int imsg=0; imsg<nmsgs; imsg++) {
      auto msg = msgs[imsg];
      CHECK_NOTHROW( global_struct = msg->get_global_struct() );
      CHECK_NOTHROW( local_struct = msg->get_local_struct() );
      if ( msg->get_sender().coord(0)==mytid) { // self message
	CHECK( global_struct->volume()==nlocal );
	CHECK( global_struct->first_index_r()==myfirst );
	CHECK( global_struct->last_index_r()==mylast );
	CHECK( local_struct->volume()==nlocal );
	CHECK( local_struct->first_index_r()==domain_coordinate_zero(dim) );
	CHECK( local_struct->last_index_r()==localpt-1 );
      } else {
	if (mytid<ntids-1) { // the other message comes from the right
	  CHECK( msg->get_sender().coord(0)==mytid+1 );
	  CHECK( global_struct->first_index_r()==mylast+1 );
	  CHECK( global_struct->last_index_r()==mylast+1 );
	  CHECK( local_struct->first_index_r()==localpt );
	  CHECK( local_struct->last_index_r()==localpt );
	}
      }
    }
  }

  {
    vector<shared_ptr<message>> msgs;
    REQUIRE_NOTHROW( msgs = shift_task->get_send_messages() );
    if (mytid>0) { // send to me and to the left
      CHECK( msgs.size()==2 );
    } else { // mytid==0: only self
      CHECK( msgs.size()==1 );
    }
    {
      for (int i=0; i<shift_task->get_send_messages().size(); i++) {
	auto msg = msgs[i];
	CHECK_NOTHROW( local_struct = msg->get_local_struct() );
	REQUIRE( local_struct!=nullptr );
	if (msg->get_receiver().coord(0)==mytid) {
	  CHECK( local_struct->first_index_r()==domain_coordinate_zero(dim) );
	  CHECK( local_struct->last_index_r()==localpt-1 );
	} else { // send to the left
	  REQUIRE( msg->get_receiver().coord(0)==mytid-1 );
	  CHECK( local_struct->first_index_r()==domain_coordinate_zero(dim) );
	  CHECK( local_struct->last_index_r()==domain_coordinate_zero(dim) );
	}
      }
    }
  }

  // check that everything is still in place
  REQUIRE( shift->get_n_in_objects()>0 );
  REQUIRE( shift->get_beta_objects().size()>0 );
  REQUIRE( shift_task->get_n_in_objects()>0 );
  REQUIRE( shift_task->get_beta_objects().size()>0 );

  // execute!
  CHECK_NOTHROW( shift_task->execute() );
  {
    int i;
    CHECK_NOTHROW( halo_data = shift_task->get_beta_object(0)->get_data(mycoord) );
    // the halo has the regular data because of the no-op
    for (i=0; i<nlocal; i++) {
      INFO( "i=" << i );
      CHECK( halo_data.at(i) == Approx(pointfunc33(i,myfirst[0])) );
    }
    if (mytid<ntids-1) { // all but the last proc have one extra halo loc
      i = nlocal;
      CHECK( halo_data.at(i) == Approx(pointfunc33(i,myfirst[0])) );
    }
    CHECK_NOTHROW( ydata = yvector->get_data(mycoord) );
    if (mytid==ntids-1) {
      for (i=0; i<nlocal-1; i++) {
	INFO( "i=" << i << " yi=" << ydata.at(i) );
	CHECK( ydata.at(i) == Approx(pointfunc33(i+1,myfirst[0])) );
      }
    } else {
      for (i=0; i<nlocal; i++) {
	INFO( "i=" << i << " yi=" << ydata.at(i) );
	CHECK( ydata.at(i) == Approx(pointfunc33(i+1,myfirst[0])) );
      }
    }
  }
}

TEST_CASE( "Scale kernel","[task][kernel][execute][33]" ) {

  INFO( "mytid=" << mytid );

  int nlocal=12;
  auto no_op = ioperator("none");
  auto block = 
    shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,-1,nlocal*ntids) );
  auto xdata = data_allocate(nlocal); //shared_ptr<double>( new double[nlocal] );
  auto
    xvector = block->new_object_from_data(xdata),
    yvector = block->new_object(block);
  auto
    my_first = block->first_index_r(mycoord),
    my_last = block->last_index_r(mycoord);
  CHECK( my_first==mycoord_coord*nlocal );
  CHECK( my_last==(mycoord_coord+1)*nlocal-1 );
  for (int i=0; i<nlocal; i++)
    xdata->at(i) = pointfunc33(i,my_first[0]);
  shared_ptr<kernel> scale;
  data_pointer halo_data,ydata;

  SECTION( "scale by constant" ) {
    scale = shared_ptr<kernel>( new mpi_kernel(xvector,yvector) );
    scale->set_name("33scale");
    scale->set_explicit_beta_distribution(block);

    SECTION( "constant in the function" ) {
      scale->set_localexecutefn(  &vecscalebytwo );
    }

    SECTION( "constant passed as context" ) {
      double *x = new double[1]; x[0] = 2.;
      REQUIRE_NOTHROW( scale->set_localexecutefn
		       ( [x] (kernel_function_args) -> void {
			 return vecscalebyc(kernel_function_call,*x); } ) );
    }
    
    /* mpi_task* */ shared_ptr<task> scale_task;

    //CHECK_NOTHROW( scale_task = (mpi_task*) scale->get_tasks().at(0) );
    CHECK_NOTHROW( scale->analyze_dependencies() );
    CHECK( scale->get_tasks().size()==1 );
    CHECK_NOTHROW( scale_task = scale->get_tasks().at(0) );
    CHECK_NOTHROW( scale->execute() );
    CHECK_NOTHROW( halo_data = scale_task->get_beta_object(0)->get_data(mycoord) );
    CHECK_NOTHROW( ydata = yvector->get_data(mycoord) );
    {
      int i;
      for (i=0; i<nlocal; i++) {
	CHECK( halo_data.at(i) == Approx( pointfunc33(i,my_first[0]) ) );
      }
    }
  }

  {
    int i;
    for (i=0; i<nlocal; i++) {
      CHECK( ydata.at(i) == Approx( 2*pointfunc33(i,my_first[0])) );
    }
  }
}

TEST_CASE( "Shift kernel modulo","[task][kernel][halo][modulo][execute][34]" ) {

  INFO( "mytid=" << mytid );

  int nlocal=10;
  shared_ptr<distribution> block = 
    shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,-1,nlocal*ntids) );
  auto xdata = data_allocate(nlocal);
  auto 
    xvector = block->new_object_from_data( xdata ),
    yvector = block->new_object(block);
  auto
    my_first = block->first_index_r(mycoord);
  for (int i=0; i<nlocal; i++)
    xdata->at(i) = pointfunc33(i,my_first[0]);
  auto shift = shared_ptr<kernel>( new mpi_kernel(xvector,yvector) );
  shift->set_name("34shift");
  shift->add_sigma_operator( ioperator("none") );
  shift->add_sigma_operator( ioperator(">>1") );
  shift->set_localexecutefn(  &vecshiftleftmodulo );
  CHECK( shift->has_type_compute() );
  
  vector<shared_ptr<task>> tasks;
  /* mpi_task* */ shared_ptr<task> shift_task;
  data_pointer halo_data,ydata;

  // SECTION( "by task" ) {
  //   CHECK_NOTHROW( shift->split_to_tasks() );
  //   REQUIRE_NOTHROW( shift->set_last_dependency().endow_beta_object(yvector) );
  //   REQUIRE_NOTHROW( tasks = shift->get_tasks() );
  //   CHECK_NOTHROW( shift_task = tasks.at(0) );
  //   CHECK_NOTHROW( shift_task->analyze_dependencies() );
  //   CHECK_NOTHROW( shift_task->execute() );
  // }
  SECTION( "by kernel" ) {
    REQUIRE_NOTHROW( shift->analyze_dependencies() );
    CHECK_NOTHROW( shift->execute() );
    REQUIRE_NOTHROW( tasks = shift->get_tasks() );
    CHECK_NOTHROW( shift_task = tasks.at(0) );
  }
  
  CHECK_NOTHROW( halo_data = shift_task->get_beta_object(0)->get_data(mycoord) );
  CHECK_NOTHROW( ydata = yvector->get_data(mycoord) );
  {
    int i=0;
    INFO( "i=" << i << " yi=" << ydata.at(i) );
    if (mytid==ntids-1) {
      for (i=0; i<nlocal-1; i++) {
	CHECK( ydata.at(i) == Approx(pointfunc33(i+1,my_first[0])) );
      }
      i = nlocal-1;
      CHECK( ydata.at(i) == Approx(pointfunc33(0,0)) );
    } else {
      for (i=0; i<nlocal; i++) {
	CHECK( ydata.at(i) == Approx(pointfunc33(i+1,my_first[0])) );
      }
    }
  }
}

TEST_CASE( "Shift from left kernel, message structure","[task][kernel][halo][35]" ) {

  INFO( "mytid=" << mytid );

  int nlocal=10, dim=1;
  shared_ptr<distribution> block = 
    shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,-1,nlocal*ntids) );
  auto xdata = data_allocate(nlocal);
  auto
    xvector = block->new_object_from_data(xdata),
    yvector = block->new_object(block);
  auto my_first = block->first_index_r(mycoord);
  for (int i=0; i<nlocal; i++)
    xdata->at(i) = pointfunc33(i,my_first[0]);
  auto shift = shared_ptr<kernel>( new mpi_kernel(xvector,yvector) );
  shift->set_name("35shift");
  shift->add_sigma_operator( ioperator("none") );
  shift->add_sigma_operator( ioperator("<=1") );
  
  /* mpi_task* */ shared_ptr<task> shift_task;
  data_pointer halo_data,ydata;

  REQUIRE_NOTHROW( shift->set_last_dependency().endow_beta_object(yvector) );
  CHECK_NOTHROW( shift->split_to_tasks() );
  CHECK_NOTHROW( shift_task = shift->get_tasks().at(0) );
  CHECK_NOTHROW( shift_task->analyze_dependencies() );

  /*
    See if the halo is properly transmitted
    Note: the halo is always larger than the original struct,
    so it is nlocal+1 (except on the 1st proc), and
    the message to self is nlocal
  */
  auto rmsgs = shift_task->get_receive_messages();
  if (mytid>0) { // everyone but the first receives from the left
    CHECK( rmsgs.size()==2 );
    for (int i=0; i<2; i++) {
      auto msg = rmsgs[i];
      auto // reproduce computations from mpi::compute_tar_index
	outer = msg->get_global_struct(); // global definition
      auto
	hstruct = msg->get_halo_struct();
      auto
	inner = msg->get_embed_struct(), // description of the halo region
	rstruct = msg->get_local_struct(); // local wrt the halo
      if (msg->get_sender().coord(0)==mytid-1) { // msg to the right
	// target location
	CHECK( rstruct->first_index_r()==domain_coordinate_zero(dim) ); // logical check
	CHECK( rstruct->last_index_r()==domain_coordinate_zero(dim) );
	CHECK( rstruct->first_index_r()==hstruct->location_of(inner) ); // expression from mpi::compute_tar_index
	//CHECK( rstruct->first_index_r()==outer->location_of(inner) ); // expression from mpi::compute_tar_index
      } else { // message to self is your first nlocal-1 elts
	CHECK( msg->get_sender()==mycoord);
	if (mytid>0) {
	  CHECK( rstruct->first_index_r()==domain_coordinate_allones(dim) );
	  CHECK( rstruct->volume()==nlocal );
	} else {
	  CHECK( rstruct->first_index_r()==domain_coordinate_allones(dim) );
	  CHECK( rstruct->volume()==nlocal-1 );
	}
	CHECK( rstruct->first_index_r()==hstruct->location_of(inner) ); // expression from mpi::compute_tar_index
	//CHECK( rstruct->first_index_r()==outer->location_of(inner) ); // expression from mpi::compute_tar_index
      }
    }
  } else {
    CHECK( rmsgs.size()==1 );
  }
  auto smsgs = shift_task->get_send_messages();
  if (mytid<ntids-1) {
    int i;
    CHECK( smsgs.size()==2 ); // everyone but the last sends to the right
    for (i=0; i<2; i++) {
      auto msg = smsgs[i];
      auto sstruct = msg->get_local_struct();
      if (msg->get_receiver().coord(0)==mytid+1) { // msg to the right
	CHECK( sstruct->first_index_r()==domain_coordinate_allones(dim)*nlocal-1 );
	CHECK( sstruct->last_index_r()==domain_coordinate_allones(dim)*nlocal-1 );
      } else { // message to self
	if (mytid<ntids-1) {
	  CHECK( sstruct->first_index_r()==domain_coordinate_zero(dim) );
	  CHECK( sstruct->volume()==nlocal );
	} else {
	  CHECK( sstruct->first_index_r()==domain_coordinate_zero(dim) );
	  CHECK( sstruct->volume()==nlocal-1 );
	}
      }
    }
  } else {
    CHECK( smsgs.size()==1 );
  }
}

TEST_CASE( "Shift right kernel, execute","[task][kernel][halo][execute][36]" ) {

  INFO( "mytid=" << mytid );

  int nlocal=10, dim=1;
  shared_ptr<distribution> block =   shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,-1,nlocal*ntids) );
  auto xdata = data_allocate(nlocal);
  auto 
    xvector = block->new_object_from_data(xdata),
    yvector = block->new_object(block);
  auto my_first = block->first_index_r(mycoord);
  for (int i=0; i<nlocal; i++)
    xdata->at(i) = pointfunc33(i,my_first[0]);
  auto shift = shared_ptr<kernel>( new mpi_kernel(xvector,yvector) );
  shift->set_name("36shift");
  shift->add_sigma_operator( ioperator("none") );
  shift->add_sigma_operator( ioperator("<=1") );
  shift->set_localexecutefn(  &vecshiftrightbump );
  
  /* mpi_task* */ shared_ptr<task> shift_task;
  data_pointer halo_data,ydata;

  CHECK_NOTHROW( shift->analyze_dependencies() );
  CHECK_NOTHROW( shift_task = shift->get_tasks().at(0) );
  CHECK_NOTHROW( shift->set_last_dependency().set_name("36mpi-dependency") );

  auto msgs = shift_task->get_receive_messages();
  if (mytid==0)
    CHECK( msgs.size()==1 );
  else
    CHECK( msgs.size()==2 );

  shared_ptr<object> halo;
  CHECK_NOTHROW( halo = shift->get_beta_object(0) );
  {
    INFO( "investigating halo <<" << halo->get_name() << ">>" );
    if (mytid==0)
      CHECK( halo->first_index_r(mycoord)==domain_coordinate_zero(dim) );
    else
      CHECK( halo->first_index_r(mycoord)==yvector->first_index_r(mycoord)-1);
  }

  CHECK_NOTHROW( shift->execute() );

  CHECK_NOTHROW( halo_data = halo->get_data(mycoord) );
  {
    int i;
    if (mytid==0) { // there is no left halo, so h[i]==f[i]
      for (i=0; i<nlocal; i++) {
	INFO( "i=" << i << " h[i]=" << halo_data.at(i) );
	CHECK( halo_data.at(i) == Approx(pointfunc33(i,my_first[0])) );
      }
    } else { // there is a left halo, so h[i]==f(i-1)
      for (i=0; i<nlocal; i++) {
	INFO( "i=" << i << " h[i]=" << halo_data.at(i) );
	CHECK( halo_data.at(i) == Approx(pointfunc33(i-1,my_first[0])) );
      }
    }
  }

  CHECK_NOTHROW( ydata = yvector->get_data(mycoord) );
  {
    int i;
    if (mytid==0) {
      i = 0;
      //CHECK( ydata.at(i] == Approx(pointfunc33(i,my_first[0])) );
      for (i=1; i<nlocal; i++) {
	INFO( "i=" << i << " y[i]=" << ydata.at(i) );
	CHECK( ydata.at(i) == Approx(pointfunc33(i-1,my_first[0])) );
      }
    } else {
      for (i=0; i<nlocal; i++) {
	INFO( "i=" << i << " y[i]=" << ydata.at(i) );
	CHECK( ydata.at(i) == Approx(pointfunc33(i-1,my_first[0])) );
      }
    }
  }
}

TEST_CASE( "Shift from left kernel modulo, execute","[task][kernel][halo][execute][modulo][37]" ) {

  INFO( "mytid=" << mytid );

  int nlocal=10, dim=1;
  shared_ptr<distribution> block = 
    shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,-1,nlocal*ntids) );
  int nglobal = block->outer_size();
  auto xdata = data_allocate(nlocal);
  auto 
    xvector = block->new_object_from_data(xdata),
    yvector = block->new_object(block);
  auto my_first = block->first_index_r(mycoord);
  for (int i=0; i<nlocal; i++)
    xdata->at(i) = pointfunc33(i,my_first[0]);
  auto shift = shared_ptr<kernel>( new mpi_kernel(xvector,yvector) );
  shift->set_name("37shift");
  shift->add_sigma_operator( ioperator("none") );
  shift->add_sigma_operator( ioperator("<<1") );
  shift->set_localexecutefn(  &vecshiftrightmodulo );
  
  /* mpi_task* */ shared_ptr<task> shift_task;
  data_pointer halo_data,ydata;

  REQUIRE_NOTHROW( shift->set_last_dependency().endow_beta_object(yvector) );
  CHECK_NOTHROW( shift->split_to_tasks() );
  CHECK_NOTHROW( shift_task = shift->get_tasks().at(0) );
  CHECK_NOTHROW( shift_task->analyze_dependencies() );

  CHECK_NOTHROW( shift->execute() );
  CHECK_NOTHROW( halo_data = shift_task->get_beta_object(0)->get_data(mycoord) );
  CHECK_NOTHROW( ydata = yvector->get_data(mycoord) );
  {
    int i;
    if (mytid==0) { // hi=f(i-1) except for i=0: hi=f(n-1)
      for (i=1; i<nlocal; i++) {
	INFO( "i=" << i << " hi=" << halo_data.at(i) );
	CHECK( halo_data.at(i) == Approx(pointfunc33(i-1,my_first[0])) );
      }
      i = 0;
      CHECK( halo_data.at(i) == Approx(pointfunc33(nglobal-1,my_first[0])) );
    } else { // there is a left halo, so hi==f(i-1)
      for (i=0; i<nlocal; i++) {
	INFO( "i=" << i << " hi=" << halo_data.at(i) );
	CHECK( halo_data.at(i) == Approx(pointfunc33(i-1,my_first[0])) );
      }
    }
  }
  {
    int i;
    if (mytid==0) {
      for (i=1; i<nlocal; i++) {
	INFO( "i=" << i << " yi=" << ydata.at(i) );
	CHECK( ydata.at(i) == Approx(pointfunc33(i-1,my_first[0])) );
      }
      i = 0;
      CHECK( ydata.at(i) == Approx(pointfunc33(nglobal-1,my_first[0])) );
    } else {
      for (i=0; i<nlocal; i++) {
	INFO( "i=" << i << " yi=" << ydata.at(i) );
	CHECK( ydata.at(i) == Approx(pointfunc33(i-1,my_first[0])) );
      }
    }
  }
}

TEST_CASE( "Shift from left kernel bump, execute","[task][kernel][halo][execute][modulo][38]" ) {

  INFO( "mytid=" << mytid << " out of " << ntids );

  int nlocal=10, dim=1;
  shared_ptr<distribution> block = 
    shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,-1,nlocal*ntids) );
  int nglobal = block->outer_size();
  auto xdata = data_allocate(nlocal);
  auto 
    xvector = block->new_object_from_data(xdata),
    yvector = block->new_object(block);
  auto my_first = block->first_index_r(mycoord);
  for (int i=0; i<nlocal; i++)
    xdata->at(i) = pointfunc33(i,my_first[0]);
  auto shift = shared_ptr<kernel>( new mpi_kernel(xvector,yvector) );
  shift->set_name("38shift");
  shift->add_sigma_operator( ioperator("none") );
  shift->add_sigma_operator( ioperator("<=1") );
  shift->set_localexecutefn(  &vecshiftrightbump );

  /* mpi_task* */ shared_ptr<task> shift_task;
  data_pointer halo_data,ydata;

  REQUIRE_NOTHROW( shift->set_last_dependency().endow_beta_object(yvector) );
  CHECK_NOTHROW( shift->split_to_tasks() );
  CHECK_NOTHROW( shift_task = shift->get_tasks().at(0) );
  CHECK_NOTHROW( shift_task->analyze_dependencies() );

  REQUIRE_NOTHROW( shift->execute() );
  CHECK_NOTHROW( halo_data = shift_task->get_beta_object(0)->get_data(mycoord) );
  CHECK_NOTHROW( ydata = yvector->get_data(mycoord) );
  {
    int i;
    if (mytid==0) { // there is no halo to the left
      for (i=0; i<nlocal; i++) {
	INFO( "i=" << i << " hi=" << halo_data.at(i) );
	CHECK( halo_data.at(i) == Approx(pointfunc33(i,my_first[0])) );
      }
    } else { // there is a left halo, so hi==f(i-1)
      for (i=0; i<nlocal; i++) {
	INFO( "i=" << i << " hi=" << halo_data.at(i) );
	CHECK( halo_data.at(i) == Approx(pointfunc33(i-1,my_first[0])) );
      }
    }
  }
  {
    int i;
    if (mytid==0) {
      for (i=1; i<nlocal; i++) {
	INFO( "i=" << i << " yi=" << ydata.at(i) );
	CHECK( ydata.at(i) == Approx(pointfunc33(i-1,my_first[0])) );
      }
      i = 0;
      // location zero is undefined CHECK( ydata.at(0]==1 );
    } else {
      for (i=0; i<nlocal; i++) {
	INFO( "i=" << i << " yi=" << ydata.at(i) );
	//	CHECK( ydata.at(i) == Approx(pointfunc33(i-1,my_first[0])) );
      }
    }
  }
}

TEST_CASE( "Three point sum modulo, execute","[task][kernel][halo][execute][modulo][39]" ) {

  INFO( "mytid=" << mytid );

  int nlocal=10, dim=1;
  shared_ptr<distribution> block = 
    shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,-1,nlocal*ntids) );
  int nglobal = block->outer_size();
  auto xdata = data_allocate(nlocal);
  auto
    xvector = block->new_object_from_data(xdata),
    yvector = block->new_object(block);
  auto
    my_first = block->first_index_r(mycoord),
    my_last = block->last_index_r(mycoord);
  for (int i=0; i<nlocal; i++)
    xdata->at(i) = pointfunc33(i,my_first[0]);
  auto sum = shared_ptr<kernel>( new mpi_kernel(xvector,yvector) );
  sum->set_name("39sum");
  sum->add_sigma_operator( ioperator("none") );
  sum->add_sigma_operator( ioperator("<<1") );
  sum->add_sigma_operator( ioperator(">>1") );
  sum->set_localexecutefn(  &threepointsummod );

  /* mpi_task* */ shared_ptr<task> sum_task;
  data_pointer halo_data,ydata;

  REQUIRE_NOTHROW( sum->set_last_dependency().endow_beta_object(yvector) );
  CHECK_NOTHROW( sum->split_to_tasks() );
  CHECK_NOTHROW( sum_task = sum->get_tasks().at(0) );
  CHECK_NOTHROW( sum_task->analyze_dependencies() );

  CHECK_NOTHROW( sum->execute() );
  CHECK_NOTHROW( halo_data = sum_task->get_beta_object(0)->get_data(mycoord) );
  CHECK_NOTHROW( ydata = yvector->get_data(mycoord) );

  processor_coordinate *last_proc;
  REQUIRE_NOTHROW( last_proc = new processor_coordinate( vector<int>{ntids-1} ) );
  {
    int i=0, lastv = pointfunc33(nlocal-1,block->first_index_r(last_proc).coord(0));
    INFO( "i=" << i << " hi=" << halo_data.at(i) );
    if (mytid==0) {
      for (i=1; i<nlocal+2; i++) {
	CHECK( halo_data.at(i) == Approx(pointfunc33(i-1,my_first[0])) );
      }
      i = 0;
      CHECK( halo_data.at(i) == Approx(lastv) );
    } else if (mytid==ntids-1) {
      for (i=0; i<nlocal+1; i++) {
	CHECK( halo_data.at(i) == Approx(pointfunc33(i-1,my_first[0])) );
      }
      i = nlocal+1;
      CHECK( halo_data.at(i) == Approx(pointfunc33(0,0)) );
    } else {
      for (i=0; i<nlocal+2; i++) {
	CHECK( halo_data.at(i) == Approx(pointfunc33(i-1,my_first[0])) );
      }
    }
  }
  {
    int i=0, lastv = pointfunc33(nlocal-1,block->first_index_r(last_proc).coord(0));
    INFO( "i=" << i << " yi=" << ydata.at(i) );
    if (mytid==0) {
      for (i=1; i<nlocal; i++) {
	CHECK( ydata.at(i) == Approx(3*pointfunc33(i,my_first[0])) );
      }
      i = 0; // last+0+1
      CHECK( ydata.at(i) == Approx(pointfunc33(i,my_first[0])+pointfunc33(i+1,my_first[0])+lastv) );
    } else if (mytid==ntids-1) {
      for (i=0; i<nlocal-1; i++) {
	CHECK( ydata.at(i) == Approx(3*pointfunc33(i,my_first[0])) );
      }
      i = nlocal-1; // last-1+last+0
      CHECK( ydata.at(i) == Approx(pointfunc33(i,my_first[0])+pointfunc33(i-1,my_first[0])+pointfunc33(0,0)) );
    } else {
      for (i=0; i<nlocal-1; i++) {
	CHECK( ydata.at(i) == Approx(3*pointfunc33(i,my_first[0])) );
      }
    }
  }
}

TEST_CASE( "Three point sum bump, execute","[task][kernel][halo][execute][39]" ) {

  INFO( "mytid=" << mytid );

  int nlocal=10, dim=1;
  shared_ptr<distribution> block = 
    shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,-1,nlocal*ntids) );
  int nglobal = block->outer_size();
  auto xdata = data_allocate(nlocal);
  auto
    xvector = block->new_object_from_data(xdata),
    yvector = block->new_object(block);
  auto
    my_first = block->first_index_r(mycoord),
    my_last = block->last_index_r(mycoord);
  for (int i=0; i<nlocal; i++)
    xdata->at(i) = pointfunc33(i,my_first[0]);
  auto sum = shared_ptr<kernel>( new mpi_kernel(xvector,yvector) );
  sum->set_name("39sum");
  sum->add_sigma_operator( ioperator("none") );
  sum->add_sigma_operator( ioperator("<=1") );
  sum->add_sigma_operator( ioperator(">=1") );
  sum->set_localexecutefn(  &threepointsumbump );

  shared_ptr<task> sum_task;
  shared_ptr<object> halo;
  data_pointer halo_data,ydata;

  CHECK_NOTHROW( sum->analyze_dependencies() );
  CHECK_NOTHROW( sum_task = sum->get_tasks().at(0) );

  auto msgs = sum_task->get_receive_messages();
  if (ntids==1) {
    CHECK( msgs.size()==1 );
  } else if (mytid==0 || mytid==ntids-1)
    CHECK( msgs.size()==2 );
  else
    CHECK( msgs.size()==3 );
  for ( auto m : msgs ) { // (auto m=msgs.begin(); m!=msgs.end(); ++m) {
    int snd = m->get_sender().coord(0); INFO( "msg from " << snd );
    if (snd!=mytid) {
      if (mytid==0)
	CHECK( snd==1 );
      else if (mytid==ntids-1)
	CHECK( snd==ntids-2);
      else 
	CHECK( (snd==mytid-1 || snd==mytid+1) );
    }
  }

  CHECK_NOTHROW( sum->execute() );
  CHECK_NOTHROW( halo = sum_task->get_beta_object(0) );
  CHECK_NOTHROW( halo_data = halo->get_data(mycoord) );
  CHECK_NOTHROW( ydata = yvector->get_data(mycoord) );
  {
    int i=0;
    INFO( "i=" << i << " hi=" << halo_data.at(i) );
    if (ntids==1) {
      CHECK( halo->first_index_r(mycoord)==domain_coordinate_zero(dim) );
      CHECK( halo->volume(mycoord)==nlocal );
      for (i=0; i<nlocal; i++) {
	CHECK( halo_data.at(i) == Approx(pointfunc33(i,my_first[0])) );
      }
    } else if (mytid==0) {
      CHECK( halo->first_index_r(mycoord)==domain_coordinate_zero(dim) );
      CHECK( halo->volume(mycoord)==(nlocal+1) );
      for (i=0; i<nlocal+1; i++) {
	CHECK( halo_data.at(i) == Approx(pointfunc33(i,my_first[0])) );
      }
    } else if (mytid==ntids-1) {
      CHECK( halo->first_index_r(mycoord)==(my_first-1) );
      CHECK( halo->volume(mycoord)==(nlocal+1) );
      for (i=0; i<nlocal-1; i++) {
	CHECK( halo_data.at(i) == Approx(pointfunc33(i-1,my_first[0])) );
      }
    } else {
      CHECK( halo->first_index_r(mycoord)==(my_first-1) );
      CHECK( halo->volume(mycoord)==(nlocal+2) );
      for (i=0; i<nlocal+2; i++) {
	CHECK( halo_data.at(i) == Approx(pointfunc33(i-1,my_first[0])) );
      }
    }
  }
  {
    int i=0;
    if (ntids==1) {
      i = 0; // last+0+1
      CHECK( ydata.at(i)==Approx(pointfunc33(i,my_first[0])+pointfunc33(i+1,my_first[0])) );
      for (i=1; i<nlocal-1; i++) {
	INFO( "i=" << i << " yi=" << ydata.at(i) );
	CHECK( ydata.at(i)==Approx(pointfunc33(i-1,my_first[0])+pointfunc33(i,my_first[0])+pointfunc33(i+1,my_first[0])) );
      }
      i = nlocal-1; // last-1+last+0
      CHECK( ydata.at(i) == Approx(pointfunc33(i-1,my_first[0])+pointfunc33(i,my_first[0])) );
    } else if (mytid==0) {
      i = 0; // last+0+1
      CHECK( ydata.at(i)==Approx(pointfunc33(i,my_first[0])+pointfunc33(i+1,my_first[0])) );
      for (i=1; i<nlocal; i++) {
	INFO( "i=" << i << " yi=" << ydata.at(i) );
	CHECK( ydata.at(i)==Approx(pointfunc33(i-1,my_first[0])+pointfunc33(i,my_first[0])+pointfunc33(i+1,my_first[0])) );
      }
    } else if (mytid==ntids-1) {
      for (i=0; i<nlocal-1; i++) {
	INFO( "i=" << i << " yi=" << ydata.at(i) );
	CHECK( ydata.at(i)==Approx(pointfunc33(i-1,my_first[0])+pointfunc33(i,my_first[0])+pointfunc33(i+1,my_first[0])) );
	//	CHECK( ydata.at(i) == Approx(3*pointfunc33(i,my_first[0])) );
      }
      i = nlocal-1; // last-1+last+0
      CHECK( ydata.at(i) == Approx(pointfunc33(i-1,my_first[0])+pointfunc33(i,my_first[0])) );
    } else {
      for (i=0; i<nlocal-1; i++) {
	INFO( "i=" << i << " yi=" << ydata.at(i) );
	CHECK( ydata.at(i)==Approx(pointfunc33(i-1,my_first[0])+pointfunc33(i,my_first[0])+pointfunc33(i+1,my_first[0])) );
      }
    }
  }
}

TEST_CASE( "Test explicit beta","[beta][distribution][modulo][40]" ) {

  INFO( "mytid=" << mytid );

  int localsize=15,gsize = localsize*ntids;
  auto block
    = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,localsize,-1) );
  auto
    my_first = block->first_index_r(mycoord),
    my_last = block->last_index_r(mycoord);
  shared_ptr<distribution> left,right,wide;
  int
    iscont = block->has_type(distribution_type::CONTIGUOUS),
    isblok = block->has_type(distribution_type::BLOCKED);
  CHECK( ( iscont || isblok ) );
	 
  // test left shift
  REQUIRE_NOTHROW( left = block->operate( ioperator("<<1") ) );
  CHECK( left->get_is_orthogonal() );
  CHECK( left->has_type_locally_contiguous() );
  CHECK( !left->has_type(distribution_type::UNDEFINED) );
  CHECK( block->first_index_r(mycoord)==my_first );
  CHECK( left->first_index_r(mycoord)==my_first-1 );
  // test right shift
  REQUIRE_NOTHROW( right = block->operate( ioperator(">>1") ) );
  CHECK( block->first_index_r(mycoord)==my_first );
  CHECK( right->first_index_r(mycoord)==my_first+1 );
  CHECK( !right->has_type(distribution_type::UNDEFINED) );
  // union
  REQUIRE_NOTHROW( wide = left->distr_union(right) );
  CHECK( wide->first_index_r(mycoord)==my_first-1 ); // distributions can stick out
  CHECK( wide->last_index_r(mycoord)==my_last+1 );

  auto
    in = block->new_object(block),
    out = block->new_object(block);
  CHECK( ( !arch.get_can_embed_in_beta() || !in->has_data_status_allocated() ) );
  REQUIRE_NOTHROW( in->allocate() );
  REQUIRE( in->has_data_status_allocated() );

  shared_ptr<kernel> threepoint;
  REQUIRE_NOTHROW( threepoint = shared_ptr<kernel>( new mpi_kernel(in,out) ) );
  threepoint->set_localexecutefn(  &threepointsummod );
  threepoint->set_name("40threepoint");
  REQUIRE_NOTHROW( threepoint->set_explicit_beta_distribution(wide) );
  REQUIRE_NOTHROW( threepoint->analyze_dependencies() );

  vector<shared_ptr<task>> tasks;
  REQUIRE_NOTHROW( tasks = threepoint->get_tasks() );
  CHECK( tasks.size()==1 );
  {
    shared_ptr<task> threetask;
    REQUIRE_NOTHROW( threetask = tasks.at(0) );
    vector<shared_ptr<message>> rmsgs;
    REQUIRE_NOTHROW( rmsgs = threetask->get_receive_messages() );
    CHECK( rmsgs.size()==3 );
  }

  data_pointer indata; REQUIRE_NOTHROW( indata = in->get_data(mycoord) );
  CHECK( in->volume(mycoord)==localsize );
  for (index_int i=0; i<localsize; i++)
    indata.at(i) = 2.;
  REQUIRE_NOTHROW( threepoint->execute() );
  data_pointer outdata; REQUIRE_NOTHROW( outdata = out->get_data(mycoord) );
  CHECK( out->volume(mycoord)==localsize );
  for (index_int i=0; i<localsize; i++) {
    INFO( "i=" << i << " data[i]=" << outdata.at(i) );
    CHECK( outdata.at(i) == Approx(6.) );
  }
}

TEST_CASE( "Operations between different distributions","[kernel][distribution][41]" ) {

  int nlocal = 21, nglobal = ntids*nlocal;
  vector<index_int> locals(ntids);
  for (int tid=0; tid<ntids-1; tid++) locals.at(tid) = nlocal-1;
  locals.at(ntids-1) = nglobal - (ntids-1)*(nlocal-1);
  shared_ptr<distribution> reg,irr;
  CHECK_NOTHROW( reg =   shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,nlocal,-1) ) );
  CHECK_NOTHROW( irr =   shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,locals) ) );
  CHECK( reg->volume(mycoord)==nlocal );
  if (mytid<ntids-1)
    CHECK( irr->volume(mycoord)==(nlocal-1) );
  else
    CHECK( irr->volume(mycoord)==(nlocal+ntids-1) );
  shared_ptr<object> in1,in2,out;
  REQUIRE_NOTHROW( in1 = reg->new_object(reg) );
  REQUIRE_NOTHROW( in2 = irr->new_object(irr) );
  REQUIRE_NOTHROW( in1->allocate() );
  REQUIRE_NOTHROW( in2->allocate() );
  REQUIRE_NOTHROW( out = reg->new_object(reg) );
  {
    data_pointer data; index_int localsize;
    REQUIRE_NOTHROW( data = in1->get_data(mycoord) );
    localsize = in1->volume(mycoord);
    for (index_int i=0; i<localsize; i++)
      REQUIRE_NOTHROW( data.at(i) = 1. );
    REQUIRE_NOTHROW( data = in2->get_data(mycoord) );
    localsize = in2->volume(mycoord);
    for (index_int i=0; i<localsize; i++)
      REQUIRE_NOTHROW( data.at(i) = 2. );
  }
  shared_ptr<kernel> k;
  REQUIRE_NOTHROW( k = shared_ptr<kernel>( new mpi_sum_kernel(in1,in2,out) ) );
  REQUIRE_NOTHROW( k->analyze_dependencies() );
  REQUIRE_NOTHROW( k->execute() );
  {
    data_pointer data; index_int localsize;
    REQUIRE_NOTHROW( data = out->get_data(mycoord) );
    localsize = out->volume(mycoord);
    for (index_int i=0; i<localsize; i++)
      CHECK( data.at(i)==Approx(3.) );
  }
}

TEST_CASE( "communication avoiding distributions, explicit","[kernel][extend][42]" ) {
  // VLE the whole extend function is doubtful. maybe give up on it.
  int dim = 1;
  INFO( "mytid=" << mytid );
  int nlocal=100,nglobal=nlocal*ntids;
  shared_ptr<distribution> block,extend;
  REQUIRE_NOTHROW( block =   shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,nlocal,-1) ) );
  REQUIRE_NOTHROW( extend =   shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,nlocal,-1) ) );
  auto 
    my_first = block->first_index_r(mycoord),
    my_last = block->last_index_r(mycoord);
  INFO( fmt::format("Comparing against first={}, last={}",
		    my_first.as_string(),my_last.as_string()) );
  INFO( fmt::format("start with mystruct={}",
		    extend->get_processor_structure(mycoord)->as_string()) );

  shared_ptr<multi_indexstruct> left;
  if (mytid!=0) {
    REQUIRE_NOTHROW
      ( left = shared_ptr<multi_indexstruct>
	  { new contiguous_multi_indexstruct(my_first-1) } );
  } else {
    left = shared_ptr<multi_indexstruct>
      { new empty_multi_indexstruct(dim) };
  }
  INFO( fmt::format("Extending on the left with {}",left->as_string()) );
  REQUIRE_NOTHROW( extend = extend->extend(mycoord,left) );

  INFO( fmt::format("Comparing first {} of extended against my first {}",
		    extend->first_index_r(mycoord).as_string(),
		    my_first.as_string()) );
  if (mytid!=0) {
    INFO( "should be off by one for me!=0" );
    CHECK( extend->first_index_r(mycoord)==(my_first-1) );
  } else {
    INFO( "should be equal for me==0" );
    CHECK( extend->first_index_r(mycoord)==my_first );
  }

  shared_ptr<multi_indexstruct> right;
  if (mytid!=ntids-1) {
    REQUIRE_NOTHROW
      ( right = shared_ptr<multi_indexstruct>
	  { new contiguous_multi_indexstruct( my_last+1 ) } );
  } else {
    REQUIRE_NOTHROW
      ( right = shared_ptr<multi_indexstruct>
	  { new empty_multi_indexstruct(dim) } );
  }
  REQUIRE_NOTHROW( extend = extend->extend(mycoord,right) );
  INFO( fmt::format("after extending right, mystruct={}",
		    extend->get_processor_structure(mycoord)->as_string()) );
  if (mytid!=ntids-1) {
    INFO( "should be off by one for me!=ntids-1" );
    CHECK( extend->last_index_r(mycoord)==my_last+1 );
  } else {
    INFO( "should be equal for me==ntids-1" );
    CHECK( extend->last_index_r(mycoord)==my_last );
  }

  shared_ptr<object> v0,v1,v2;
  REQUIRE_NOTHROW( v0 = block->new_object(block) );
  REQUIRE_NOTHROW( v1 = extend->new_object(extend) );
  REQUIRE_NOTHROW( v2 = block->new_object(block) );

  shared_ptr<kernel> k1,k2;
  REQUIRE_NOTHROW( k1 = shared_ptr<kernel>( new mpi_kernel( v0,v1 ) ) );
  k1->set_name("k1");
  k1->add_sigma_operator( ioperator("none") );
  k1->add_sigma_operator( ioperator("<=1") );
  k1->add_sigma_operator( ioperator(">=1") );
  k1->set_localexecutefn(  &threepointsumbump );
  REQUIRE_NOTHROW( k2 = shared_ptr<kernel>( new mpi_kernel( v1,v2 ) ) );
  k2->set_name("k2");
  k2->add_sigma_operator( ioperator("none") );
  k2->add_sigma_operator( ioperator("<=1") );
  k2->add_sigma_operator( ioperator(">=1") );
  k2->set_localexecutefn(  &threepointsumbump );

  shared_ptr<task> t1,t2; vector<shared_ptr<message>> msgs;
  REQUIRE_NOTHROW( k1->analyze_dependencies() );
  REQUIRE_NOTHROW( t1 = k1->get_tasks().at(0) );
  REQUIRE_NOTHROW( msgs = t1->get_receive_messages() );
  if (mytid==0 || mytid==ntids-1)
    CHECK( msgs.size()==2 );
  else
    CHECK( msgs.size()==3);

  REQUIRE_NOTHROW( k2->analyze_dependencies() );
  REQUIRE_NOTHROW( t2 = k2->get_tasks().at(0) );
  REQUIRE_NOTHROW( msgs = t2->get_receive_messages() );
  CHECK( msgs.size()==1 );
}

TEST_CASE( "communication avoiding distributions, use of beta","[kernel][avoid][43]" ) {
  int nlocal=100,nglobal=nlocal*ntids, dim=1;
  shared_ptr<distribution> block,extend;
  REQUIRE_NOTHROW
    ( block = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,nlocal,-1) ) );
  auto
    my_first = block->first_index_r(mycoord),
    my_last = block->last_index_r(mycoord);

  ioperator noop("none"), right(">=1"), left("<=1");

  const char *from;
  SECTION( "extended distribution from kernel" ) { from = "kernel";
    auto t0 = block->new_object(block), t1 = block->new_object(block);
    t0->set_name("blocked-t0"); t1->set_name("blocked-t1");
    shared_ptr<kernel> k = shared_ptr<kernel>( new mpi_kernel(t0,t1) );
    k->set_name("ktest");
    k->add_sigma_operator( noop );
    k->add_sigma_operator( left);
    k->add_sigma_operator( right );
    REQUIRE_NOTHROW( k->analyze_dependencies() );
    REQUIRE_NOTHROW( extend = k->last_dependency().get_beta_object()->get_distribution() );
  }
  
  SECTION( "extended distribution from signature" ) { from = "signature";
    signature_function f;
    f.add_sigma_operator( noop );
    f.add_sigma_operator( left);
    f.add_sigma_operator( right );
    REQUIRE_NOTHROW
      ( extend = shared_ptr<distribution>
	( new mpi_distribution( f.derive_beta_structure(block,block->get_enclosing_structure() ) ) ) );
  }
  INFO( "mytid=" << mytid );
  INFO( "distribution extended from " << from );
  INFO( format("original distribution: {}",block->as_string()) );
  INFO( format("extended distribution: {}",extend->as_string()) );

  shared_ptr<object> v0,v1,v2;
  REQUIRE_NOTHROW( v0 = block->new_object(block) ); v0->set_name("v0");
  REQUIRE_NOTHROW( v1 = extend->new_object(extend) ); v1->set_name("v1");
  REQUIRE_NOTHROW( v2 = block->new_object(block) ); v2->set_name("v2");
  shared_ptr<kernel> k1,k2;
  REQUIRE_NOTHROW( k1 = shared_ptr<kernel>( new mpi_kernel( v0,v1 ) ) );
  k1->set_name("k1");
  k1->add_sigma_operator( noop );
  k1->add_sigma_operator( left );
  k1->add_sigma_operator( right );
  k1->set_localexecutefn(  &threepointsumbump );
  REQUIRE_NOTHROW( k2 = shared_ptr<kernel>( new mpi_kernel( v1,v2 ) ) );
  k2->set_name("k2");
  k2->add_sigma_operator( noop );
  k2->add_sigma_operator( left);
  k2->add_sigma_operator( right );
  k2->set_localexecutefn(  &threepointsumbump );

  shared_ptr<task> t1,t2; vector<shared_ptr<message>> msgs;
  REQUIRE_NOTHROW( k1->analyze_dependencies() );
  REQUIRE_NOTHROW( t1 = k1->get_tasks().at(0) );
  REQUIRE_NOTHROW( msgs = t1->get_receive_messages() );
  if (ntids==1) 
    CHECK( msgs.size()==1 );
  else if (mytid==0 || mytid==ntids-1)
    CHECK( msgs.size()==2 );
  else
    CHECK( msgs.size()==3);

  REQUIRE_NOTHROW( k2->analyze_dependencies() );
  REQUIRE_NOTHROW( t2 = k2->get_tasks().at(0) );
  REQUIRE_NOTHROW( msgs = t2->get_receive_messages() );
  CHECK( msgs.size()==1 );
}

TEST_CASE( "s-step with communication avoiding distributions, iterated","[kernel][avoid][44]" ) {

  architecture *aa; decomposition decomp;
  REQUIRE_NOTHROW( aa = env->make_architecture() );
  int can_embed;
  SECTION( "no embedding" ) { can_embed = 0; }
  SECTION( "yes embedding" ) { can_embed = 1; }
  REQUIRE_NOTHROW( aa->set_can_embed_in_beta(can_embed) );
  REQUIRE_NOTHROW( decomp = mpi_decomposition(aa) );

  int nlocal=100,nglobal=nlocal*ntids, dim=1; int nsteps = 3;
  INFO( "mytid=" << mytid );
  INFO( "embedding: " << can_embed );

  // Set up the usual three point stencil
  ioperator noop("none"), right(">=1"), left("<=1");
  signature_function sigma_f;
  sigma_f.add_sigma_operator( noop );
  sigma_f.add_sigma_operator( left);
  sigma_f.add_sigma_operator( right );

  // Distributions and objects
  
  // Block distribution for first and last object
  auto block_dist = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,nglobal) );
  auto halo_dists = vector<shared_ptr<distribution>>(nsteps);
  auto objects = new shared_ptr<object>[nsteps+1];
  objects[0] = block_dist->new_object(block_dist);
  objects[nsteps] = block_dist->new_object(block_dist);

  // Recursively derived non-disjoint distributions in between
  // global structure for truncation
  auto global_structure = block_dist->get_global_structure();
  INFO( "global structure: " << global_structure->as_string() );
  for (int istep=nsteps-1; istep>=0; istep--) {
    INFO( "step " << istep << " in [0-" << nsteps << "]" );
    shared_ptr<distribution> out_dist;
    if (istep==nsteps-1)
      out_dist = block_dist; else out_dist = halo_dists[istep+1];
    REQUIRE_NOTHROW
      ( halo_dists[istep] =
	//make_shared<distribution>(shared_ptr<mpi_distribution>
	shared_ptr<distribution>( make_shared<mpi_distribution>
				  ( sigma_f.derive_beta_structure(out_dist,global_structure) )
				  )
	);
    INFO( format("halo structure: {} vs global_structure: {}",
		 halo_dists[istep]->get_global_structure()->as_string(),
		 global_structure->as_string() ) );
    CHECK( halo_dists[istep]->get_global_structure()->equals( global_structure ) );
  }

  for (int istep=1; istep<nsteps; istep++)
    REQUIRE_NOTHROW( objects[istep] = halo_dists[istep]->new_object(halo_dists[istep]) );
  // check the local sizes of the objects: tapering up
  for (int istep=nsteps; istep>0; istep--) {
    int excess = nsteps-istep;
    INFO( fmt::format("Step {} out of {}, excess={}\n",istep,nsteps,excess) );
    if (ntids==1)
      CHECK( objects[istep]->volume(mycoord)==nlocal );
    else if (mytid==0 || mytid==ntids-1)
      CHECK( objects[istep]->volume(mycoord)==nlocal+excess );
    else
      CHECK( objects[istep]->volume(mycoord)==nlocal+2*excess );
  }
  

  algorithm sstep = mpi_algorithm(decomp);
  sstep.add_kernel( shared_ptr<kernel>( new mpi_origin_kernel(objects[0]) ) );
  vector<shared_ptr<kernel>> kernels(nsteps);
  for (int istep=0; istep<nsteps; istep++) {
    auto k = shared_ptr<kernel>( new mpi_kernel(objects[istep],objects[istep+1]) );
    k->set_localexecutefn( &threepointsumbump );
    k->set_explicit_beta_distribution(halo_dists[istep]);
    kernels.at(istep) = k;
    sstep.add_kernel(k);
  }
    
  REQUIRE_NOTHROW( sstep.analyze_dependencies() );

  for (int istep=nsteps-1; istep>=0; istep--) {
    int excess = nsteps-istep;
    shared_ptr<object> halo,in,out;
    INFO( "step " << istep );
    REQUIRE_NOTHROW( halo = kernels.at(istep)->get_beta_object(0) );
    REQUIRE_NOTHROW( in = kernels.at(istep)->get_in_object(0) );
    REQUIRE_NOTHROW( out = kernels.at(istep)->get_out_object() );
    INFO(
	 "out structure: " << out->get_processor_structure(mycoord)->as_string() << "\n" <<
	 "halo structure: " << halo->get_processor_structure(mycoord)->as_string() << "\n" <<
	 "in structure: " << in->get_processor_structure(mycoord)->as_string()
	 );
    if (ntids==1)
      CHECK( halo->volume(mycoord)==nlocal );
    else if (mytid==0 || mytid==ntids-1)
      CHECK( halo->volume(mycoord)==nlocal+excess );
    else
      CHECK( halo->volume(mycoord)==nlocal+2*excess );
    if (istep>0) {
      // halo and in should be aligned so that there is no message
      CHECK( halo->first_index_r(mycoord)==in->first_index_r(mycoord) );
      CHECK( halo->last_index_r(mycoord)==in->last_index_r(mycoord) );
      // halo and out have the usual relation with sticking out 1
      if (mytid==0)
	CHECK( halo->first_index_r(mycoord)==out->first_index_r(mycoord) );
      else 
	CHECK( halo->first_index_r(mycoord)==out->first_index_r(mycoord)-1 );
      if (mytid==ntids-1)
	CHECK( halo->last_index_r(mycoord)==out->last_index_r(mycoord) );
      else
	CHECK( halo->last_index_r(mycoord)==out->last_index_r(mycoord)+1 );
    } else { // in step 0 the halo is bit bigger
      if (mytid==0) 
	CHECK( halo->first_index_r(mycoord)==in->first_index_r(mycoord) );
      else
	CHECK( halo->first_index_r(mycoord)==in->first_index_r(mycoord)-excess );
      if (mytid==ntids-1)
	CHECK( halo->last_index_r(mycoord)==in->last_index_r(mycoord) );
      else
	CHECK( halo->last_index_r(mycoord)==in->last_index_r(mycoord)+excess );
    }
  }

  for (int istep=0; istep<nsteps; istep++) {
    shared_ptr<task> t;
    REQUIRE_NOTHROW( t = kernels.at(istep)->get_tasks().at(0) );
    REQUIRE( t!=nullptr );
    vector<shared_ptr<message>> msgs;
    REQUIRE_NOTHROW( msgs = t->get_receive_messages() );
    INFO( "level " << istep );
    memory_buffer w;
    for ( auto m : msgs ) {
      format_to(w.end(),"{}->{} ",m->get_sender().as_string(),m->get_receiver().as_string()); }
    INFO( "messages: " << to_string(w) );
    if (istep==0) {
      if (ntids==1)
	CHECK( msgs.size()==1 );
      else if (mytid==0 || mytid==ntids-1)
	CHECK( msgs.size()==2 ); // self + 1 neigh
      else
	CHECK( msgs.size()==3 ); // self + 2 neigh
    } else {
      CHECK( msgs.size()==1 ); // only self message
    }
  }

  REQUIRE_NOTHROW( sstep.execute(true) );
}

TEST_CASE( "Kernel data graph: test acyclicity","[kernel][queue][50]" ) {

  int localsize = 20,gsize = localsize*ntids;
  shared_ptr<distribution> block = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,localsize,-1) );
  shared_ptr<object> object1,object2,object3,object4;
  shared_ptr<kernel> kernel1,kernel2,kernel3;
  mpi_algorithm *queue;
  char *object_name,*test_name;

  SECTION( "unnamed objects" ) {
    object1 = block->new_object(block);
    object2 = block->new_object(block);
    object3 = block->new_object(block);
    object4 = block->new_object(block);

    kernel1 = shared_ptr<kernel>( new mpi_kernel(object1,object2) );
    kernel1->set_name("50kernel1");
    kernel2 = shared_ptr<kernel>( new mpi_kernel(object1,object3) );
    kernel2->set_name("50kernel2");
    kernel3 = shared_ptr<kernel>( new mpi_kernel(object2,object3) );
    kernel3->set_name("50kernel3");

    algorithm queue = mpi_algorithm(decomp);
    REQUIRE_NOTHROW( queue.add_kernel(kernel1) );
    REQUIRE_NOTHROW( queue.add_kernel(kernel2) );
    REQUIRE_NOTHROW( queue.add_kernel(kernel3) );

  }

  SECTION( "named objects" ) {
    object1 = block->new_object(block);
    object1->set_name("object1");
    object2 = block->new_object(block);
    object2->set_name("object2");
    object3 = block->new_object(block);
    object3->set_name("object3");
    object4 = block->new_object(block);
    object4->set_name("object4");

    kernel1 = shared_ptr<kernel>( new mpi_kernel(object1,object2) );
    kernel1->set_name("make2");
    kernel2 = shared_ptr<kernel>( new mpi_kernel(object1,object3) );
    kernel2->set_name("make3");
    kernel3 = shared_ptr<kernel>( new mpi_kernel(object2,object3) );
    kernel3->set_name("make4");

    algorithm queue = mpi_algorithm(decomp);
    REQUIRE_NOTHROW( queue.add_kernel(kernel1) );
    REQUIRE_NOTHROW( queue.add_kernel(kernel2) );
    REQUIRE_NOTHROW( queue.add_kernel(kernel3) );

    {
      // there are two kernels that use object1, none that generate it
      vector<shared_ptr<kernel>> def,use;
      REQUIRE_NOTHROW( queue.get_data_relations("object1",def,use) );
      CHECK( def.size()==0 );
      CHECK( use.size()==2 );
    }
    {
      // there are two kernels that generate object3; we do not allow that.
      vector<shared_ptr<kernel>> def,use;
      REQUIRE_NOTHROW( queue.get_data_relations("object3",def,use) );
      CHECK( def.size()==2 );
    }
  }

  SECTION( "kernel predecessors" ) {
    object1 = block->new_object(block);
    object1->set_name("object1");
    object2 = block->new_object(block);
    object2->set_name("object2");
    object3 = block->new_object(block);
    object3->set_name("object3");
    object4 = block->new_object(block);
    object4->set_name("object4");

    kernel1 = shared_ptr<kernel>( new mpi_kernel(object1,object2) );
    kernel1->set_name("make2");
    kernel2 = shared_ptr<kernel>( new mpi_kernel(object2,object3) );
    kernel2->set_name("make3");

    queue = new mpi_algorithm(decomp);
    REQUIRE_NOTHROW( queue->add_kernel(kernel1) );
    REQUIRE_NOTHROW( queue->add_kernel(kernel2) );

    // shared_ptr<kernel> def;
    // vector<kernel*> *use;
    // // there are two kernels that use object1, none that generate it
    // REQUIRE_NOTHROW( queue->get_predecessors("make3",&use) );
    // REQUIRE( use->size()==1 );
    // // there are two kernels that generate object3; we do not allow that.
    // CHECK( (*use)[0]->get_name()=="make2" );
  }
}

TEST_CASE( "Mapping between redundant distros","[redundant][70]" ) {

  auto
    din = shared_ptr<distribution>( new mpi_replicated_distribution(decomp) ),
    dout = shared_ptr<distribution>( new mpi_replicated_distribution(decomp) );
  CHECK( din->has_defined_type() );
  auto				     
    scalar_in =  din->new_object( din ),
    scalar_out =  dout->new_object( dout );
  CHECK( scalar_in->get_distribution()->has_defined_type() );
  scalar_in->allocate(); scalar_out->allocate();
  auto indata = scalar_in->get_data(mycoord),
    outdata = scalar_out->get_data(mycoord);
  indata.at(0) = 15.3;

  auto copy = shared_ptr<kernel>( new mpi_kernel( scalar_in,scalar_out ) );
  copy->set_explicit_beta_distribution(dout);
  shared_ptr<task> copy_task;

  CHECK_NOTHROW( copy->analyze_dependencies() );
  CHECK_NOTHROW( copy_task = copy->get_tasks().at(0) );
  vector<shared_ptr<message>> msgs;
  REQUIRE_NOTHROW( msgs = copy_task->get_receive_messages() );
  CHECK( msgs.size()==1 );
  CHECK( msgs.at(0)->get_receiver()==mycoord );
  CHECK( msgs.at(0)->get_sender()==mycoord );
}

TEST_CASE( "Interpolation by restriction", "[stretch][distribution][71]" ) {

  int nlocal=10, dim=1;
  auto target_dist = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,-1,nlocal*ntids) ),
    source_dist = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,-1,2*nlocal*ntids) );
  auto
    myfirst = target_dist->first_index_r(mycoord),
    mylast = target_dist->last_index_r(mycoord);
auto
    target = target_dist->new_object(target_dist),
source = source_dist->new_object(source_dist);
 auto restrict = shared_ptr<kernel>( new mpi_kernel(source,target) );
  
  CHECK_NOTHROW( restrict->add_sigma_operator( ioperator("*2") ) );
  /* mpi_task* */ shared_ptr<task> t;
  REQUIRE_NOTHROW( restrict->set_last_dependency().endow_beta_object(target) );
  CHECK_NOTHROW( restrict->split_to_tasks() );
  CHECK( restrict->get_tasks().size()==1 );
  CHECK_NOTHROW( t = restrict->get_tasks().at(0) ) ;

  {
    vector<multi_ioperator*> ops;
    CHECK_NOTHROW( ops = restrict->last_dependency().get_operators() );
    ioperator beta_op;
    CHECK_NOTHROW( beta_op = ops.at(0)->get_operator(0) );
    CHECK( beta_op.is_restrict_op() );

    auto gamma_struct = target_dist->get_processor_structure(mycoord);
    shared_ptr<multi_indexstruct> beta_block;

    // spell it out: operate_and_breakup
    CHECK_NOTHROW( beta_block = gamma_struct->operate
		   ( beta_op,source_dist->get_enclosing_structure() ) );
    CHECK( beta_block->get_component(0)->stride()==2 );
    vector<shared_ptr<message>> msgs;

    SECTION( "carefully" ) {
      REQUIRE_NOTHROW( msgs = source_dist->messages_for_segment
		       ( 0,mycoord,self_treatment::INCLUDE,
			 beta_block,beta_block ) );
      CHECK( msgs.size()==1 );
      auto mtmp = msgs.at(0);
      auto global_struct = mtmp->get_global_struct(),
	local_struct = mtmp->get_local_struct();
      auto localpt = domain_coordinate( vector<index_int>{nlocal} );
      CHECK( global_struct->local_size_r()==localpt );
      CHECK( global_struct->first_index_r()==myfirst*2 ); // src in the big alpha vec
      CHECK( ( global_struct->last_index_r()-global_struct->first_index_r() )==localpt*2-2 );
      // tar relative to the halo
      CHECK( local_struct->first_index_r()==domain_coordinate_zero(dim) );
    }

    // SECTION( "same thing in one go" ) {
    //   CHECK_NOTHROW( msgs = source_dist->analyze_one_dependence
    // 		     (mytid,0,beta_op,target_dist,beta_block) );
    // }
  }
}

TEST_CASE( "Scale queue","[queue][execute][105]" ) {

  INFO( "mytid=" << mytid );

  int nlocal=17,nsteps=1, dim=1;
  ioperator no_op("none");
  shared_ptr<distribution> block = 
    shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,-1,nlocal*ntids) );
  auto
    my_first = block->first_index_r(mycoord),
    my_last = block->last_index_r(mycoord);
  CHECK( my_first==mycoord_coord*nlocal );
  CHECK( my_last==(mycoord_coord+1)*nlocal-1 );

  //  auto xdata = make_shared<vector<double>>(vector<double>(nlocal));
  data_pointer xdata;
  auto
    xvector = block->new_object(block),
    *yvector = new shared_ptr<object>[nsteps];
  xvector->set_name("initial-x");
  REQUIRE_NOTHROW( xvector->allocate() );
  REQUIRE( xvector->has_data_status_allocated() );
  REQUIRE_NOTHROW( xdata = xvector->get_data(mycoord) );
  for (int i=0; i<nlocal; i++)
    xdata.at(i) = pointfunc33(i,my_first[0]);
  for (int iv=0; iv<nsteps; iv++) {
    yvector[iv] = block->new_object(block);
    yvector[iv]->set_name(fmt::format("y@step{}",iv));
    //yvector[iv]->allocate(); // VLE disable this!
  }

  vector<shared_ptr<kernel>> kernels(nsteps+1);
  
  for (int iv=0; iv<=nsteps; iv++) {
    shared_ptr<kernel> k;
    if (iv==0) {
      k = shared_ptr<kernel>( new mpi_origin_kernel(xvector) );
      k->set_name(fmt::format("origin"));
    } else {
      if (iv==1) {
	k = shared_ptr<kernel>( new mpi_kernel(xvector,yvector[0]) );
      } else {
	k = shared_ptr<kernel>( new mpi_kernel(yvector[iv-2],yvector[iv-1]) );
      }
      k->set_name(fmt::format("update{}",iv-1));
      k->set_localexecutefn(  &vecscalebytwo );
      k->add_sigma_operator( no_op );
    }
    kernels[iv] = k;
  }

  mpi_algorithm *queue;
  CHECK_NOTHROW( queue = new mpi_algorithm(decomp) );
  // {
  //   shared_ptr<kernel> k;
  //   REQUIRE_NOTHROW( k = shared_ptr<kernel>( new mpi_origin_kernel(xvector) ) );
  //   k->set_name("originate x");
  //   REQUIRE_NOTHROW( queue->add_kernel(k) );
  // }

  const char *path;
  SECTION( "just execute kernels" ) {
    for (int iv=0; iv<nsteps+1; iv++) {
      shared_ptr<distribution> beta;
      REQUIRE_NOTHROW( kernels[iv]->analyze_dependencies() );
      if (iv>0) {
	REQUIRE_NOTHROW( beta = kernels[iv]->last_dependency().get_beta_object()->get_distribution() );
	CHECK( beta->get_processor_structure(mycoord)
	       ->equals( block->get_processor_structure(mycoord) ) );
      }
      REQUIRE_NOTHROW( kernels[iv]->execute() );
    }
  }
  SECTION( "actual queue" ) {
    SECTION( "declaring kernels in sequence" ) {
      SECTION ( "from default zero" ) {
	path = "in sequence";
      }
      SECTION( "from 10" ) {
	path = "in sequence with step zero offset";
	REQUIRE_NOTHROW( queue->set_kernel_zero(10) );
      }
      for (int iv=0; iv<nsteps+1; iv++) {
	CHECK_NOTHROW( queue->add_kernel(kernels[iv]) );
      }
    }
    SECTION( "declaring kernels in reverse sequence" ) {
      SECTION ( "from default zero" ) {
	path = "reversed";
      }
      SECTION( "from 10" ) {
	path = "reversed from 10";
	REQUIRE_NOTHROW( queue->set_kernel_zero(10) );
      }
      for (int iv=nsteps; iv>=0; iv--) {
	CHECK_NOTHROW( queue->add_kernel(kernels[iv]) );
      }
    }
    INFO( "kernels declared: " << path );

    REQUIRE_NOTHROW( queue->analyze_dependencies() );
    vector<shared_ptr<task>> tsks;
    REQUIRE_NOTHROW( tsks = queue->get_tasks() );
    CHECK( tsks.size()==(nsteps+1) );
    vector<shared_ptr<task>> predecessors;
    for ( auto t : tsks ) {
      vector<shared_ptr<message>> msgs;
      int step = t->get_step();
      CHECK( step>=0 );
      CHECK_NOTHROW( predecessors = t->get_predecessors() );
      INFO( "step: " << step );
      if (t->has_type_origin()) {
	CHECK( predecessors.size()==0 );
      } else {
	CHECK( predecessors.size()==1 );
	CHECK_NOTHROW( msgs = t->get_receive_messages() );
	CHECK( msgs.size()==1 );
	CHECK_NOTHROW( msgs = t->get_send_messages() );
	CHECK( msgs.size()==1 );
      }
    }
    vector<shared_ptr<task>> exits;
    REQUIRE_NOTHROW( exits = queue->get_exit_tasks() );
    CHECK( exits.size()==1 ); // that's locally

    CHECK_NOTHROW( queue->execute() );
    CHECK( queue->get_all_tasks_executed() );
  }

  // let's inspect some halos
  // int found=0;
  // for (auto t=queue->get_tasks().begin(); t!=queue->get_tasks().end(); ++t) {
  //   if (t->get_step()==yvector[0]->get_object_number()) {
  shared_ptr<task> tsk; REQUIRE_NOTHROW( tsk = kernels[1]->get_tasks().at(0) );
  //      found++;
  {
    shared_ptr<object> h;
    data_pointer hdata;
    REQUIRE_NOTHROW( h=tsk->get_beta_object(0) );
    CHECK( h->volume(mycoord)==nlocal );
    INFO( "halo data from: " << h->get_name() );
    REQUIRE_NOTHROW( hdata = h->get_data(mycoord) );
    for (int i=0; i<nlocal; i++) {
      INFO( "hvalue:" << i << ":" << hdata.at(i) );
      CHECK( hdata.at(i) == Approx( xdata.at(i) ) );
    }
  }
  //  REQUIRE( found==1 );
  
  for (int step=0; step<nsteps; step++) {
    data_pointer ydata;
    INFO( "step: " << step );
    CHECK_NOTHROW( ydata = yvector[step]->get_data(mycoord) );
    for (int i=0; i<nlocal; i++) {
      INFO( "yvalue:" << i << ":" << ydata.at(i) );
      CHECK( ydata.at(i) == Approx( pointfunc33(i,my_first[0]) * pow(2,step+1) ) );
    }
  }
}

TEST_CASE( "Threepoint queue bump","[queue][execute][halo][106]" ) {

  INFO( "mytid=" << mytid );

  int nlocal=7,nsteps=2, dim=1;
  shared_ptr<distribution> block = 
    shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,-1,nlocal*ntids) );
  auto
    my_first = block->first_index_r(mycoord),
    my_last = block->last_index_r(mycoord);
  CHECK( my_first==mycoord_coord*nlocal );
  CHECK( my_last==(mycoord_coord+1)*nlocal-1 );

  ioperator no_op("none"), right_shift(">=1"), left_shift("<=1");

  auto xdata = data_allocate(nlocal);
  data_pointer ydata;
  auto
    xvector = block->new_object_from_data(xdata);
  auto 
    yvector = vector<shared_ptr<object>>(nsteps);
  xvector->set_name("xorigin");
  for (int i=0; i<nlocal; i++)
    xdata->at(i) = 1.;
  for (int iv=0; iv<nsteps; iv++) {
    yvector[iv] = block->new_object(block);
    yvector[iv]->set_name(fmt::format("yvector{}",iv));
  }
  int last_object_number;
  REQUIRE_NOTHROW( last_object_number = yvector.at(nsteps-1)->get_object_number() );

  mpi_algorithm queue;
  CHECK_NOTHROW( queue = mpi_algorithm(decomp) );
  {
    shared_ptr<kernel> k = shared_ptr<kernel>( new mpi_origin_kernel(xvector) );
    REQUIRE_NOTHROW( queue.add_kernel(k) );
  }
  for (int iv=0; iv<nsteps; iv++) {
    shared_ptr<kernel> k;
    if (iv==0) {
      k = shared_ptr<kernel>( new mpi_kernel(xvector,yvector[0]) );
    } else {
      k = shared_ptr<kernel>( new mpi_kernel(yvector[iv-1],yvector[iv]) );
    }
    k->set_name(fmt::format("compute-y{}",iv));
    k->add_sigma_operator( no_op );
    k->add_sigma_operator( left_shift );
    k->add_sigma_operator( right_shift );
    k->set_localexecutefn(  &threepointsumbump );
    CHECK_NOTHROW( queue.add_kernel(k) );
  }
  
  { // do all kernels have a proper step number?
    vector<shared_ptr<kernel>> kernels;
    REQUIRE_NOTHROW( kernels = queue.get_kernels() );
    for ( auto k : kernels ) {
      int kn,kv;
      REQUIRE_NOTHROW( kv = k->get_out_object()->get_object_number() );
      REQUIRE_NOTHROW( kn = k->get_step() );
      CHECK( kn==kv );
    }
  }

  const char *section, *analysis;
  int opt=0; int queue_analysis = 0;
  
  SECTION( "unoptimized" ) {
    section = "unoptimized";

    SECTION( "single analysis call" ) {
      analysis = "in one";
      CHECK_NOTHROW( queue.analyze_dependencies() );
    }

    SECTION( "detailed analysis" ) {
      analysis = "detailed";
      vector<shared_ptr<kernel>> kernels;
      CHECK_NOTHROW( kernels = queue.get_kernels() );
      CHECK( kernels.size()==(nsteps+1) );
      int ik = 0,io,iozero;
      // 1: loop over all kernels
      for ( auto k : kernels ) { 
	int in_object_number=-1;
	if (!k->has_type_origin()) {
	  REQUIRE_NOTHROW
	    ( in_object_number
	      = k->last_dependency().get_in_object()->get_object_number() );
	} 
	// 2: analyze kernel
	k->analyze_dependencies();
	auto ktasks = k->get_tasks();
	// 3: loop over tasks
	for ( auto t : ktasks ) {
	  auto coord = t->get_domain(); int tid = coord.coord(0);
	  // lots of task testing
	  CHECK( tid==mytid ); // mpi-dependent
	  vector<shared_ptr<message>> msgs;
	  if (t->has_type_origin()) {
	    CHECK( t->get_n_in_objects()==0 );
	  } else {
	    REQUIRE_NOTHROW( msgs = t->get_send_messages() );
	    CHECK( t->get_n_in_objects()>0 );
	    // the invector is the product of the previous kernel
	    shared_ptr<object> inobject;
	    CHECK_NOTHROW( inobject = t->get_in_object(0) );
	    int origin;
	    REQUIRE_NOTHROW( origin = inobject->get_object_number() );
	    shared_ptr<task> otsk;
	    REQUIRE_NOTHROW( otsk = queue.find_task_by_coordinates(origin,coord) );
	    // msgs can come from one of three neighbours except at the edge
	    if (ntids==1)
	      CHECK( msgs.size()==1 );
	    else if (mytid==0 || mytid==ntids-1)
	      CHECK( msgs.size()==2 );
	    else
	      CHECK( msgs.size()==3 );
	    for ( auto m : msgs )
	      CHECK( ( m->get_sender()==mycoord
		       || m->get_sender()==mycoord-1
		       || m->get_sender()==mycoord+1 ) );
	  }
	  // 4: add task to graph
	  queue.add_task( t );
	  t->find_other_task_by_coordinates =
	    [queue] (int s,processor_coordinate &d) -> shared_ptr<task> {
	    return queue.find_task_by_coordinates(s,d); };
	}
	ik++;
      }
      // kernels have been analyzed;
      REQUIRE_NOTHROW( queue.find_predecessors() );
      auto tasks = queue.get_tasks();
      for ( auto t : tasks ) {
	auto predecessors = t->get_predecessors();
	if (t->has_type_origin())
	  CHECK( predecessors.size()==0 );
	else {
	  if (ntids==1)
	    CHECK( predecessors.size()==1 );
	  else if (mytid==0 || mytid==ntids-1)
	    CHECK( predecessors.size()==2 );
	  else
	    CHECK( predecessors.size()==3 );
	  for ( auto t : predecessors ) {
	    int proc;
	    CHECK_NOTHROW( proc = t->get_domain().coord(0) );
	    CHECK( proc==mytid );
	  }
	}
      }
    }

    INFO( format("Version {}, analysis: {}",section,analysis) );
    vector<shared_ptr<task>> tsks;
    REQUIRE_NOTHROW( tsks = queue.get_tasks() );
    CHECK( tsks.size()==(nsteps+1) );
    for ( auto t : tsks ) {
      std::vector<std::shared_ptr<task>> predecessors;
      CHECK_NOTHROW( predecessors = t->get_predecessors() );
      INFO( "step=" << t->get_step() );
      if (t->has_type_origin()) {
	// receive messages have been cleared
	CHECK( t->get_receive_messages().size()==0 );
	CHECK( predecessors.size()==0 );
      } else {
	INFO( format("Computer task has beta distribution: {}",
		     t->get_beta_object(0)->get_distribution()->as_string()) );
	if (ntids==1) {
	  REQUIRE( t->get_receive_messages().size()==1 );
	  CHECK( predecessors.size()==1 );
	} else if (mytid==0 || mytid==ntids-1) {
	  REQUIRE( t->get_receive_messages().size()==2 );
	  CHECK( predecessors.size()==2 );
	} else {
	  REQUIRE( t->get_receive_messages().size()==3 );
	  CHECK( predecessors.size()==3 );
	}
      }
    }

  }

  SECTION( "analysis with optimization" ) {
    section = "optimized"; opt = 1; queue_analysis = 1;
    
    REQUIRE_NOTHROW( queue.analyze_dependencies() );
    REQUIRE_NOTHROW( queue.optimize() );

    vector<shared_ptr<task>> exits;
    REQUIRE_NOTHROW( exits = queue.get_exit_tasks() );
    CHECK( exits.size()==1 ); // that's locally

    vector<shared_ptr<task>> tsks;
    REQUIRE_NOTHROW( tsks = queue.get_tasks() );
    //for (auto t=tsks.begin(); t!=tsks.end(); ++t) {
    for ( auto t : tsks ) {
      INFO( "step=" << t->get_step() );
      if (!t->has_type_origin()) {
	//CHECK_THROWS( t->get_send_messages().size() );
	CHECK( t->get_send_messages().size()==0 );
	//CHECK_THROWS( t->get_receive_messages().size() );
	CHECK( t->get_receive_messages().size()==0 );
      }
      if (t->get_out_object()->get_object_number()==last_object_number) {
	CHECK( t->get_post_messages().size()==0 );
	CHECK( t->get_xpct_messages().size()==0 );
      } else {
	if (mytid==0 || mytid==ntids-1) {
	  CHECK( t->get_post_messages().size()==2 );
	  CHECK( t->get_xpct_messages().size()==2 );
	} else {
	  CHECK( t->get_post_messages().size()==3 );
	  CHECK( t->get_xpct_messages().size()==3 );
	}
      }
    }
  }
  
  INFO( fmt::format("P={}, queue was {}",mytid,section) );

  CHECK_NOTHROW( queue.execute() );

  /*
   * The rest of this unit test is probably predicated
   * upon queue analysis having been performed
   */  
  if (!queue_analysis) return;
  CHECK( queue.get_all_tasks_executed() );
  CHECK( queue.get_all_msgs_completed() );

  // let's inspect some halos
  int found=0;
  for (auto t : queue.get_tasks() ) {
    if (t->get_step()==yvector[0]->get_object_number()) {
      found++;
      shared_ptr<object> h; data_pointer data;
      REQUIRE_NOTHROW( h=t->get_beta_object(0) );
      if (mytid==0 || mytid==ntids-1)
	CHECK( h->volume(mycoord)==(nlocal+1) );
      else
	CHECK( h->volume(mycoord)==(nlocal+2) );
      REQUIRE_NOTHROW( data = h->get_data(mycoord) );
      index_int lo=0,hi=nlocal+2;
      if (mytid==0)       hi -= 1;
      if (mytid==ntids-1) hi -= 1;
      for (index_int i=0; i<hi; i++) {
	INFO( fmt::format("zero step halo, yvalue-{}: {}",i,data.at(i)) );
	CHECK( data.at(i)==Approx(1.) );
      }
    }
  }
  REQUIRE( found==1 );
  
  {
    for (int s=0; s<nsteps; s++) {
      int i; index_int lo=0,hi=nlocal;
      if (mytid==0) lo += s+1;
      if (mytid==ntids-1) hi -= s+1;
      CHECK_NOTHROW( ydata = yvector[s]->get_data(mycoord) );
      for (i=lo; i<hi; i++) {
	INFO( fmt::format("step {}: yloc[{}] = {}",s,i,ydata.at(i)) );
	CHECK( ydata.at(i) == Approx( pow(3,s+1) ) );
      }
    }
  }

}

TEST_CASE( "Threepoint queue module","[queue][execute][halo][modulo][107]" ) {

  INFO( "mytid=" << mytid );

  int nlocal=7,nsteps=2, dim=1;
  shared_ptr<distribution> block = 
    shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,-1,nlocal*ntids) );
  auto
    my_first = block->first_index_r(mycoord),
    my_last = block->last_index_r(mycoord);
  CHECK( my_first==mycoord_coord*nlocal );
  CHECK( my_last==(mycoord_coord+1)*nlocal-1 );

  ioperator no_op("none"), right_shift_mod(">>1"), left_shift_mod("<<1");

  auto xdata = data_allocate(nlocal);
  data_pointer ydata;
  auto
    xvector = block->new_object_from_data(xdata);
  auto 
    yvector = vector<shared_ptr<object>>(nsteps);
  xvector->set_name("xorigin");
  for (int i=0; i<nlocal; i++)
    xdata->at(i) = 1.;
  for (int iv=0; iv<nsteps; iv++) {
    yvector[iv] = block->new_object(block);
    yvector[iv]->set_name(fmt::format("yvector{}",iv));
  }
  int last_object_number;
  REQUIRE_NOTHROW( last_object_number = yvector[nsteps-1]->get_object_number() );

  mpi_algorithm queue;
  CHECK_NOTHROW( queue = mpi_algorithm(decomp) );
  {
    shared_ptr<kernel> k = shared_ptr<kernel>( new mpi_origin_kernel(xvector) );
    REQUIRE_NOTHROW( queue.add_kernel(k) );
  }
  for (int iv=0; iv<nsteps; iv++) {
    shared_ptr<kernel> k;
    if (iv==0) {
      k = shared_ptr<kernel>( new mpi_kernel(xvector,yvector[0]) );
    } else {
      k = shared_ptr<kernel>( new mpi_kernel(yvector[iv-1],yvector[iv]) );
    }
    k->set_name(fmt::format("compute-y{}",iv));
    k->add_sigma_operator( no_op );
    k->add_sigma_operator( left_shift_mod );
    k->add_sigma_operator( right_shift_mod );
    k->set_localexecutefn(  &threepointsummod );
    CHECK_NOTHROW( queue.add_kernel(k) );
  }
  
  { // do all kernels have a proper step number?
    vector<shared_ptr<kernel>> kernels;
    REQUIRE_NOTHROW( kernels = queue.get_kernels() );
    for ( auto k : kernels ) {
      int kn,kv;
      REQUIRE_NOTHROW( kv = k->get_out_object()->get_object_number() );
      REQUIRE_NOTHROW( kn = k->get_step() );
      CHECK( kn==kv );
    }
  }

  const char *section; int opt=0; int queue_analysis = 0;
  
  SECTION( "unoptimized" ) {
    section = "unoptimized";
    
    SECTION( "detailed analysis" ) {
      vector<shared_ptr<kernel>> kernels;
      CHECK_NOTHROW( kernels = queue.get_kernels() );
      CHECK( kernels.size()==(nsteps+1) );
      int ik = 0,io,iozero;
      // 1: loop over all kernels
      for ( auto k : kernels ) { 
	int in_object_number=-1;
	if (!k->has_type_origin()) {
	  REQUIRE_NOTHROW
	    ( in_object_number
	      = k->last_dependency().get_in_object()->get_object_number() );
	} 
	// 2: analyze kernel
	k->analyze_dependencies();
	auto ktasks = k->get_tasks();
	// 3: loop over tasks
	//for (vector<task*>::iterator t=ktasks.begin(); t!=ktasks.end(); ++t) {
	for ( auto t : ktasks ) {
	  auto coord = t->get_domain(); int tid = coord.coord(0);
	  // lots of task testing
	  CHECK( tid==mytid ); // mpi-dependent
	  vector<shared_ptr<message>> msgs;
	  if (t->has_type_origin()) {
	    CHECK( t->get_n_in_objects()==0 );
	  } else {
	    REQUIRE_NOTHROW( msgs = t->get_send_messages() );
	    CHECK( t->get_n_in_objects()>0 );
	    // the invector is the product of the previous kernel
	    shared_ptr<object> inobject;
	    CHECK_NOTHROW( inobject = t->get_in_object(0) );
	    int origin;
	    REQUIRE_NOTHROW( origin = inobject->get_object_number() );
	    shared_ptr<task> otsk;
	    REQUIRE_NOTHROW( otsk = queue.find_task_by_coordinates(origin,coord) );
	    // msgs can come from one of three neighbours
	    CHECK( msgs.size()==3 );
	    //for (vector<shared_ptr<message>>::iterator m=msgs.begin(); m!=msgs.end(); ++m)
	    for ( auto m : msgs )
	      CHECK( ( m->get_sender().coord(0)==tid
		       || m->get_sender().coord(0)==MOD(tid+1,ntids)
		       || m->get_sender().coord(0)==MOD(tid-1,ntids) ) );
	  }
	  // 4: add task to graph
	  queue.add_task( t );
	  t->find_other_task_by_coordinates =
	    [queue] (int s,processor_coordinate &d) -> shared_ptr<task> {
	    return queue.find_task_by_coordinates(s,d); };
	}
	ik++;
      }
      // kernels have been analyzed;
      REQUIRE_NOTHROW( queue.find_predecessors() );
      auto tasks = queue.get_tasks();
      //for (vector<task*>::iterator t=tasks.begin(); t!=tasks.end(); ++t) {
      for ( auto t : tasks ) {
	auto predecessors = t->get_predecessors();
	if (t->has_type_origin())
	  CHECK( predecessors.size()==0 );
	else {
	  CHECK( predecessors.size()==3 );
	  for ( auto t : predecessors ) {
	    //(vector<task*>::iterator t=ps->begin(); t!=ps->end(); ++t) {
	    int proc;
	    CHECK_NOTHROW( proc = t->get_domain().coord(0) );
	    CHECK( proc==mytid );
	  }
	}
      }
    }

    SECTION( "single analysis call" ) {
      CHECK_NOTHROW( queue.analyze_dependencies() );
    }

    vector<shared_ptr<task>> tsks;
    REQUIRE_NOTHROW( tsks = queue.get_tasks() );
    CHECK( tsks.size()==(nsteps+1) );
    vector<shared_ptr<task>> predecessors;
    //for (auto t=tsks.begin(); t!=tsks.end(); ++t) {
    for ( auto t : tsks ) {
      CHECK_NOTHROW( predecessors = t->get_predecessors() );
      INFO( "step=" << t->get_step() );
      if (t->has_type_origin()) {
	REQUIRE( t->get_receive_messages().size()==0 );
	CHECK( predecessors.size()==0 );
      } else {
	REQUIRE( t->get_receive_messages().size()==3 );
	CHECK( predecessors.size()==3 );
      }
    }

  }

  SECTION( "analysis with optimization" ) {
    section = "optimized"; opt = 1; queue_analysis = 1;
    
    REQUIRE_NOTHROW( queue.analyze_dependencies() );
    REQUIRE_NOTHROW( queue.optimize() );

    vector<shared_ptr<task>> exits;
    REQUIRE_NOTHROW( exits = queue.get_exit_tasks() );
    CHECK( exits.size()==1 ); // that's locally

    vector<shared_ptr<task>> tsks;
    REQUIRE_NOTHROW( tsks = queue.get_tasks() );
    //for (auto t=tsks.begin(); t!=tsks.end(); ++t) {
    for ( auto t : tsks ) {
      INFO( "step=" << t->get_step() );
      if (!t->has_type_origin()) {
	CHECK( t->get_send_messages().size()==0 );
	CHECK( t->get_receive_messages().size()==0 );
      }
      if (t->get_out_object()->get_object_number()==last_object_number) {
	CHECK( t->get_post_messages().size()==0 );
	CHECK( t->get_xpct_messages().size()==0 );
      } else {
	CHECK( t->get_post_messages().size()==3 );
	CHECK( t->get_xpct_messages().size()==3 );
      }
    }
  }
  
  INFO( "queue was " << section );

  CHECK_NOTHROW( queue.execute() );

  /*
   * The rest of this unit test is probably predicated
   * upon queue analysis having been performed
   */  
  if (!queue_analysis) return;
  CHECK( queue.get_all_tasks_executed() );

  // let's inspect some halos
  int found=0;
  //for (auto t=queue.get_tasks().begin(); t!=queue.get_tasks().end(); ++t) {
  for (auto t : queue.get_tasks() ) {
    if (t->get_step()==yvector[0]->get_object_number()) {
      found++;
      shared_ptr<object> h; data_pointer data;
      REQUIRE_NOTHROW( h=t->get_beta_object(0) );
      CHECK( h->volume(mycoord)==(nlocal+2) );
      REQUIRE_NOTHROW( data = h->get_data(mycoord) );
      for (index_int i=0; i<nlocal+2; i++)
	CHECK( data.at(i)==Approx(1.) );
    }
  }
  REQUIRE( found==1 );
  
  {
    for (int s=0; s<nsteps; s++) {
      int i;
      CHECK_NOTHROW( ydata = yvector[s]->get_data(mycoord) );
      for (i=0; i<nlocal; i++) {
	INFO( "step " << s << ", yvalue-" << i << ":" << ydata.at(i) );
	CHECK( ydata.at(i) == Approx( pow(3,s+1) ) );
      }
    }
  }

}

TEST_CASE( "Threepoint queue with gen kernel","[queue][execute][halo][modulo][108]" ) {

  INFO( "mytid=" << mytid );

  int nlocal=17,nsteps=4;
  shared_ptr<distribution> block = 
    shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,-1,nlocal*ntids) );
  auto
    my_first = block->first_index_r(mycoord),
    my_last = block->last_index_r(mycoord);
  CHECK( my_first==mytid*nlocal );
  CHECK( my_last==(mytid+1)*nlocal-1 );

  ioperator no_op("none"), right_shift_mod(">>1"), left_shift_mod("<<1");

  auto xvector = block->new_object(block);
  auto yvector = vector<shared_ptr<object>>(nsteps);
  for (int iv=0; iv<nsteps; iv++) {
    yvector[iv] = block->new_object(block);
  }
  int last_object_number;
  REQUIRE_NOTHROW( last_object_number = yvector[nsteps-1]->get_object_number() );

  mpi_algorithm queue;
  CHECK_NOTHROW( queue = mpi_algorithm(decomp) );
  {
    shared_ptr<kernel> k = shared_ptr<kernel>( new mpi_origin_kernel(xvector) );
    k->set_localexecutefn(  &vecset );
    CHECK_NOTHROW( queue.add_kernel( k ) );
  }
  for (int iv=0; iv<nsteps; iv++) {
    shared_ptr<kernel> k;
    if (iv==0) {
      k = shared_ptr<kernel>( new mpi_kernel(xvector,yvector[0]) );
    } else {
      k = shared_ptr<kernel>( new mpi_kernel(yvector[iv-1],yvector[iv]) );
    }
    k->add_sigma_operator( no_op );
    k->add_sigma_operator( left_shift_mod );
    k->add_sigma_operator( right_shift_mod );
    k->set_localexecutefn(  &threepointsummod );
    CHECK_NOTHROW( queue.add_kernel(k) );
  }
  
  CHECK_NOTHROW( queue.analyze_dependencies() );
  vector<shared_ptr<task>> tsks;
  REQUIRE_NOTHROW( tsks = queue.get_tasks() );
  CHECK( tsks.size()==(nsteps+1) );
  vector<shared_ptr<task>> predecessors;
  //for (vector<task*>::iterator t=tsks.begin(); t!=tsks.end(); ++t) {
  for ( auto t : tsks ) {
    CHECK_NOTHROW( predecessors = t->get_predecessors() );
    if (!t->has_type_origin()) {
      CHECK( predecessors.size()==3 );
    } else {
      CHECK( predecessors.size()==0 );
    }
  }

  SECTION( "jit" ) {
    CHECK_NOTHROW( queue.execute() );
  }

  SECTION( "asap" ) {
    CHECK_NOTHROW( queue.optimize() );
    CHECK_NOTHROW( queue.execute() );
  }
  CHECK( queue.get_all_tasks_executed() );

  {
    data_pointer ydata;
    for (int s=0; s<nsteps; s++) {
      int i;
      CHECK_NOTHROW( ydata = yvector[s]->get_data(mycoord) );
      for (i=0; i<nlocal; i++) {
	INFO( "step " << s << ", yvalue-" << i << ":" << ydata.at(i) );
	CHECK( ydata.at(i) == Approx( pow(3,s+1) ) );
      }
    }
  }
}

TEST_CASE( "Queue optimization (like 106) with arch options","[queue][execute][109]" ) {

  INFO( "mytid=" << mytid );

  int nlocal=7,nsteps=2;
  mpi_architecture *aa; int opt; const char *path;
  REQUIRE_NOTHROW( aa = new mpi_architecture(arch) ); //env->make_architecture() );

  SECTION( "unoptimized" ) { path = "unoptimized";
    opt = 0;
  }
  SECTION( "optimized" ) { path = "optimized";
    opt = 1;
    aa->set_can_message_overlap();
  }
  INFO( fmt::format("Execution path: {}",path) );
  decomposition decomp;
  REQUIRE_NOTHROW( decomp = mpi_decomposition(aa) );
  
  auto block = 
    shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,-1,nlocal*ntids) );
  auto
    my_first = block->first_index_r(mycoord),
    my_last = block->last_index_r(mycoord);
  CHECK( my_first==mycoord_coord*nlocal );
  CHECK( my_last==(mycoord_coord+1)*nlocal-1 );

  ioperator no_op("none"), right_shift_mod(">>1"), left_shift_mod("<<1");

  auto xdata = data_allocate(nlocal);
  data_pointer ydata;
  auto xvector = block->new_object_from_data(xdata);
  auto yvector = vector<shared_ptr<object>>(nsteps);
  xvector->set_name("xorigin");
  for (int i=0; i<nlocal; i++)
    xdata->at(i) = 1.;
  for (int iv=0; iv<nsteps; iv++) {
    yvector[iv] = block->new_object(block);
    yvector[iv]->set_name(fmt::format("yvector{}",iv));
  }
  int last_object_number;
  REQUIRE_NOTHROW( last_object_number = yvector[nsteps-1]->get_object_number() );

  mpi_algorithm queue;
  CHECK_NOTHROW( queue = mpi_algorithm(decomp) );
  {
    shared_ptr<kernel> k = shared_ptr<kernel>( new mpi_origin_kernel(xvector) );
    REQUIRE_NOTHROW( queue.add_kernel(k) );
  }
  for (int iv=0; iv<nsteps; iv++) {
    shared_ptr<kernel> k;
    if (iv==0) {
      k = shared_ptr<kernel>( new mpi_kernel(xvector,yvector[0]) );
    } else {
      k = shared_ptr<kernel>( new mpi_kernel(yvector[iv-1],yvector[iv]) );
    }
    k->set_name(fmt::format("compute-y{}",iv));
    k->add_sigma_operator( no_op );
    k->add_sigma_operator( left_shift_mod );
    k->add_sigma_operator( right_shift_mod );
    k->set_localexecutefn(  &threepointsummod );
    CHECK_NOTHROW( queue.add_kernel(k) );
  }
  
  {
    REQUIRE_NOTHROW( queue.analyze_dependencies() );
    REQUIRE_NOTHROW( queue.optimize() );

    vector<shared_ptr<task>> exits;
    REQUIRE_NOTHROW( exits = queue.get_exit_tasks() );
    CHECK( exits.size()==1 ); // that's locally

    vector<shared_ptr<task>> tsks;
    REQUIRE_NOTHROW( tsks = queue.get_tasks() );
    //for (auto t=tsks.begin(); t!=tsks.end(); ++t) {
    for ( auto t : tsks ) {
      INFO( "step=" << t->get_step() );
      if (!t->has_type_origin()) {
	// these have been cleared in the optimization
	CHECK( t->get_send_messages().size()==0 );
	CHECK( t->get_receive_messages().size()==0 );
      }
      if (t->get_out_object()->get_object_number()==last_object_number) {
	CHECK( t->get_post_messages().size()==0 );
	CHECK( t->get_xpct_messages().size()==0 );
      } else {
	CHECK( t->get_post_messages().size()==3 );
	CHECK( t->get_xpct_messages().size()==3 );
      }
    }
  }
  
  CHECK_NOTHROW( queue.execute() );

  /*
   * The rest of this unit test is probably predicated
   * upon queue analysis having been performed
   */  
  CHECK( queue.get_all_tasks_executed() );

  // let's inspect some halos
  int found=0;
  for ( auto t : queue.get_tasks() ) {
    if (t->get_step()==yvector[0]->get_object_number()) {
      found++;
      shared_ptr<object> h; data_pointer data;
      REQUIRE_NOTHROW( h=t->get_beta_object(0) );
      CHECK( h->volume(mycoord)==(nlocal+2) );
      REQUIRE_NOTHROW( data = h->get_data(mycoord) );
      for (index_int i=0; i<nlocal+2; i++)
	CHECK( data.at(i)==Approx(1.) );
    }
  }
  REQUIRE( found==1 );
  
  {
    for (int s=0; s<nsteps; s++) {
      int i;
      CHECK_NOTHROW( ydata = yvector[s]->get_data(mycoord) );
      for (i=0; i<nlocal; i++) {
	INFO( "step " << s << ", yvalue-" << i << ":" << ydata.at(i) );
	CHECK( ydata.at(i) == Approx( pow(3,s+1) ) );
      }
    }
  }

}

TEST_CASE( "Scale queue analysis with embedding","[queue][execute][embed][110]" ) {

  INFO( "mytid=" << mytid );

  architecture *aa; int power; const char *path;
  REQUIRE_NOTHROW( aa = env->make_architecture() );

  // SECTION( "safe mode" ) { path = "safe";
  //   power = 0;
  // }
  SECTION( "power mode" ) { path = "power";
    power = 1;
    aa->set_power_mode();
  }
  INFO( "using " << path << " mode" );

  mpi_decomposition decomp;
  REQUIRE_NOTHROW( decomp = mpi_decomposition(aa) );
  if (power)
    CHECK( decomp.get_can_embed_in_beta() );
  else
    CHECK( !decomp.get_can_embed_in_beta() );

  int nlocal=22;
  auto no_op = ioperator("none");
  shared_ptr<distribution> block = 
    shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,-1,nlocal*ntids) );
  auto
    my_first = block->first_index_r(mycoord),
    my_last = block->last_index_r(mycoord);
  CHECK( my_first==mycoord_coord*nlocal );
  CHECK( my_last==(mycoord_coord+1)*nlocal-1 );
  
  algorithm queue;
  CHECK_NOTHROW( queue = mpi_algorithm(decomp) );

  data_pointer xdata,ydata;
  auto
    xvector = block->new_object(block),
    yvector = block->new_object(block);
  xvector->set_name("originx"); yvector->set_name("computedy");

  auto gen_kernel = shared_ptr<kernel>( new mpi_kernel(xvector) );
  gen_kernel->set_localexecutefn( &vecset );

  auto mult_kernel = shared_ptr<kernel>( new mpi_kernel(xvector,yvector) );
  mult_kernel->add_sigma_operator( ioperator("none") );
  mult_kernel->set_localexecutefn( &vecscalebytwo );

  CHECK_NOTHROW( queue.add_kernel(gen_kernel) );
  CHECK_NOTHROW( queue.add_kernel(mult_kernel) );
  
  // invectors are embeddable
  if (power)
    CHECK( !xvector->has_data_status_allocated() );
  else
    CHECK( xvector->has_data_status_allocated() );
  CHECK_NOTHROW( queue.analyze_dependencies() );
  CHECK( xvector->has_data_status_allocated() );
  if (power) {
    CHECK( xvector->has_data_status_inherited() );
  }
  
  int nmsgs; REQUIRE_NOTHROW( nmsgs = env->nmessages_sent( env->mode_summarize_entities() ) );
  CHECK_NOTHROW( queue.execute() );
  REQUIRE_NOTHROW( nmsgs = env->nmessages_sent( env->mode_summarize_entities() )-nmsgs );
  if (power==0) // we only count send messages
    CHECK( nmsgs==ntids );
  else
    CHECK( nmsgs==0 );
}

TEST_CASE( "Queue reuse","[queue][reuse][115]" ) {

  int nlocal = 10;
  auto blocked = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,nlocal,-1) );

  for (int nsteps=0; nsteps<4; nsteps++) {
    INFO( "iterations: " << nsteps );

    auto
      x = blocked->new_object(blocked),
      y = blocked->new_object(blocked),
      z = blocked->new_object(blocked);
    x->allocate(); y->allocate(); z->allocate();
    CHECK( x->has_data_status_allocated() );

    double two=2., three=3.;
    algorithm queue = mpi_algorithm(decomp);
    auto
      make = shared_ptr<kernel>( new mpi_origin_kernel(x) ),
      first = shared_ptr<kernel>( new mpi_scale_kernel(two,x,y) ),
      second = shared_ptr<kernel>( new mpi_scale_kernel(three,y,z) );
    shared_ptr<vector<double>> xdata;
    REQUIRE_NOTHROW( xdata = x->get_numa_data_pointer() ); // get_data(mycoord) );
    for (index_int i=0; i<nlocal; i++)
      xdata->at(i) = 1.;

    REQUIRE_NOTHROW( queue.add_kernel( make ) );
    REQUIRE_NOTHROW( queue.add_kernel( first ) );
    REQUIRE_NOTHROW( queue.add_kernel( second ) );

    for (int i=0; i<nsteps; i++) {
      INFO( "it: " << i );

      // objects for this iteration
      shared_ptr<object> xnew,ynew,znew;
      REQUIRE_NOTHROW( xnew = blocked->new_object_from_data(xdata) );
      INFO( "reused xnew status: " << xnew->data_status_as_string() );
      CHECK( xnew->has_data_status_reused() );
      REQUIRE_NOTHROW( ynew = blocked->new_object_from_data(y->get_numa_data_pointer()) );
      REQUIRE_NOTHROW( znew = blocked->new_object_from_data(z->get_numa_data_pointer()) );
      
      // copy data from previous iteration
      shared_ptr<kernel> copy;
      REQUIRE_NOTHROW( copy = shared_ptr<kernel>( new mpi_copy_kernel(z,xnew) ) );
      REQUIRE_NOTHROW( queue.add_kernel(copy) );

      // add kernels on xnew/ynew/znew
      REQUIRE_NOTHROW
	( first = shared_ptr<kernel>( new mpi_scale_kernel(two,xnew,ynew) ) );
      REQUIRE_NOTHROW( queue.add_kernel(first) );

      REQUIRE_NOTHROW
	( second = shared_ptr<kernel>( new mpi_scale_kernel(three,ynew,znew) ) );
      REQUIRE_NOTHROW( queue.add_kernel(second) );

      z = znew;
    }
    REQUIRE_NOTHROW( queue.analyze_dependencies() );
    REQUIRE_NOTHROW( queue.execute() );
    data_pointer zdata;
    REQUIRE_NOTHROW( zdata = z->get_data(mycoord) );
    for (index_int i=0; i<nlocal; i++)
      CHECK( zdata.at(i)==Approx( pow(6,nsteps+1) ) );
  }
}

TEST_CASE( "test embedding by memory counting","[kernel][halo][embed][120]" ) {
  double allocated; REQUIRE_NOTHROW( allocated = env->get_allocated_space() );
  INFO( "allocated at unittest 120 start: " << allocated );
  index_int nlocal = 100, nglobal = ntids*nlocal;
  int do_embed;
  decomposition decomp; shared_ptr<distribution> block;
  shared_ptr<object> xvector,yvector;
  algorithm queue;
  architecture *aa;
  REQUIRE_NOTHROW( aa = env->make_architecture() );

  SECTION( "allocate at once" ) {
    do_embed = 0; }
  SECTION( "allocate later" ) {
    do_embed = 1; }
  INFO( "embedding: " << do_embed );

  REQUIRE_NOTHROW( aa->set_can_embed_in_beta(do_embed) );
  REQUIRE_NOTHROW( decomp = mpi_decomposition(aa) );
  REQUIRE_NOTHROW( block = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,nlocal,-1) ) );
  REQUIRE_NOTHROW( queue = mpi_algorithm(decomp) );
  
  REQUIRE_NOTHROW( xvector = block->new_object(block) );
  if (do_embed==0) {
    CHECK( xvector->has_data_status_allocated() );
    CHECK( env->get_allocated_space()==Approx(allocated+nglobal) );
  } else {
    CHECK( !xvector->has_data_status_allocated() );
    CHECK( env->get_allocated_space()==Approx(allocated) );
  }

  REQUIRE_NOTHROW( yvector = block->new_object(block) );
  REQUIRE_NOTHROW( queue.add_kernel( shared_ptr<kernel>( new mpi_origin_kernel(xvector) ) ) );
  REQUIRE_NOTHROW( queue.add_kernel( shared_ptr<kernel>( new mpi_copy_kernel(xvector,yvector) ) ) );
  REQUIRE_NOTHROW( queue.analyze_dependencies() );
  if (do_embed==0)
    CHECK( env->get_allocated_space()==Approx(allocated+3*nglobal) );
  else
    CHECK( env->get_allocated_space()==Approx(allocated+2*nglobal) );
}

TEST_CASE( "Embedding inputs in halo","[kernel][halo][execute][embed][121]" ) {
  // copied from 39

  architecture *aa; int do_embed; int dim =1;
  REQUIRE_NOTHROW( aa = env->make_architecture() );
  SECTION( "allocate at once" ) {
    do_embed = 0; }
  SECTION( "allocate later" ) {
    do_embed = 1; }
  REQUIRE_NOTHROW( aa->set_can_embed_in_beta(do_embed) );
  REQUIRE_NOTHROW( decomp = mpi_decomposition(aa) );
  INFO( "mytid=" << mytid << "; embedding: " << do_embed );

  int nlocal=10;
  shared_ptr<distribution> block = 
    shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,-1,nlocal*ntids) );
  int nglobal = block->outer_size();
  auto
    xvector = block->new_object(block),
    yvector = block->new_object(block);
  xvector->set_name("xvector"); yvector->set_name("yvector");

  algorithm queue = mpi_algorithm(decomp);
  auto make_x = shared_ptr<kernel>( new mpi_kernel(xvector) );
  REQUIRE_NOTHROW( make_x->set_localexecutefn(&vecsetlinear) );
  REQUIRE_NOTHROW( queue.add_kernel(make_x) );

  auto sum = shared_ptr<kernel>( new mpi_kernel(xvector,yvector) );
  sum->set_name("120sum");
  sum->add_sigma_operator( ioperator("none") );
  sum->add_sigma_operator( ioperator("<=1") );
  sum->add_sigma_operator( ioperator(">=1") );
  sum->set_localexecutefn(  &threepointsumbump );
  REQUIRE_NOTHROW( queue.add_kernel(sum) );
  
  CHECK( ( !arch.get_can_embed_in_beta() || !xvector->has_data_status_allocated() ) );
  CHECK_NOTHROW( queue.analyze_dependencies() );
  CHECK( xvector->has_data_status_allocated() );
  CHECK_NOTHROW( queue.execute() );
  shared_ptr<object> halo;
  data_pointer halo_data,ydata;
  CHECK_NOTHROW( halo = sum->get_beta_object(0) );
  CHECK_NOTHROW( halo_data = halo->get_data(mycoord) );
  CHECK_NOTHROW( ydata = yvector->get_data(mycoord) );
  auto
    my_first = block->first_index_r(mycoord),
    my_last = block->last_index_r(mycoord);
  {
    auto halo_dist = halo->get_distribution();
    if (mytid==0) {
      CHECK( halo_dist->first_index_r(mycoord)==domain_coordinate_zero(dim) );
      CHECK( halo->volume(mycoord)==(nlocal+1) );
      for (int i=0; i<nlocal+1; i++) {
	INFO( "i=" << i << " haloi=" << halo_data.at(i) );
	CHECK( halo_data.at(i) == Approx(pointfunc33(i,my_first[0])) );
      }
    } else if (mytid==ntids-1) {
      CHECK( halo_dist->first_index_r(mycoord)==(my_first-1) );
      CHECK( halo->volume(mycoord)==(nlocal+1) );
      for (int i=0; i<nlocal+1; i++) {
	INFO( "i=" << i << " haloi=" << halo_data.at(i) );
	CHECK( halo_data.at(i) == Approx(pointfunc33(i-1,my_first[0])) );
      }
    } else {
      CHECK( halo_dist->first_index_r(mycoord)==(my_first-1) );
      CHECK( halo->volume(mycoord)==(nlocal+2) );
      for (int i=0; i<nlocal+2; i++) {
	INFO( "i=" << i << " haloi=" << halo_data.at(i) );
	CHECK( halo_data.at(i) == Approx(pointfunc33(i-1,my_first[0])) );
      }
    }
  }
  {
    int i=0;
    if (mytid==0) {
      i = 0; // last+0+1
      CHECK( ydata.at(i)==Approx(pointfunc33(i,my_first[0])+pointfunc33(i+1,my_first[0])) );
      for (i=1; i<nlocal; i++) {
	INFO( "i=" << i << " yi=" << ydata.at(i) );
	CHECK( ydata.at(i)==
	       Approx(pointfunc33(i-1,my_first[0])+pointfunc33(i,my_first[0])+pointfunc33(i+1,my_first[0])) );
      }
    } else if (mytid==ntids-1) {
      for (i=0; i<nlocal-1; i++) {
	INFO( "i=" << i << " yi=" << ydata.at(i) );
	CHECK( ydata.at(i)==
	       Approx(pointfunc33(i-1,my_first[0])+pointfunc33(i,my_first[0])+pointfunc33(i+1,my_first[0])) );
      }
      i = nlocal-1; // last-1+last+0
      CHECK( ydata.at(i) == Approx(pointfunc33(i-1,my_first[0])+pointfunc33(i,my_first[0])) );
    } else {
      for (i=0; i<nlocal-1; i++) {
	INFO( "i=" << i << " yi=" << ydata.at(i) );
	CHECK( ydata.at(i)==
	       Approx(pointfunc33(i-1,my_first[0])+pointfunc33(i,my_first[0])+pointfunc33(i+1,my_first[0])) );
      }
    }
  }
}

TEST_CASE( "queue re-execution","[queue][reexecute][130]" ) {
  int nlocal = 10;
  auto blocked = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,nlocal,-1) );
  auto
    x = blocked->new_object(blocked),
    y = blocked->new_object(blocked);

  algorithm *x2y,*y2x;
  REQUIRE_NOTHROW( x2y = new mpi_algorithm(decomp) );
  REQUIRE_NOTHROW( y2x = new mpi_algorithm(decomp) );

  double one = 1.e0, two = 2.e0;
  REQUIRE_NOTHROW( x2y->add_kernel( shared_ptr<kernel>( new mpi_setconstant_kernel(x,one) ) ) );
  REQUIRE_NOTHROW( x2y->add_kernel( shared_ptr<kernel>( new mpi_scale_kernel(two,x,y) ) ) );
  REQUIRE_NOTHROW( x2y->analyze_dependencies() );

  REQUIRE_NOTHROW( y2x->add_kernel( shared_ptr<kernel>( new mpi_origin_kernel(y) ) ) );
  REQUIRE_NOTHROW( y2x->add_kernel( shared_ptr<kernel>( new mpi_copy_kernel(y,x) ) ) );
  REQUIRE_NOTHROW( y2x->analyze_dependencies() );

  REQUIRE_NOTHROW( x2y->execute() );
  { INFO("check if y is scaled from x");
    data_pointer ydata;
    REQUIRE_NOTHROW( ydata = y->get_data(mycoord) );
    CHECK( ydata.at(0)==Approx(2.) );
  }
  // now copy y back to x; this queue has a vecnoset origin kernel for y
  REQUIRE_NOTHROW( y2x->execute() );
  { INFO("check if x is copied from y");
    data_pointer xdata;
    REQUIRE_NOTHROW( xdata = x->get_data(mycoord) );
    CHECK( xdata.at(0)==Approx(2.) );
  }

  REQUIRE_NOTHROW( x2y->clear_has_been_executed() );
  REQUIRE_NOTHROW( x2y->execute() );
  { INFO("check if y is scaled from x again");
    data_pointer ydata;
    REQUIRE_NOTHROW( ydata = y->get_data(mycoord) );
    CHECK( ydata.at(0)==Approx(4.) );
  }
}

TEST_CASE( "multi-dimensional kernels","[multi][kernel][200]" ) {
  // this initial code comes from distribution[100]
  if (ntids<4) { printf("need at least 4 procs for grid\n"); return; }
  int dim = 2;
  int ntids_i,ntids_j;
  for (int n=sqrt(ntids); n>=1; n--)
    if (ntids%n==0) {
      ntids_i = ntids/n; ntids_j = n; break; }
  if (ntids_i==1 || ntids_j==1) { printf("Could not split processor grid\n"); return; }

  decomposition mdecomp;
  processor_coordinate layout;
  REQUIRE_NOTHROW( layout = arch.get_proc_layout(dim) );
  CHECK( layout[0]==ntids_i );
  CHECK( layout[1]==ntids_j );
  REQUIRE_NOTHROW( mdecomp = mpi_decomposition(arch,layout) );
  CHECK( mdecomp.get_dimensionality()==dim );

  processor_coordinate mycoord;
  REQUIRE_NOTHROW( mycoord = mdecomp.coordinate_from_linear(mytid) );
  int mytid_i = mycoord.coord(0), mytid_j = mycoord.coord(1);
  fmt::print("Two-d mycoord={}\n",mycoord.as_string());
  {
    processor_coordinate chkcoord;
    REQUIRE_NOTHROW( chkcoord = processor_coordinate(mytid,mdecomp) );
    CHECK( chkcoord.coord(0)==mytid_i );
    CHECK( chkcoord.coord(1)==mytid_j );
    int chktid;
    REQUIRE_NOTHROW( chktid = mycoord.linearize(mdecomp) );
    INFO( "mycoord " << mycoord.as_string() <<
	  " linearizes on " << layout.as_string() <<
	  " as " << chktid );    
    CHECK( chktid==mytid );
  }
  int nlocal = 10; index_int g;
  domain_coordinate
    local_size(  vector<index_int>{nlocal+1,nlocal+2} ),
    global_size( vector<index_int>
		 { ntids_i*local_size.coord(0),ntids_j*local_size.coord(1) } );

  shared_ptr<distribution> d;
  REQUIRE_NOTHROW
    ( d = shared_ptr<distribution>( make_shared<mpi_block_distribution>(mdecomp,global_size) ) );
  return;

  shared_ptr<multi_indexstruct> local_domain; // we test this in struct[100]
  REQUIRE_NOTHROW( local_domain = d->get_processor_structure(mycoord) );
  CHECK( local_domain->local_size_r()==local_size );
  // for (int id=0; id<dim; id++)
  //   CHECK( local_domain->local_size_r().coord(id)==nlocal+id+1 );

  shared_ptr<object> o1,r1;
    REQUIRE_NOTHROW( o1 = d->new_object(d) );
  REQUIRE_NOTHROW( r1 = d->new_object(d) );

  shared_ptr<kernel> copy;
  REQUIRE_NOTHROW( copy = shared_ptr<kernel>( new mpi_kernel(o1,r1) ) );
  const char *sigma;
  SECTION( "explicit beta" ) { sigma = "explicit beta";
    REQUIRE_NOTHROW( copy->set_last_dependency().set_explicit_beta_distribution(r1->get_distribution()) );
  }
  printf("restore [200] sigma function\n");
  // SECTION( "sigma function id" ) { sigma = "id function";
  //   REQUIRE_NOTHROW( copy->set_last_dependency().set_signature_function_function
  // 		     ( new multi_sigma_operator
  // 		       ( dim, [] (shared_ptr<multi_indexstruct> i) -> multi_indexstruct* {
  // 			 return i->make_clone(); } ) ) );
  //   //	   ( [] (index_int i) -> indexstruct* { return new contiguous_indexstruct(i); } ) );
  // }
  INFO( "sigma specified by: " << sigma );
  REQUIRE_NOTHROW( copy->set_localexecutefn( &veccopy ) );
  REQUIRE_NOTHROW( copy->analyze_dependencies() );
  {
    shared_ptr<task> tsk; REQUIRE_NOTHROW( tsk = copy->get_tasks().at(0) );
    vector<shared_ptr<message>> msgs;
    REQUIRE_NOTHROW( msgs = tsk->get_receive_messages() );
    CHECK( msgs.size()==1 );
    REQUIRE_NOTHROW( msgs = tsk->get_send_messages() );
    CHECK( msgs.size()==1 );
  }

  REQUIRE_NOTHROW( copy->execute() );

}


TEST_CASE( "multi-dimensional reverse","[multi][kernel][201]" ) {
  // this initial code comes from distribution[100]
  if (ntids<4) { printf("need at least 4 procs for grid\n"); return; }

  int dim = 2;
  processor_coordinate layout;
  REQUIRE_NOTHROW( layout = arch.get_proc_layout(dim) );
  decomposition mdecomp;
  REQUIRE_NOTHROW( mdecomp = mpi_decomposition(arch,layout) );
  processor_coordinate mycoord,flipcoord;
  REQUIRE_NOTHROW( mycoord = mdecomp.coordinate_from_linear(mytid) );
  int
    ntids_i = layout.coord(0), ntids_j = layout.coord(1),
    mytid_i = mycoord.coord(0), mytid_j = mycoord.coord(1);

  
  vector<int> flip_coords
  { layout.coord(0)-mycoord.coord(0)-1,layout.coord(1)-mycoord.coord(1)-1 };
  REQUIRE_NOTHROW( flipcoord = processor_coordinate( flip_coords ) );
  int fliptid; REQUIRE_NOTHROW( fliptid = flipcoord.linearize(layout) );
  INFO( "proc " << mytid << "=" << mycoord.as_string() <<
	"; flipped " << fliptid << "=" << flipcoord.as_string() );
  CHECK( fliptid==(ntids-mytid-1) );

  int nlocal = 10; shared_ptr<distribution> d;
  domain_coordinate domain{ vector<index_int>{ntids_i*(nlocal+1),ntids_j*(nlocal+2)} };
  REQUIRE_NOTHROW( d = shared_ptr<distribution>( make_shared<mpi_block_distribution>(mdecomp,domain) ) );

  shared_ptr<object> o1,r1;
  REQUIRE_NOTHROW( o1 = d->new_object(d) );
  REQUIRE_NOTHROW( r1 = d->new_object(d) );

  // make the structure of the beta distribution
  parallel_structure transstruct;
  REQUIRE_NOTHROW( transstruct = parallel_structure(mdecomp) );
  CHECK( transstruct.get_dimensionality()==dim );
  transstruct.set_name("transposed structure");
  for (int ip=0; ip<ntids; ip++) {
    processor_coordinate pcoord;
    REQUIRE_NOTHROW( pcoord = mdecomp.coordinate_from_linear(ip) );
    processor_coordinate *pflip;
    REQUIRE_NOTHROW( pflip = new processor_coordinate
		     ( vector<int>{layout.coord(0)-pcoord.coord(0)-1,
			 layout.coord(1)-pcoord.coord(1)-1} ) );
    REQUIRE_NOTHROW
      ( transstruct.set_processor_structure(pcoord,d->get_processor_structure(pflip)) );
  }
  REQUIRE_NOTHROW( transstruct.set_structure_type(distribution_type::BLOCKED) );

  INFO( "transdist local: " << transstruct.get_processor_structure(mycoord)->as_string() );
  CHECK( transstruct.has_type_blocked() );
  shared_ptr<distribution> transdist;
  REQUIRE_NOTHROW( transdist = shared_ptr<distribution>( new mpi_distribution(transstruct) ) );
  CHECK( transdist->get_dimensionality()==dim );
  CHECK( transdist->has_type_blocked() );

  shared_ptr<kernel> flip,make;
  data_pointer data;

  REQUIRE_NOTHROW( make = shared_ptr<kernel>( new mpi_origin_kernel(o1) ) );
  REQUIRE_NOTHROW
    ( make->set_localexecutefn
      ( [] (kernel_function_args) -> void {
	vecsetconstant(kernel_function_call,(double)mytid); } ) );
  REQUIRE_NOTHROW( make->analyze_dependencies() );
  REQUIRE_NOTHROW( make->execute() );
  REQUIRE_NOTHROW( data = o1->get_data(mycoord) );
  for (int i=0; i<o1->volume(mycoord); i++)
    CHECK( data.at(i)==Approx((double)mytid) );

  REQUIRE_NOTHROW( flip = shared_ptr<kernel>( new mpi_kernel(o1,r1) ) );
  REQUIRE_NOTHROW( flip->set_last_dependency().set_explicit_beta_distribution(transdist) );
  REQUIRE_NOTHROW( flip->set_localexecutefn( &veccopy ) );
  REQUIRE_NOTHROW( flip->analyze_dependencies() );

  auto tsks = flip->get_tasks(); REQUIRE( tsks.size()==1 );
  auto tsk = tsks.at(0);
  auto msgs = tsk->get_receive_messages();
  CHECK( msgs.size()==1 );
  {
    auto msg = msgs.at(0);
    CHECK( msg->get_receiver()==mycoord );
    INFO( "message from " << msg->get_sender().as_string() <<
	  ", should come from " << flipcoord.as_string() );
    CHECK( msg->get_sender()==flipcoord );
  }
  msgs = tsk->get_send_messages();
  CHECK( msgs.size()==1 );
  {
    auto msg = msgs.at(0);
    CHECK( msg->get_sender()==mycoord );
    INFO( "message to " << msg->get_receiver().as_string() <<
	  ", should go to " << flipcoord.as_string() );
    CHECK( msg->get_receiver()==flipcoord );
  }

  REQUIRE_NOTHROW( flip->execute() );
  { INFO("inspect beta");
    shared_ptr<object> beta; 
    REQUIRE_NOTHROW( beta = flip->get_beta_object(0) );
    REQUIRE_NOTHROW( data = beta->get_data(mycoord) );
    CHECK( data.at(0) ==Approx(fliptid) );
  }
  { INFO("inspect output");
    REQUIRE_NOTHROW( data = r1->get_data(mycoord) );
    for (int i=0; i<o1->volume(mycoord); i++)
      CHECK( data.at(i)==Approx(fliptid) );
  }
}

TEST_CASE( "two-dimensional transpose","[multi][kernel][202]" ) {
  // this initial code comes from distribution[100]
  if (ntids<4) { printf("need at least 4 procs for grid\n"); return; }

  int dim = 2;
  processor_coordinate layout;
  REQUIRE_NOTHROW( layout = arch.get_proc_layout(dim) );
  decomposition mdecomp;
  REQUIRE_NOTHROW( mdecomp = mpi_decomposition(arch,layout) );
  processor_coordinate mycoord,flipcoord;
  REQUIRE_NOTHROW( mycoord = mdecomp.coordinate_from_linear(mytid) );
  int
    ntids_i = layout.coord(0), ntids_j = layout.coord(1),
    mytid_i = mycoord.coord(0), mytid_j = mycoord.coord(1);
  if (ntids_i!=ntids_j) {
    printf("Square processor array for now\n"); return; }
  CHECK( mytid==mytid_j+ntids_j*mytid_i );
  int fliptid = mytid_i+ntids_i*mytid_j;
  REQUIRE_NOTHROW( flipcoord = processor_coordinate( vector<int>{mytid_j,mytid_i} ) );
  INFO( "proc " << mytid << "=" << mycoord.as_string() <<
	"; flipped " << fliptid << "=" << flipcoord.as_string() );

  int nlocal = 10; index_int g;
  vector<index_int> domain_vector;
  g = ntids_i*(nlocal+1); domain_vector.push_back(g);
  g = ntids_j*(nlocal+2); domain_vector.push_back(g);
  domain_coordinate domain{ domain_vector};
  shared_ptr<distribution> d;
  REQUIRE_NOTHROW( d = shared_ptr<distribution>( make_shared<mpi_block_distribution>(mdecomp,domain) ) );

  shared_ptr<object> o1,r1;
  REQUIRE_NOTHROW( o1 = d->new_object(d) );
  REQUIRE_NOTHROW( r1 = d->new_object(d) );

  // make the structure of the beta distribution
  //snippet transdist
  // remove this snippet definition once we have a transpose template
  parallel_structure transstruct;
  REQUIRE_NOTHROW( transstruct = parallel_structure(mdecomp) );
  CHECK( transstruct.get_dimensionality()==dim );
  transstruct.set_name("transposed structure");
  for (int p=0; p<ntids; p++) {
    //for (int p=ntids-1; p>=0; p--) {
    processor_coordinate pcoord;
    REQUIRE_NOTHROW( pcoord = mdecomp.coordinate_from_linear(p) );
    processor_coordinate *pflip;
    REQUIRE_NOTHROW( pflip = new processor_coordinate
		     ( vector<int>{pcoord.coord(1),pcoord.coord(0)} ) );
    // fmt::print("Trans struct in {} comes from {}: {}\n",
    // 	       pcoord->as_string(),pflip->as_string(),
    // 	       d->get_processor_structure(pflip)->as_string());
    REQUIRE_NOTHROW
      ( transstruct.set_processor_structure(pcoord,d->get_processor_structure(pflip)) );
  }
  //snippet end
  REQUIRE_NOTHROW( transstruct.set_structure_type(distribution_type::BLOCKED) );
  printf("Getting transstruct\n");
  for (int ip=0; ip<ntids; ip++) {
    auto ic = mdecomp.coordinate_from_linear(ip);
    auto str = transstruct.get_processor_structure(ic);
    //    fmt::print("{} struct @{} : {}\n",ic->as_string(),(long)str,str->as_string());
  }
  INFO( "transdist local: " << transstruct.get_processor_structure(mycoord)->as_string() );

  CHECK( transstruct.has_type_blocked() );
  shared_ptr<distribution> transdist;
  REQUIRE_NOTHROW( transdist = shared_ptr<distribution>( new mpi_distribution(transstruct) ) );
  CHECK( transdist->get_dimensionality()==dim );
  CHECK( transdist->has_type_blocked() );
  return;
  shared_ptr<kernel> flip,make;
  data_pointer data;

  REQUIRE_NOTHROW( make = shared_ptr<kernel>( new mpi_origin_kernel(o1) ) );
  REQUIRE_NOTHROW
    ( make->set_localexecutefn
      ( [] (kernel_function_args) -> void {
	vecsetconstant(kernel_function_call,(double)mytid); } ) );
  REQUIRE_NOTHROW( make->analyze_dependencies() );
  REQUIRE_NOTHROW( make->execute() );
  REQUIRE_NOTHROW( data = o1->get_data(mycoord) );
  for (int i=0; i<o1->volume(mycoord); i++)
    CHECK( data.at(i)==Approx((double)mytid) );

  REQUIRE_NOTHROW( flip = shared_ptr<kernel>( new mpi_kernel(o1,r1) ) );
  REQUIRE_NOTHROW( flip->set_last_dependency().set_explicit_beta_distribution(transdist) );
  REQUIRE_NOTHROW( flip->set_localexecutefn( &veccopy ) );
  REQUIRE_NOTHROW( flip->analyze_dependencies() );

  auto tsks = flip->get_tasks(); REQUIRE( tsks.size()==1 );
  auto tsk = tsks.at(0); auto msgs = tsk->get_receive_messages(); CHECK( msgs.size()==1 );
  {
    auto msg = msgs.at(0);
    CHECK( msg->get_receiver()==mycoord );
    INFO( "message from " << msg->get_sender().as_string() );
    CHECK( msg->get_sender()==flipcoord );
  }

  REQUIRE_NOTHROW( flip->execute() );
  REQUIRE_NOTHROW( data = r1->get_data(mycoord) );
  for (int i=0; i<o1->volume(mycoord); i++)
    CHECK( data.at(i)==Approx(fliptid) );
}

TEST_CASE( "any-d function","[multi][stencil][204]" ) {
  SECTION( "1d" ) {
    domain_coordinate
      size( vector<index_int>{5} ),
      offset( vector<index_int>{0} ),
      x( vector<index_int>{0} );
    // first with zero offset
    CHECK( INDEXanyD(x,offset,size,1)==0 );
    REQUIRE_NOTHROW( x.at(0) = 3 );
    CHECK( INDEXanyD(x,offset,size,1)==3 );
    // nonzero offset
    REQUIRE_NOTHROW( offset.at(0) = 1 );
    REQUIRE_NOTHROW( x.at(0) = 1 );
    CHECK( INDEXanyD(x,offset,size,1)==0 );
    REQUIRE_NOTHROW( x.at(0) = 4 );
    CHECK( INDEXanyD(x,offset,size,1)==3 );
  }
  SECTION( "2d" ) {
    domain_coordinate
      size( vector<index_int>{5,10} ),
      offset( vector<index_int>{0,0} ),
      x( vector<index_int>{0,0} );
    // first with zero offset
    REQUIRE_NOTHROW( x.at(0)=1 );
    REQUIRE_NOTHROW( x.at(1)=3 );
    CHECK( INDEXanyD(x,offset,size,2)==13 );
    REQUIRE_NOTHROW( offset.at(0)=1 );
    REQUIRE_NOTHROW( offset.at(1)=2 );
    CHECK( INDEXanyD(x,offset,size,2)==1 );
  }
}

TEST_CASE( "high dimensional stencil","[multi][stencil][205]" ) {
  int dim;
  SECTION( "dim 2" ) {
    dim = 2;
  }

  INFO( fmt::format("Stencil test in {} dimensions",dim) );
  processor_coordinate layout;
  REQUIRE_NOTHROW( layout = arch.get_proc_layout(dim) );
  mpi_decomposition decomp;
  REQUIRE_NOTHROW( decomp = mpi_decomposition(arch,layout) );
  INFO( fmt::format("domains decomposition: {}",decomp.as_string()) );
  processor_coordinate mycoord;
  REQUIRE_NOTHROW( mycoord = decomp.coordinate_from_linear(mytid) );

  // define a stencil operator with a +1 in every direction
  stencil_operator diff_operator(dim);
  for (int id=0; id<dim; id++) {
    domain_coordinate leg = domain_coordinate_zero(dim);
    leg.at(id) = 1;
    REQUIRE_NOTHROW( diff_operator.add(leg) );
    leg.at(id) = -1;
    REQUIRE_NOTHROW( diff_operator.add(leg) );
  }

  // make multi-d objects
  index_int nglobal = 5*ntids;
  shared_ptr<distribution> multi_domain;
  vector<index_int> size_vector(dim);
  domain_coordinate gsize_vector(2);
  for (int id=0; id<dim; id++)
    REQUIRE_NOTHROW( gsize_vector.at(id) = nglobal );
  REQUIRE_NOTHROW( multi_domain = shared_ptr<distribution>
		   ( make_shared<mpi_block_distribution>(decomp,gsize_vector) ) );
  
  shared_ptr<object> ingrid,outgrid;
  REQUIRE_NOTHROW( ingrid = multi_domain->new_object(multi_domain) );
  REQUIRE_NOTHROW( outgrid = multi_domain->new_object(multi_domain) );
  CHECK( ingrid->volume(mycoord)==Approx( pow(nglobal,dim)/ntids ) );

  shared_ptr<kernel> apply;
  REQUIRE_NOTHROW( apply  = shared_ptr<kernel>( new mpi_kernel(ingrid,outgrid) ) );
  REQUIRE_NOTHROW( apply->set_last_dependency().add_sigma_stencil(diff_operator) );
  REQUIRE_NOTHROW( apply->set_localexecutefn( &central_difference_anyd ) );
  REQUIRE_NOTHROW( apply->analyze_dependencies() );
  shared_ptr<object> halo;
  REQUIRE_NOTHROW( halo = apply->get_beta_object(0) );
  CHECK( halo->get_enclosing_structure()->volume()==Approx( pow(nglobal,dim) ) );
  REQUIRE_NOTHROW( apply->execute() );
}

TEST_CASE( "column rotating","[multi][kernel][220]" ) {
  int dim = 2;
  processor_coordinate layout;
  REQUIRE_NOTHROW( layout = arch.get_proc_layout(dim) );
  mpi_decomposition mdecomp;
  REQUIRE_NOTHROW( mdecomp = mpi_decomposition(arch,layout) );
  processor_coordinate mycoord,*rotcoord;
  REQUIRE_NOTHROW( mycoord = mdecomp.coordinate_from_linear(mytid) );
  int
    ntids_i = layout.coord(0), ntids_j = layout.coord(1),
    mytid_i = mycoord.coord(0), mytid_j = mycoord.coord(1);
  if (ntids_i!=ntids_j) {
    printf("\nNeed square processor array for Cannon\n\n"); return; }
  CHECK( mytid==mytid_j+ntids_j*mytid_i );

  // rotate the columns down: column j down by j
  int new_i = (mytid_i+mytid_j)%ntids_i;
  int rottid = new_i*ntids_j + mytid_j;
  REQUIRE_NOTHROW( rotcoord = new processor_coordinate( vector<int>{new_i,mytid_j} ) );
  INFO( "proc " << mytid << "=" << mycoord.as_string() <<
	"; rotated " << rottid << "=" << rotcoord->as_string() );

  int nlocal = 10;
  vector<index_int> domain_vector{ ntids_i*nlocal,ntids_j*nlocal };
  shared_ptr<distribution> d;
  REQUIRE_NOTHROW( d = shared_ptr<distribution>
		   ( make_shared<mpi_block_distribution>(mdecomp,domain_coordinate(domain_vector) ) ) );
  shared_ptr<object> o1,r1;
  REQUIRE_NOTHROW( o1 = d->new_object(d) );
  REQUIRE_NOTHROW( r1 = d->new_object(d) );
  shared_ptr<kernel> setconstantp = shared_ptr<kernel>( new mpi_origin_kernel(o1) );
  setconstantp->set_localexecutefn(vecsetconstantp);
  REQUIRE_NOTHROW( setconstantp->analyze_dependencies() );
  REQUIRE_NOTHROW( setconstantp->execute() );
  {
    data_pointer data;
    REQUIRE_NOTHROW( data = o1->get_data(mycoord) );
    CHECK( data.at(0)==Approx(mytid) );
  }

  SECTION( "flip processor coordinates" ) {
    // beta says p_i,p_j comes from p_j,p_i
    parallel_structure transstruct;
    REQUIRE_NOTHROW( transstruct = parallel_structure(mdecomp) );
    CHECK( transstruct.get_dimensionality()==dim );
    transstruct.set_name("transposed structure");
    for (int p=0; p<ntids; p++) {
      //for (int p=ntids-1; p>=0; p--) {
      processor_coordinate pcoord;
      REQUIRE_NOTHROW( pcoord = mdecomp.coordinate_from_linear(p) );
      processor_coordinate *pflip;
      REQUIRE_NOTHROW( pflip = new processor_coordinate
		       ( vector<int>{pcoord.coord(1),pcoord.coord(0)} ) );
      REQUIRE_NOTHROW
	( transstruct.set_processor_structure(pcoord,d->get_processor_structure(pflip)) );
    }
    REQUIRE_NOTHROW( transstruct.set_structure_type(distribution_type::BLOCKED) );

    shared_ptr<distribution> transbeta;
    auto rotate = shared_ptr<kernel>( new mpi_kernel(o1,r1) );
    REQUIRE_NOTHROW( transbeta = shared_ptr<distribution>( new mpi_distribution(transstruct) ) );
    REQUIRE_NOTHROW( rotate->set_explicit_beta_distribution(transbeta) );    
    REQUIRE_NOTHROW( rotate->set_localexecutefn( &veccopy ) );
    return;
    {
      data_pointer data;
      REQUIRE_NOTHROW( data = r1->get_data(mycoord) );
      int mytid_i = mytid/ntids_j, mytid_j = mytid%ntids_j;
      int
	newtid_i = (mytid_i+mytid_j)%ntids_i,
	//rottid = mytid_j + mytid_i*ntids_j;
	rottid = mytid_j + newtid_i * ntids_j;
      CHECK( data.at(0)==Approx(rottid) );
    }
  }
  SECTION( "rotate i" ) {
    // beta says p_i,p_j comes from p_i-1,p_j
    parallel_structure transstruct;
    REQUIRE_NOTHROW( transstruct = parallel_structure(mdecomp) );
    CHECK( transstruct.get_dimensionality()==dim );
    transstruct.set_name("transposed structure");
    for (int p=0; p<ntids; p++) {
      //for (int p=ntids-1; p>=0; p--) {
      processor_coordinate pcoord;
      REQUIRE_NOTHROW( pcoord = mdecomp.coordinate_from_linear(p) );
      processor_coordinate pflip = pcoord.rotate( vector<int>{1,0},layout );
      REQUIRE_NOTHROW
	( transstruct.set_processor_structure(pcoord,d->get_processor_structure(pflip)) );
    }
    REQUIRE_NOTHROW( transstruct.set_structure_type(distribution_type::BLOCKED) );

    shared_ptr<distribution> transbeta;
    auto rotate = shared_ptr<kernel>( new mpi_kernel(o1,r1) );
    REQUIRE_NOTHROW( transbeta = shared_ptr<distribution>( new mpi_distribution(transstruct) ) );
    REQUIRE_NOTHROW( rotate->set_explicit_beta_distribution(transbeta) );    
    REQUIRE_NOTHROW( rotate->set_localexecutefn( &veccopy ) );
    return;
    {
      data_pointer data;
      REQUIRE_NOTHROW( data = r1->get_data(mycoord) );
      int mytid_i = mytid/ntids_j, mytid_j = mytid%ntids_j;
      int
	newtid_i = (mytid_i+1)%ntids_i,
	rottid = newtid_i * ntids_j + mytid_j;
      CHECK( data.at(0)==Approx(rottid) );
    }
  }
  SECTION( "rotate j" ) {
    // beta says p_i,p_j comes from p_i-1,p_j
    parallel_structure transstruct;
    REQUIRE_NOTHROW( transstruct = parallel_structure(mdecomp) );
    CHECK( transstruct.get_dimensionality()==dim );
    transstruct.set_name("transposed structure");
    for (int p=0; p<ntids; p++) {
      //for (int p=ntids-1; p>=0; p--) {
      processor_coordinate pcoord;
      REQUIRE_NOTHROW( pcoord = mdecomp.coordinate_from_linear(p) );
      processor_coordinate pflip = pcoord.rotate( vector<int>{0,1},layout );
      REQUIRE_NOTHROW
	( transstruct.set_processor_structure(pcoord,d->get_processor_structure(pflip)) );
    }
    REQUIRE_NOTHROW( transstruct.set_structure_type(distribution_type::BLOCKED) );

    shared_ptr<distribution> transbeta;
    auto rotate = shared_ptr<kernel>( new mpi_kernel(o1,r1) );
    REQUIRE_NOTHROW( transbeta = shared_ptr<distribution>( new mpi_distribution(transstruct) ) );
    REQUIRE_NOTHROW( rotate->set_explicit_beta_distribution(transbeta) );    
    REQUIRE_NOTHROW( rotate->set_localexecutefn( &veccopy ) );
    return;
    {
      data_pointer data;
      REQUIRE_NOTHROW( data = r1->get_data(mycoord) );
      int mytid_i = mytid/ntids_j, mytid_j = mytid%ntids_j;
      int
	newtid_j = (mytid_j+1)%ntids_j,
	rottid = newtid_j * ntids_i + mytid_i;
      CHECK( data.at(0)==Approx(rottid) );
    }
  }
}

#if 0

TEST_CASE( "binned distributions","[distribution][bin][300]" ) {
  auto blocked = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,10*ntids) );
  auto input = blocked->new_object(blocked);
  auto make_input = shared_ptr<kernel>( new mpi_origin_kernel(input) );
  make_input->set_localexecutefn( &vecsetlinear );
  make_input->analyze_dependencies();
  make_input->execute();
  shared_ptr<distribution> binned;
  REQUIRE_NOTHROW( binned = shared_ptr<distribution>( new mpi_binned_distribution(decomp,input) ) );
}

#endif
