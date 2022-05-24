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
 **** unit tests for the LULESH proxy application
 ****
 ****************************************************************/

#include <stdlib.h>
#include <math.h>

#include "catch.hpp"

#include "mpi_base.h"
#include "mpi_ops.h"
#include "mpi_static_vars.h"
#include "unittest_functions.h"
#include "imp_functions.h"
#include "lulesh_functions.h"

using std::vector;
using std::function;
using std::make_shared;
using std::shared_ptr;

TEST_CASE( "testing the support functions, pointwise","[01]" ) {
  domain_coordinate c; domain_coordinate d;
  SECTION( "one" ) {
    SECTION( "even" ) {
      REQUIRE_NOTHROW( c = domain_coordinate( vector<index_int>{6} ) );
    }
    SECTION( "odd" ) {
      REQUIRE_NOTHROW( c = domain_coordinate( vector<index_int>{7} ) );
    }
    REQUIRE_NOTHROW( d = signature_coordinate_element_to_local(c) );
    CHECK( d.get_dimensionality()==1 );
    CHECK( d.coord(0)==3 );
  }
  SECTION( "three" ) {
    REQUIRE_NOTHROW( c = domain_coordinate( vector<index_int>{6,7,8} ) );
    REQUIRE_NOTHROW( d = signature_coordinate_element_to_local(c) );
    CHECK( d.get_dimensionality()==3 );
    CHECK( d.coord(0)==3 );
    CHECK( d.coord(1)==3 );
    CHECK( d.coord(2)==4 );
  }
}

TEST_CASE( "testing support for global to local mapping","[03]" ) {
  // for each local point we need to know the global point it comes from
  for (int dim=1; dim<=3; dim++) {
    INFO( "testing dim=" << dim );
    auto local_multi = multi_indexstruct(dim);
    shared_ptr<multi_indexstruct> global_multi;
    for (int idim=0; idim<dim; idim++) {
      auto local_struct =
	shared_ptr<indexstruct>( new contiguous_indexstruct( 5+idim,10+2*idim ) );
      REQUIRE_NOTHROW( local_multi.set_component(idim,local_struct) );
    }
    REQUIRE_NOTHROW( global_multi = signature_global_node_to_local(local_multi) );
    INFO( fmt::format("local: {}\nglobal: {}",
		      local_multi.as_string(),global_multi->as_string() ) );
    domain_coordinate
      gfirst = global_multi->first_index_r(), lfirst = local_multi.first_index_r();
    INFO( "testing first" );
    for (int idim=0; idim<dim; idim++) {
      index_int g = gfirst[idim], l = lfirst[idim];
      CHECK( g==(l+1)/2 );
      // if (l%2==0)
      // 	CHECK( g==l/2 );
      // else
      // 	CHECK( g==(l+1)/2 );
    }
  }
}

TEST_CASE( "support functions two-d","[multi][20]" ) {
  if (ntids!=4 ) { printf("lulesh [20] needs square number\n"); return; }

  INFO( "mytid=" << mytid );
  int dim = 2;

  // two-d decomposition
  auto layout = arch.get_proc_layout(dim);
  decomposition decomp = mpi_decomposition(arch,layout);
  CHECK( decomp.get_dimensionality()==dim );
  processor_coordinate mycoord = decomp.coordinate_from_linear(mytid);
  INFO( "mycoord=" << mycoord.as_string() );
  
  // distributions and objects
  shared_ptr<distribution> elements_dist; shared_ptr<object> elements;

  index_int local_nelements = 10;
  vector<index_int> elements_domain;
  for (int id=0; id<dim; id++)
    elements_domain.push_back( local_nelements*(layout.coord(id)) );
  domain_coordinate domain_size( elements_domain );
  REQUIRE_NOTHROW( elements_dist = shared_ptr<distribution>
		   ( make_shared<mpi_block_distribution>(decomp,domain_size) ) );
  CHECK( elements_dist->global_volume()==local_nelements*local_nelements*ntids );
  REQUIRE_NOTHROW( elements = shared_ptr<object>( new mpi_object(elements_dist) ) );
  elements->set_name("elements2D");

  shared_ptr<kernel> init_elements;
  REQUIRE_NOTHROW( init_elements = shared_ptr<kernel>( new mpi_origin_kernel(elements) ) );
  REQUIRE_NOTHROW( init_elements->set_localexecutefn( &vecsetlinear2d ) );
  REQUIRE_NOTHROW( init_elements->analyze_dependencies() );
  REQUIRE_NOTHROW( init_elements->execute() );
}

TEST_CASE( "elements to local nodes two-d","[multi][21]" ) {
  INFO( "mytid=" << mytid );
  int dim;

  const char *path;
  SECTION( "2D" ) { 
    dim = 2; path = "2d";
  }
  // SECTION( "3D" ) { 
  //   dim = 3; path = "3d";
  // }
  INFO( "dimensionality: " << path );

  // two-d decomposition
  auto layout = arch.get_proc_layout(dim);
  decomposition decomp = mpi_decomposition(arch,layout);
  processor_coordinate mycoord = decomp.coordinate_from_linear(mytid);
  INFO( "mycoord=" << mycoord.as_string() );
  
  // distributions and objects
  shared_ptr<distribution> elements_dist, local_nodes_dist;
  shared_ptr<object> elements,local_nodes;

  index_int local_nelements = 10;
  vector<index_int> elements_domain;
  for (int id=0; id<dim; id++)
    elements_domain.push_back( local_nelements*(layout.coord(id)) );
  domain_coordinate domain_size( elements_domain );
  REQUIRE_NOTHROW( elements_dist = shared_ptr<distribution>
		   ( make_shared<mpi_block_distribution>(decomp,domain_size) ) );
  {
    index_int gv=ntids; for (int id=0; id<dim; id++) gv *= local_nelements;
    CHECK( elements_dist->global_volume()==gv );
  }
  REQUIRE_NOTHROW( elements = shared_ptr<object>( new mpi_object(elements_dist) ) );
  elements->set_name("elements2D");
  {
    domain_coordinate first_element(dim);
    REQUIRE_NOTHROW( first_element = elements_dist->first_index_r(mycoord) );
    INFO( "proc " << mytid << "=" << mycoord.as_string() <<
	  " has first element " << first_element.as_string() );
  }

  shared_ptr<kernel> init_elements,init_local_nodes;
  REQUIRE_NOTHROW( init_elements = shared_ptr<kernel>( new mpi_origin_kernel(elements) ) );
  REQUIRE_NOTHROW( init_elements->set_localexecutefn( &vecsetlinear ) );
  REQUIRE_NOTHROW( init_elements->analyze_dependencies() );
  REQUIRE_NOTHROW( init_elements->execute() );

  data_pointer data; index_int len;
  REQUIRE_NOTHROW( data = elements->get_data(mycoord) );
  REQUIRE_NOTHROW( len = elements->volume(mycoord) );
  {
    index_int myfirst;
    REQUIRE_NOTHROW( myfirst = elements_dist->first_index_r(mycoord).linear_location_in
		     (elements->get_distribution()->global_last_index()) );
    CHECK( data.at(0)==Approx(myfirst) );
  }

  // local nodes: on each side twice the number of elements
  vector<index_int> local_nodes_domain;
  for (int id=0; id<dim; id++ )
    local_nodes_domain.push_back( 2*elements_domain[id] );
  domain_coordinate local_nodes_size(local_nodes_domain);
  REQUIRE_NOTHROW( local_nodes_dist = shared_ptr<distribution>
		   ( make_shared<mpi_block_distribution>(decomp,local_nodes_size) ) );
  REQUIRE_NOTHROW( local_nodes = shared_ptr<object>( new mpi_object(local_nodes_dist) ) );
  local_nodes->set_name("local_nodes");
  // both distributions are conforming
  CHECK( local_nodes->volume(mycoord)==(1<<dim)*elements->volume(mycoord) );

  // element -> local node is a local replication
  shared_ptr<kernel> element_to_local_nodes;
  REQUIRE_NOTHROW( element_to_local_nodes = shared_ptr<kernel>( new mpi_kernel(elements,local_nodes) ) );
  element_to_local_nodes->set_name("element_to_local_nodes");
  // sigma operator from coordinate-to-coordinate mapping
  REQUIRE_NOTHROW
    ( element_to_local_nodes->set_last_dependency().set_signature_function_function
      ( multi_sigma_operator
	( dim,
	  function< shared_ptr<multi_indexstruct>(const multi_indexstruct&) >{
	  [] (const multi_indexstruct&i) -> shared_ptr<multi_indexstruct> {
	    return signature_struct_element_to_local(i); } } ) ) );
  REQUIRE_NOTHROW( element_to_local_nodes->set_localexecutefn
		   ( [dim] ( kernel_function_args )  -> void {
		     return element_to_local_function( kernel_function_call,dim); } ) );
  REQUIRE_NOTHROW( element_to_local_nodes->analyze_dependencies() );

  // inspect the halo
  shared_ptr<object> beta;
  REQUIRE_NOTHROW( beta = element_to_local_nodes->last_dependency().get_beta_object() );
  for (int id=0; id<dim; id++) {
    index_int firstnode,firstbeta;
    REQUIRE_NOTHROW( firstnode = local_nodes->get_distribution()->first_index_r(mycoord).coord(id) );
    REQUIRE_NOTHROW( firstbeta = beta->get_distribution()->first_index_r(mycoord).coord(id) );
    // an element has 4 nodes, but only 2 in each direction
    CHECK( firstnode==2*firstbeta );
  }

  { // there should be just one message, and that is local
    shared_ptr<task> local_task;
    REQUIRE_NOTHROW( local_task = element_to_local_nodes->get_tasks().at(0) );
    vector<shared_ptr<message>> msgs;
    REQUIRE_NOTHROW( msgs = local_task->get_receive_messages() );
    REQUIRE( msgs.size()==1 );
    shared_ptr<message> msg;
    REQUIRE_NOTHROW( msg = msgs.at(0) );
    CHECK( msg->get_sender()==mycoord );
    CHECK( msg->get_receiver()==mycoord );
  }
  REQUIRE_NOTHROW( element_to_local_nodes->execute() );

  {
    // after execution inspect the halo data
    domain_coordinate
      first_node = local_nodes->get_distribution()->first_index_r(mycoord),
      elt_from_node = first_node.operate( ioperator("/2") );
    data_pointer hdata;
    REQUIRE_NOTHROW( hdata = beta->get_data(mycoord) );
    CHECK( hdata.at(0)==Approx( elements->get_distribution()->linearize(elt_from_node) ) );
  }

  { // node data should be 4-fold replication
    data_pointer ldata; REQUIRE_NOTHROW( ldata = local_nodes->get_data(mycoord) );
    index_int ndata = local_nodes->volume(mycoord);
    for (int id=0; id<dim; id++) {
      index_int first_element = local_nodes->get_distribution()->first_index_r(mycoord).coord(id)/2;
      CHECK( first_element==elements->get_distribution()->first_index_r(mycoord).coord(id) );
    }
    index_int
      first_element = elements->get_distribution()->linearize
          ( elements->get_distribution()->first_index_r(mycoord) ),
      inodes = local_nodes->get_distribution()->local_size_r(mycoord).coord(0);
    CHECK( ldata.at(0)==Approx(first_element) );
    // for (index_int i=0; i<ndata; i++) {
    //   INFO( "element " << i );
    //   index_int ei = (i/inodes)/2, ej = i/2;
    //   CHECK( ldata.at(i)==Approx(first_element+i/4) );
    // }
  }
}

TEST_CASE( "local to global nodes 2d","[multi][22]" ) {
  if (arch.get_over_factor()>1) {
    printf("For now let's not overdecompose\n"); return; }
  if (arch.nprocs()<4) {
    printf("Need at least 4 procs for two-d lulesh\n"); return; }
  
  INFO( "mytid=" << mytid );
  int dim = 2;

  // two-d decomposition
  auto layout = arch.get_proc_layout(dim);
  decomposition decomp = mpi_decomposition(arch,layout);
  processor_coordinate mycoord = decomp.coordinate_from_linear(mytid);
  INFO( "mycoord=" << mycoord.as_string() );
  
  // distributions and objects
  shared_ptr<distribution> local_nodes_dist, global_nodes_dist;
  shared_ptr<object> local_nodes,global_nodes;
  
  // local nodes: on each side twice the number of elements
  index_int local_nelements = 10;
  vector<index_int> local_nodes_domain;
  for (int id=0; id<dim; id++ )
    local_nodes_domain.push_back( 2*local_nelements*layout.coord(id) );
  domain_coordinate local_nodes_size(local_nodes_domain);
  REQUIRE_NOTHROW( local_nodes_dist = shared_ptr<distribution>
		   ( make_shared<mpi_block_distribution>(decomp,local_nodes_size) ) );
  REQUIRE_NOTHROW( local_nodes = shared_ptr<object>( new mpi_object(local_nodes_dist) ) );
  CHECK( local_nodes->get_dimensionality()==dim );
  local_nodes->set_name("local_nodes");

  // 1 node per element in each direction, except the last which gets 2
  vector<index_int> global_nodes_domain;
  for (int id=0; id<dim; id++)
    global_nodes_domain.push_back( local_nelements*layout.coord(id)+1 );
  domain_coordinate global_nodes_size(global_nodes_domain);
  REQUIRE_NOTHROW( global_nodes_dist = shared_ptr<distribution>
		   ( make_shared<mpi_block_distribution>(decomp,global_nodes_size) ) );
  REQUIRE_NOTHROW( global_nodes = shared_ptr<object>( new mpi_object(global_nodes_dist) ) );
  CHECK( global_nodes->get_dimensionality()==dim );
  global_nodes->set_name("global_nodes");

  // initialize the elements
  shared_ptr<kernel> init_local_nodes; // VLE replace by setconstant_kernel
  REQUIRE_NOTHROW( init_local_nodes = shared_ptr<kernel>( new mpi_origin_kernel(local_nodes) ) );
  REQUIRE_NOTHROW( init_local_nodes->set_localexecutefn( &vecsetconstantone ) );
  REQUIRE_NOTHROW( init_local_nodes->analyze_dependencies() );
  REQUIRE_NOTHROW( init_local_nodes->execute() );
  {
    data_pointer data;
    REQUIRE_NOTHROW( data = local_nodes->get_data(mycoord) );
    CHECK( data.at(0)==Approx(1.) );
    index_int g = local_nodes->volume(mycoord);
    CHECK( data.at(g-1)==Approx(1.) );
  }

  // local node summing to global is fun
  shared_ptr<kernel> local_to_global_nodes;
  REQUIRE_NOTHROW( local_to_global_nodes = shared_ptr<kernel>( new mpi_kernel(local_nodes,global_nodes) ) );
  domain_coordinate g = global_nodes->get_distribution()->global_last_index();
  REQUIRE_NOTHROW
    (
  //snippet n2t4lastelement
     local_to_global_nodes->set_last_dependency().set_signature_function_function
     ( multi_sigma_operator
       ( dim,
	 function< shared_ptr<multi_indexstruct>(const multi_indexstruct &g) >{
	   [local_nodes] (const multi_indexstruct &g) -> shared_ptr<multi_indexstruct> {
	     auto enc = local_nodes->get_enclosing_structure();
	     return signature_local_from_global( g,*(enc.get()) ); } } ) )
  //snippet end
     );
  REQUIRE_NOTHROW
    ( local_to_global_nodes->set_localexecutefn
      ( [local_nodes] ( kernel_function_args ) -> void {
	auto enclosure = local_nodes->get_enclosing_structure();
	local_to_global_function( kernel_function_call,*(enclosure.get())); } ) );
  REQUIRE_NOTHROW( local_to_global_nodes->add_trace_level(trace_level::MESSAGE) );
  CHECK( local_to_global_nodes->has_trace_level(trace_level::MESSAGE) );
  REQUIRE_NOTHROW( local_to_global_nodes->analyze_dependencies() );
  REQUIRE_NOTHROW( local_to_global_nodes->execute() );

  // VLE need to check the output....
  {
    shared_ptr<object> beta;
    REQUIRE_NOTHROW( beta = local_to_global_nodes->get_beta_object(0) );
    data_pointer data,hdata;
    REQUIRE_NOTHROW( data = global_nodes->get_data(mycoord) );
    REQUIRE_NOTHROW( hdata = beta->get_data(mycoord) );
    CHECK( hdata.at(0)>0 );
    CHECK( hdata.at(1)>0 );
    domain_coordinate nlocal(dim),f(dim),l(dim);
    REQUIRE_NOTHROW( nlocal = global_nodes->get_distribution()->local_size_r(mycoord) );
    if (dim==2) {
      if ( mycoord.coord(0)==0 && mycoord.coord(1)==0 )
	CHECK( data.at(0)==Approx(1.) );
      else if ( mycoord.coord(0)==0 && mycoord.coord(1)>0 )
	CHECK( data.at(0)==Approx(2.) );
      else if ( mycoord.coord(0)>0 && mycoord.coord(1)==0 )
	CHECK( data.at(0)==Approx(2.) );
      else
	CHECK( data.at(0)==Approx(4.) );
    } else printf("Can only test data with dim=2\n");
  }
}

TEST_CASE( "global back to local nodes","[multi][23]" ) {
  if (arch.get_over_factor()>1) {
    printf("For now let's not overdecompose\n"); return; }
  if (arch.nprocs()<4) {
    printf("Need at least 4 procs for two-d lulesh\n"); return; }
  
  INFO( "mytid=" << mytid );
  int dim = 2;

  // two-d decomposition
  auto layout = arch.get_proc_layout(dim);
  decomposition decomp = mpi_decomposition(arch,layout);
  processor_coordinate mycoord = decomp.coordinate_from_linear(mytid);
  INFO( "mycoord=" << mycoord.as_string() );
  
  // distributions and objects
  shared_ptr<distribution> local_nodes_dist, global_nodes_dist;
  shared_ptr<object> local_nodes,global_nodes;
  
  // local nodes: in each direction two elements per processor
  index_int local_nelements = 2;
  vector<index_int> local_nodes_domain;
  for (int id=0; id<dim; id++ ) // two local nodes per element
    local_nodes_domain.push_back( 2*local_nelements*layout.coord(id) );
  domain_coordinate local_nodes_size(local_nodes_domain);
  REQUIRE_NOTHROW( local_nodes_dist = shared_ptr<distribution>
		   ( make_shared<mpi_block_distribution>(decomp,local_nodes_size) ) );
  REQUIRE_NOTHROW( local_nodes = shared_ptr<object>( new mpi_object(local_nodes_dist) ) );
  local_nodes->set_name("local_nodes");

  // 1 node per element in each direction, except the last which gets 2
  vector<index_int> global_nodes_domain;
  for (int id=0; id<dim; id++)
    global_nodes_domain.push_back( local_nelements*layout.coord(id)+1 );
  domain_coordinate global_nodes_size(global_nodes_domain);
  REQUIRE_NOTHROW( global_nodes_dist = shared_ptr<distribution>
		   ( make_shared<mpi_block_distribution>(decomp,global_nodes_size) ) );
  REQUIRE_NOTHROW( global_nodes = shared_ptr<object>( new mpi_object(global_nodes_dist) ) );
  global_nodes->set_name("global_nodes");

  // initialize the global nodes
  shared_ptr<kernel> init_global_nodes;
  REQUIRE_NOTHROW( init_global_nodes = shared_ptr<kernel>( new mpi_origin_kernel(global_nodes) ) );
  REQUIRE_NOTHROW( init_global_nodes->set_localexecutefn( &vecsetlinear ) );
  REQUIRE_NOTHROW( init_global_nodes->analyze_dependencies() );
  REQUIRE_NOTHROW( init_global_nodes->execute() );

  // global nodes are distributed to local nodes
  shared_ptr<kernel> global_to_local_nodes;
  REQUIRE_NOTHROW( global_to_local_nodes =
		   shared_ptr<kernel>( new mpi_kernel(global_nodes,local_nodes) ) );
  domain_coordinate g = global_nodes->get_distribution()->global_last_index();
  REQUIRE_NOTHROW
    ( global_to_local_nodes->set_last_dependency().set_signature_function_function
      ( multi_sigma_operator
	(dim,
	 function< shared_ptr<multi_indexstruct>( const multi_indexstruct & ) >{
	  [] ( const multi_indexstruct &i ) -> shared_ptr<multi_indexstruct>{
	    return signature_global_node_to_local(i); } } ) ) );
  auto local_nodes_global_domain = local_nodes->get_enclosing_structure(),
    global_nodes_global_domain = global_nodes->get_distribution()->get_enclosing_structure();
  REQUIRE_NOTHROW
    ( global_to_local_nodes->set_localexecutefn
      ( [local_nodes_global_domain] ( kernel_function_args ) -> void {
	function_global_node_to_local
	  ( kernel_function_call,*(local_nodes_global_domain.get())); } ) );
  REQUIRE_NOTHROW( global_to_local_nodes->analyze_dependencies() );

  shared_ptr<task> gl_task;
  REQUIRE_NOTHROW( gl_task = global_to_local_nodes->get_tasks().at(0) );
  shared_ptr<distribution> gl_beta;
  REQUIRE_NOTHROW( gl_beta = gl_task->last_dependency().get_beta_distribution() );
  shared_ptr<multi_indexstruct> gl_beta_struct;
  REQUIRE_NOTHROW( gl_beta_struct = gl_beta->get_processor_structure(mycoord) );
  CHECK( gl_beta_struct->volume()>0 );
  CHECK( global_nodes_global_domain->contains(gl_beta_struct) );
  INFO( "sending " << global_nodes->get_distribution()->get_processor_structure(mycoord)->as_string()
	<< "\n -> " << gl_beta_struct->as_string()
	<< "\n -> " << local_nodes->get_distribution()->get_processor_structure(mycoord)->as_string() );
  auto msgs = gl_task->get_receive_messages();
  REQUIRE_NOTHROW( global_to_local_nodes->execute() );

  // VLE need to check the output....
  {
    data_pointer data;
    REQUIRE_NOTHROW( data = global_nodes->get_data(mycoord) );
    domain_coordinate nlocal(dim),f(dim),l(dim);
    REQUIRE_NOTHROW( nlocal = global_nodes->get_distribution()->local_size_r(mycoord) );
    if (dim==2) {
      for ( auto i=f.coord(0); i<=l.coord(0); i++ ) {
	for ( auto j=f.coord(1); j<=l.coord(1); j++ ) {
	  index_int
	    loc = (i-f.coord(0)) * nlocal[1] + (j-f[1]), gloc = i * nlocal[1] + j;
	  CHECK( data.at(loc)>=gloc );
	}
      }
    } else printf("Can only test data with dim=2\n");
  }

}

TEST_CASE( "local nodes back to elements","[lulesh][24]" ) {
  INFO( "mytid=" << mytid );
  int dim = 2;

  // two-d decomposition
  auto layout = arch.get_proc_layout(dim);
  decomposition decomp = mpi_decomposition(arch,layout);
  processor_coordinate mycoord = decomp.coordinate_from_linear(mytid);
  INFO( "mycoord=" << mycoord.as_string() );
  
  // distributions and objects
  shared_ptr<distribution> elements_dist, local_nodes_dist;
  shared_ptr<object> elements,local_nodes;

  // elements
  index_int local_nelements = 10;
  vector<index_int> elements_domain;
  for (int id=0; id<dim; id++)
    elements_domain.push_back( local_nelements*(layout.coord(id)) );
  domain_coordinate elements_size(elements_domain);
  REQUIRE_NOTHROW( elements_dist = shared_ptr<distribution>
		   ( make_shared<mpi_block_distribution>(decomp,elements_size) ) );
  CHECK( elements_dist->global_volume()==local_nelements*local_nelements*ntids );
  REQUIRE_NOTHROW( elements = shared_ptr<object>( new mpi_object(elements_dist) ) );
  elements->set_name("elements2D");

  // local nodes: on each side twice the number of elements
  vector<index_int> local_nodes_domain;
  for (int id=0; id<dim; id++ )
    local_nodes_domain.push_back( 2*elements_domain[id] );
  domain_coordinate local_nodes_size(local_nodes_domain);
  REQUIRE_NOTHROW( local_nodes_dist = shared_ptr<distribution>
		   ( make_shared<mpi_block_distribution>(decomp,local_nodes_size) ) );
  REQUIRE_NOTHROW( local_nodes = shared_ptr<object>( new mpi_object(local_nodes_dist) ) );
  local_nodes->set_name("local_nodes");
  // both distributions are conforming
  CHECK( local_nodes->volume(mycoord)==(1<<dim)*elements->volume(mycoord) );

  shared_ptr<kernel> init_local_nodes;
  REQUIRE_NOTHROW( init_local_nodes = shared_ptr<kernel>( new mpi_origin_kernel(local_nodes) ) );
  REQUIRE_NOTHROW( init_local_nodes->set_localexecutefn( &vecsetlinear ) );
  REQUIRE_NOTHROW( init_local_nodes->analyze_dependencies() );
  REQUIRE_NOTHROW( init_local_nodes->execute() );

  // check local nodes values
  data_pointer data; index_int len;
  REQUIRE_NOTHROW( data = local_nodes->get_data(mycoord) );
  REQUIRE_NOTHROW( len = local_nodes->volume(mycoord) );
  {
    INFO( "local nodes global structure: "
	  << local_nodes->get_enclosing_structure()->as_string() );
    { INFO( "first index: " << local_nodes_dist->first_index_r(mycoord).as_string() );
      CHECK( data.at(0)==Approx
	     ( local_nodes_dist->first_index_r(mycoord)
	       .linear_location_in(local_nodes->get_enclosing_structure()) ) );
    }
    { INFO( "last index: " << local_nodes_dist->last_index_r(mycoord).as_string() );
      CHECK( data.at(len-1)==Approx
	     ( local_nodes_dist->last_index_r(mycoord)
	       .linear_location_in(local_nodes->get_enclosing_structure()) ) );
    }
  }

  // element -> local node is a local replication
  shared_ptr<kernel> local_nodes_to_element;
  REQUIRE_NOTHROW( local_nodes_to_element = shared_ptr<kernel>( new mpi_kernel(local_nodes,elements) ) );
  local_nodes_to_element->set_name("local_nodes_to_element");
  REQUIRE_NOTHROW
    ( local_nodes_to_element->set_last_dependency().set_signature_function_function
      ( multi_sigma_operator
	( dim,
	  function< shared_ptr<multi_indexstruct>(const multi_indexstruct&)>{
	    [dim] (const multi_indexstruct &n) -> shared_ptr<multi_indexstruct> {
	      return signature_local_to_element( dim,n ); } } ) ) );
  REQUIRE_NOTHROW( local_nodes_to_element->set_localexecutefn
		   (&local_node_to_element_function) );
  REQUIRE_NOTHROW( local_nodes_to_element->analyze_dependencies() );
  REQUIRE_NOTHROW( local_nodes_to_element->execute() );

  // inspect element values
  {
    data_pointer edata;
    REQUIRE_NOTHROW( edata = elements->get_data(mycoord) );
    domain_coordinate
      efirst = elements_dist->first_index_r(mycoord),
      elast = elements_dist->last_index_r(mycoord);
    domain_coordinate
      offsets = elements_dist->get_numa_structure()->first_index_r()
          - elements_dist->get_enclosing_structure()->first_index_r(),
      nsizes = elements_dist->get_numa_structure()->local_size_r();
    index_int loc = 0;
    for (index_int i=efirst[0]; i<=elast[0]; i++) {
      for (index_int j=efirst[1]; j<=elast[1]; j++) {
	index_int
	  nodeline = local_nodes_dist->global_size().coord(1),
	  node     = 2*i*nodeline + 2*j;
	INFO( "element (" << i << "," << j <<
	      "), top left node: " << node );
	// local nodes = 40x40, so on proc 0
	// linear values are 0,1,40,41 
	// but stored in 0,1,20,21
	CHECK( edata.at(INDEX2D(i,j,offsets,nsizes))==
	       Approx( (
			node + node+1 + node+nodeline + node+nodeline+1
			) / 4.
		       )
	       );
      }
    }
  }
}

#if 0
TEST_CASE( "distributions two-d","[multi][29]" ) {
  printf("Test 29 aborted\n"); return;

  if (arch.get_over_factor()>1) {
    printf("For now let's not overdecompose\n"); return; }
  if (arch.nprocs()<4) {
    printf("Need at least 4 procs for two-d lulesh\n"); return; }
  
  INFO( "mytid=" << mytid );
  int dim = 2;

  // two-d decomposition
  auto layout = arch.get_proc_layout(dim);
  decomposition decomp = mpi_decomposition(arch,layout);
  CHECK( decomp.get_dimensionality()==dim );
  processor_coordinate mycoord = decomp.coordinate_from_linear(mytid);
  INFO( "mycoord=" << mycoord.as_string() );
  
  // distributions and objects
  shared_ptr<distribution> elements_dist, local_nodes_dist, global_nodes_dist;
  shared_ptr<object> elements,local_nodes,global_nodes;
  
  // number of elements per side; input parameter
  index_int local_nelements = env->iargument("elements",100);
  vector<index_int> elements_domain;
  for (int id=0; id<dim; id++)
    elements_domain.push_back( local_nelements*(layout.coord(id)) );
  domain_coordinate elements_size(elements_domain);
  REQUIRE_NOTHROW( elements_dist = shared_ptr<distribution>
		   ( make_shared<mpi_block_distribution>(decomp,elements_size) ) );
  CHECK( elements_dist->get_dimensionality()==dim );
  CHECK( elements_dist->global_volume()==local_nelements*local_nelements*ntids );
  REQUIRE_NOTHROW( elements = shared_ptr<object>( new mpi_object(elements_dist) ) );
  elements->set_name("elements2D");
  {
    domain_coordinate first_element(2);
    REQUIRE_NOTHROW( first_element = elements->first_index_r(mycoord) );
    INFO( "proc " << mytid << "=" << mycoord.as_string() <<
	  " has first element " << first_element.as_string() );
    //REQUIRE( 1==0 );
  }
  // domain_coordinate elements_layout(dim);
  // REQUIRE_NOTHROW( elements_layout = elements->global_last_index() );

  // local nodes: on each side twice the number of elements
  vector<index_int> local_nodes_domain;
  for (int id=0; id<dim; id++ )
    local_nodes_domain.push_back( 2*elements_domain[id] );
  domain_coordinate local_nodes_size(local_nodes_domain);
  REQUIRE_NOTHROW( local_nodes_dist = shared_ptr<distribution>
		   ( make_shared<mpi_block_distribution>(decomp,local_nodes_size) ) );
  REQUIRE_NOTHROW( local_nodes = shared_ptr<object>( new mpi_object(local_nodes_dist) ) );
  local_nodes->set_name("local_nodes");
  // both distributions are conforming
  CHECK( local_nodes->volume(mycoord)==(1<<dim)*elements->volume(mycoord) );

  // 1 node per element in each direction, except the last which gets 2
  vector<index_int> global_nodes_domain;
  for (int id=0; id<dim; id++) {
    global_nodes_domain.push_back( elements_domain[id]+1 );
  }
  domain_coordinate global_nodes_size(global_nodes_domain);
  REQUIRE_NOTHROW( global_nodes_dist = shared_ptr<distribution>
		   ( make_shared<mpi_block_distribution>(decomp,global_nodes_size) ) );
  // one node per element, plus extra for the sides, plus one for the corne
  CHECK( global_nodes_dist->global_volume()==elements->global_volume()
	 +elements->global_size(0)+elements->global_size(1)+1 );
  REQUIRE_NOTHROW( global_nodes = shared_ptr<object>( new mpi_object(global_nodes_dist) ) );
  global_nodes->set_name("global_nodes");

  // initialize the elements
  shared_ptr<kernel> init_elements,init_local_nodes,init_global_nodes;
  REQUIRE_NOTHROW( init_elements = shared_ptr<kernel>( new mpi_origin_kernel(elements) ) );
  REQUIRE_NOTHROW( init_local_nodes = shared_ptr<kernel>( new mpi_origin_kernel(local_nodes) ) );
  REQUIRE_NOTHROW( init_global_nodes = shared_ptr<kernel>( new mpi_origin_kernel(global_nodes) ) );

  REQUIRE_NOTHROW( init_elements->set_localexecutefn( &vecsetlinear ) );
  REQUIRE_NOTHROW( init_local_nodes->set_localexecutefn( &vecsetlinear ) );
  REQUIRE_NOTHROW( init_global_nodes->set_localexecutefn( &vecsetlinear ) );

  INFO( "mycoord=" << mycoord.as_string() );

  REQUIRE_NOTHROW( init_elements->analyze_dependencies() );
  REQUIRE_NOTHROW( init_local_nodes->analyze_dependencies() );
  REQUIRE_NOTHROW( init_global_nodes->analyze_dependencies() );

  REQUIRE_NOTHROW( init_elements->execute() );
  REQUIRE_NOTHROW( init_local_nodes->execute() );
  REQUIRE_NOTHROW( init_global_nodes->execute() );

  data_pointer data; index_int len;
  REQUIRE_NOTHROW( data = elements->get_data(mycoord) );
  REQUIRE_NOTHROW( len = elements->volume(mycoord) );
  CHECK( data.at(0)==Approx
	 ( elements->first_index_r(mycoord).linear_location_in(elements->global_last_index()) ) );
  CHECK( data.at(len-1)==Approx
     ( elements->first_index_r(mycoord).linear_location_in(elements->global_last_index())+len-1 ) );

  REQUIRE_NOTHROW( data = local_nodes->get_data(mycoord) );
  REQUIRE_NOTHROW( len = local_nodes->volume(mycoord) );
  CHECK( data.at(0)==Approx
     ( local_nodes->first_index_r(mycoord).linear_location_in(local_nodes->global_last_index()) ) );
  CHECK( data.at(len-1)==Approx
     ( local_nodes->first_index_r(mycoord).linear_location_in(local_nodes->global_last_index())+len-1 ) );

  // element -> local node is a local replication
  shared_ptr<kernel> element_to_local_nodes;
  REQUIRE_NOTHROW( element_to_local_nodes = shared_ptr<kernel>( new mpi_kernel(elements,local_nodes) ) );
  element_to_local_nodes->set_name("element_to_local_nodes");
  // sigma operator from coordinate-to-coordinate mapping
  REQUIRE_NOTHROW( element_to_local_nodes->set_last_dependency().set_signature_function_function
		   ( multi_sigma_operator( 2, &signature_coordinate_element_to_local ) ) );
  REQUIRE_NOTHROW( element_to_local_nodes->set_localexecutefn
    ( [dim] ( kernel_function_args ) -> void {
      return element_to_local_function( kernel_function_call,dim ); } ) );
  REQUIRE_NOTHROW( element_to_local_nodes->set_localexecutectx(&dim) );
  REQUIRE_NOTHROW( element_to_local_nodes->analyze_dependencies() );

  // inspect the halo
  shared_ptr<object> beta;
  REQUIRE_NOTHROW( beta = element_to_local_nodes->last_dependency().get_beta_object() );
  for (int id=0; id<dim; id++) {
    index_int firstnode,firstbeta;
    REQUIRE_NOTHROW( firstnode = local_nodes->first_index_r(mycoord).coord(id) );
    REQUIRE_NOTHROW( firstbeta = beta->first_index_r(mycoord).coord(id) );
    // an element has 4 nodes, but only 2 in each direction
    CHECK( firstnode==2*firstbeta );
  }

  { // there should be just one message, and that is local
    shared_ptr<task> local_task;
    REQUIRE_NOTHROW( local_task = element_to_local_nodes->get_tasks().at(0) );
    vector<shared_ptr<message>> msgs;
    REQUIRE_NOTHROW( msgs = local_task->get_receive_messages() );
    REQUIRE( msgs.size()==1 );
    shared_ptr<message> msg;
    REQUIRE_NOTHROW( msg = msgs.at(0) );
    CHECK( msg->get_sender()==mycoord );
    CHECK( msg->get_receiver()==mycoord );
  }
  REQUIRE_NOTHROW( element_to_local_nodes->execute() );

  {
    // after execution inspect the halo data
    domain_coordinate
      first_node = local_nodes->first_index_r(mycoord),
      elt_from_node = first_node.operate( ioperator("/2") );
    data_pointer hdata;
    REQUIRE_NOTHROW( hdata = beta->get_data(mycoord) );
    CHECK( hdata.at(0)==Approx( elements->linearize(elt_from_node) ) );
  }

}
#endif
