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
 **** unit tests for tree stuff
 ****
 ****************************************************************/

#include <stdlib.h>
#include <math.h>

#include "catch.hpp"

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

index_int dup(int p,index_int i) {
  return 2*(p/2)+i;
}

index_int hlf(int p,index_int i) {
  return p/2;
}

TEST_CASE( "multistage tree collecting, recursive","[distribution][redundant][73]" ) {

  if (ntids%4!=0 ) {
    printf("Skipping example 73\n"); return; }

  // start with a bottom distribution of two points per proc
  shared_ptr<distribution> twoper = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,2,-1) );
  auto bot = twoper->new_object(twoper);
  bot->allocate();
  bot->set_name("bottom object");
  {
    CHECK( bot->volume(mycoord)==2 );
    data_pointer botdata = bot->get_data(mycoord);
    botdata.at(0) = botdata.at(1) = 1.;
  }

  // mid has one point per proc, redund is two-way redundant
  auto div2 = ioperator("/2");
  shared_ptr<distribution> unique,redund;
  shared_ptr<object> mid,top;

  REQUIRE_NOTHROW( unique = twoper->operate( div2 ) );
  REQUIRE_NOTHROW( mid = unique->new_object(unique) );
  mid->set_name("mid object");
  CHECK( mid->volume(mycoord)==1 );
  CHECK( mid->first_index_r(mycoord)==mycoord_coord );

  REQUIRE_NOTHROW( redund = unique->operate( div2 ) );
  REQUIRE_NOTHROW( top = redund->new_object(redund) );
  top->set_name("top object");
  CHECK( top->volume(mycoord)==1 );
  CHECK( top->first_index_r(mycoord)==(mycoord_coord/2) );
  CHECK( top->last_index_r(mycoord)==top->first_index_r(mycoord) );

  vector<shared_ptr<task>> tsks;
  shared_ptr<task> tsk; shared_ptr<message> msg;

  // gathering from bot to mid should be local
  shared_ptr<kernel> gather1;
  REQUIRE_NOTHROW( gather1 = shared_ptr<kernel>( new mpi_kernel(bot,mid) ) );
  gather1->set_localexecutefn(  &scansum );
  REQUIRE_NOTHROW( gather1->set_last_dependency().set_signature_function_function
		   ( [] (index_int i) -> shared_ptr<indexstruct> {
		     return doubleinterval(i); } ) );
  gather1->set_name("gather-local");
  CHECK_NOTHROW( gather1->analyze_dependencies() );
  REQUIRE_NOTHROW( tsks = gather1->get_tasks() );
  CHECK( tsks.size()==1 );
  REQUIRE_NOTHROW( tsk = tsks.at(0) );
  REQUIRE_NOTHROW( tsk->get_receive_messages() );
  CHECK( tsk->get_receive_messages().size()==1 );
  REQUIRE_NOTHROW( msg = tsk->get_receive_messages().at(0) );
  CHECK( msg->get_sender()==mycoord );
  CHECK( msg->get_receiver()==mycoord );
  REQUIRE_NOTHROW( gather1->execute() );
  {
    data_pointer data; REQUIRE_NOTHROW( data = mid->get_data(mycoord) );
    CHECK( data.at(0)==2. );
  }


  shared_ptr<kernel> gather2;
  REQUIRE_NOTHROW( gather2 = shared_ptr<kernel>( new mpi_kernel(mid,top) ) );
  gather2->set_localexecutefn(  &scansum );
  REQUIRE_NOTHROW( gather2->set_last_dependency().set_signature_function_function
		   ( [] (index_int i) -> shared_ptr<indexstruct> {
		     return doubleinterval(i); } ) );
  gather2->set_name("gather-to-redundant");
  REQUIRE_NOTHROW( gather2->analyze_dependencies() );
  REQUIRE_NOTHROW( tsks = gather2->get_tasks() );
  CHECK( tsks.size()==1 );
  REQUIRE_NOTHROW( tsk = tsks.at(0) );
  REQUIRE_NOTHROW( tsk->get_receive_messages().size() );
  CHECK( tsk->get_receive_messages().size()==2 );
  for (int imsg=0; imsg<2; imsg++) {
    REQUIRE_NOTHROW( msg = tsk->get_receive_messages().at(imsg) );
    if (mytid%2==0) {
      CHECK( (msg->get_sender()==mycoord || msg->get_sender()==mycoord+1) );
    } else {
      CHECK( (msg->get_sender()==mycoord || msg->get_sender()==mycoord-1) );
    }
  }
  REQUIRE_NOTHROW( gather2->execute() );
  {
    auto data = top->get_data(mycoord); 
    CHECK( data.at(0)==4. );
  }

}

TEST_CASE( "multistage tree collecting iterated","[distribution][redundant][74]" ) {

  if ((ntids%4)!=0) { printf("Test [74] requires multiple of 4\n"); return; }

  int points_per_proc = 4;
  index_int gsize = points_per_proc*ntids;
  int twos,nlevels;
  vector<int> levels;
  for (nlevels=1,twos=1; twos<=gsize; nlevels++,twos*=2)
    levels.push_back(twos);
  twos /= 2; nlevels -= 1;
  REQUIRE( twos==gsize ); // perfect bisection only
  CHECK( nlevels==levels.size() );

  // create the distributions and objects
  int tsize = gsize, itest=0;
  //snippet dividedistributions
  auto div2 = ioperator("/2");
  shared_ptr<distribution> distributions[nlevels];
  shared_ptr<object> objects[nlevels];
  for (int nlevel=0; nlevel<nlevels; nlevel++) {
    if (nlevel==0) {
      distributions[0]
	= shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,points_per_proc,-1) );
    } else {
      distributions[nlevel] = distributions[nlevel-1]->operate(div2);
    }
    INFO( "level: " << nlevel << "; g=" << distributions[nlevel]->outer_size() );
    objects[nlevel] = distributions[nlevel]->new_object(distributions[nlevel]);
    REQUIRE_NOTHROW( objects[nlevel]->allocate() );
    //snippet end
    if (nlevel>0) {
      auto cur = objects[nlevel],prv = objects[nlevel-1];
      index_int csize = cur->volume(mycoord),psize = prv->volume(mycoord);
      if (itest==0) // test for the trivial levels:
	CHECK( (2*csize)==psize ); // doing local combines
      else { // test for the redundant levels
	INFO( "(redundant level)" );
	CHECK( csize==1 ); // one point, not necessarily unique
      }
      if (csize==1) itest++;
    }
    if (nlevel<nlevels-1)
      tsize /= 2;
  }
  CHECK( tsize==1 );

  // create kernels
  //snippet dividekernels
  vector<shared_ptr<kernel>> kernels(nlevels-1);
  for (int nlevel=0; nlevel<nlevels-1; nlevel++) {
    INFO( "level: " << nlevel );
    char name[20];
    sprintf(name,"gather-%d",nlevel);
    kernels[nlevel] = shared_ptr<kernel>( new mpi_kernel(objects[nlevel],objects[nlevel+1]) );
    kernels[nlevel]->set_name( name );
    kernels[nlevel]->set_last_dependency().set_signature_function_function
      ( [] (index_int i) -> shared_ptr<indexstruct> {
	return doubleinterval(i); } );
    kernels[nlevel]->set_localexecutefn(  &scansum );
  }
  //snippet end

  // does this work?
  data_pointer data;
  int n;
  CHECK_NOTHROW( n = objects[0]->volume(mycoord) );
  CHECK( n==points_per_proc );
  CHECK_NOTHROW( data = objects[0]->get_data(mycoord) );
  for (int i=0; i<n; i++) 
    data.at(i) = 1.;
  int should=1.;
  for (int nlevel=0; nlevel<nlevels-1; nlevel++) {
    INFO( "level: " << nlevel );
    CHECK_NOTHROW( kernels[nlevel]->analyze_dependencies(true) );
    CHECK_NOTHROW( kernels[nlevel]->execute() );
    should *= 2;
    CHECK_NOTHROW( n = objects[nlevel+1]->volume(mycoord) );
    CHECK_NOTHROW( data = objects[nlevel+1]->get_data(mycoord) );
    for (int i=0; i<n; i++) {
      CHECK( data.at(i) == Approx(should) );
    }
  }
}

TEST_CASE( "multistage tree collecting iterated, irregular","[distribution][redundant][75]" ) {

  int nlocal = 2*mytid+1, dim=1;
  index_int n2 = ntids*ntids;
  index_int gsize = n2;

  shared_ptr<distribution> bottom_level,cur_level;
  vector<index_int> blocksizes;
  for (int tid=0; tid<ntids; tid++)
    blocksizes.push_back( 2*tid+1 );
  auto pidx = make_shared<parallel_indexstruct>(blocksizes);
  REQUIRE_NOTHROW
    ( bottom_level = shared_ptr<distribution>( new mpi_distribution(decomp,pidx) ) );
  CHECK( bottom_level->volume(mycoord)==nlocal );

  vector<shared_ptr<distribution>> levels; int nlevels;
  vector<shared_ptr<object>> objects;
  cur_level = bottom_level;
  auto div2 = ioperator("/2");
  memory_buffer w; format_to(w,"Bottom level: {}\n",bottom_level->as_string());
  for (nlevels=1; nlevels<2*ntids; nlevels++) {
    levels.push_back( cur_level );
    REQUIRE_NOTHROW( objects.push_back( cur_level->new_object(cur_level) ) );
    index_int lsize;
    REQUIRE_NOTHROW( lsize = cur_level->outer_size() );
    if (lsize==1) break;
    cur_level = cur_level->operate(div2);
    format_to(w,"Level {}: {}\n",nlevels,cur_level->as_string());
  }
  INFO( to_string(w) );
  CHECK( nlevels==levels.size() );

  mpi_algorithm *queue = new mpi_algorithm(decomp);
  {
    shared_ptr<kernel> make = shared_ptr<kernel>( new mpi_origin_kernel( objects[0] ) );
    make->set_localexecutefn( &vecsetlinear );
    queue->add_kernel(make);
  }
  for (int level=0; level<nlevels-1; level++) {
    shared_ptr<kernel> step;
    REQUIRE_NOTHROW( step = shared_ptr<kernel>( new mpi_kernel(objects[level],objects[level+1]) ) );
    step->set_name( fmt::format("coarsen-level-{}",level) );
    step->set_last_dependency().set_signature_function_function
      ( [] (index_int i) -> shared_ptr<indexstruct> {
	return doubleinterval(i); } );
    step->set_localexecutefn(  &scansum );
    REQUIRE_NOTHROW( queue->add_kernel(step) );
  }
  
  REQUIRE_NOTHROW( queue->analyze_dependencies() );
  REQUIRE_NOTHROW( queue->execute() );
  auto top = objects[nlevels-1];
  data_pointer data;
  CHECK( top->first_index_r(mycoord)==domain_coordinate_zero(dim) );
  CHECK( top->volume(mycoord)==1 );
  REQUIRE_NOTHROW( data = top->get_data(mycoord) );
  CHECK( data.at(0)==Approx( n2*(n2-1.)/2 ) );
}

