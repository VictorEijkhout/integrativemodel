/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2023
 ****
 **** Unit tests for the MPI product backend of IMP
 **** based on the CATCH framework (https://github.com/philsquared/Catch)
 ****
 **** unit tests for mpi-based distributions
 ****
 ****************************************************************/

#include <stdlib.h>
#include <math.h>

#include "catch2/catch_all.hpp"

#include "mpi_distribution.h"

using fmt::format, fmt::print;

using std::make_shared, std::shared_ptr;
using std::string;
using std::vector;

auto &the_env = mpi_environment::instance();

TEST_CASE( "creation","[mpi][distribution][01]" ) {
  INFO( "proc: " << the_env.procid() );
  {
    INFO( "1D" );
    coordinate<index_int,1> omega( 10*the_env.nprocs() );
    mpi_decomposition<1> procs( the_env );
    REQUIRE_NOTHROW( mpi_distribution<1>( omega,procs ) );
    mpi_distribution<1> d1( omega,procs );
    mpi_distribution<1> d2( omega,procs );
    REQUIRE( d1.compatible_with(d1) );
    REQUIRE( not d1.compatible_with(d2) );
    REQUIRE( not d2.compatible_with(d1) );
  }
  {
    INFO( "2D" );
    coordinate<index_int,2> omega( 10*the_env.nprocs() );
    mpi_decomposition<2> procs( the_env );
    REQUIRE_NOTHROW( mpi_distribution<2>( omega,procs ) );
  }
}

TEST_CASE( "global domains","[mpi][distribution][02]" ) {
  INFO( "proc: " << the_env.procid() );
  {
    INFO( "1D" );
    const int points_per_proc = ipower(10,1);
    index_int total_points = points_per_proc*the_env.nprocs();
    coordinate<index_int,1> omega( total_points );
    mpi_decomposition<1> procs( the_env );
    mpi_distribution<1> omega_p( omega,procs );
    REQUIRE_NOTHROW( omega_p.global_domain() );
    indexstructure<index_int,1> global_domain = omega_p.global_domain();
    INFO( "global domain: " << global_domain.as_string() );
    REQUIRE_NOTHROW( global_domain.volume() );
    index_int check_total_points = global_domain.volume();
    REQUIRE( check_total_points==total_points );
  }
  {
    INFO( "2D" );
    mpi_decomposition<2> procs( the_env );
    INFO( "Decomposition: " << procs.as_string() );

    coordinate<index_int,2> omega( procs.domain_layout()*16 /* total_points */ );
    index_int total_points = omega.span();
    mpi_distribution<2> omega_p( omega,procs );
    INFO( "domain: " << omega_p.global_domain().as_string() );

    REQUIRE_NOTHROW( omega_p.global_domain() );
    indexstructure<index_int,2> global_domain = omega_p.global_domain();
    INFO( "global domain: " << global_domain.as_string() );
    REQUIRE_NOTHROW( global_domain.volume() );
    index_int check_total_points = global_domain.volume();
     REQUIRE( check_total_points==total_points );
  }
}

TEST_CASE( "local domains","[mpi][distribution][03]" ) {
  INFO( "proc: " << the_env.procid() );
  {
    INFO( "1D" );
    const int points_per_proc = ipower(10,1);
    index_int total_points = points_per_proc*the_env.nprocs();
    coordinate<index_int,1> omega( total_points );
    mpi_decomposition<1> procs( the_env );
    mpi_distribution<1> omega_p( omega,procs );
    REQUIRE_NOTHROW( omega_p.local_domain() );
    indexstructure<index_int,1> local_domain = omega_p.local_domain();
    index_int check_total_points = the_env.allreduce_ii( local_domain.volume() );
    REQUIRE( check_total_points==total_points );
  }
  {
    INFO( "2D" );
    const int points_per_proc = pow(10,2);
    index_int total_points = points_per_proc*the_env.nprocs();
    INFO( "points/proc=" << points_per_proc << ", total=" << total_points );
    auto omega_point = endpoint<index_int,2>( points_per_proc*the_env.nprocs() );
    coordinate<index_int,2> omega( omega_point );
    INFO( "omega=" << omega.as_string() );
    mpi_decomposition<2> procs( the_env );
    INFO( "procs=" << procs.as_string() );
    mpi_distribution<2> omega_p( omega,procs );
    REQUIRE_NOTHROW( omega_p.local_domain() );
    indexstructure<index_int,2> local_domain = omega_p.local_domain();
    index_int check_total_points = the_env.allreduce_ii( local_domain.volume() );
    REQUIRE( check_total_points==total_points );
  }
}

TEST_CASE( "replicated distributions","[mpi][distribution][replication][04]" ) {
  {
    INFO( "1D" );
    const int points_per_proc = ipower(10,1);
    index_int total_points = points_per_proc*the_env.nprocs();
    coordinate<index_int,1> omega( total_points );
    mpi_decomposition<1> procs( the_env );
    REQUIRE_NOTHROW( mpi_distribution<1>( omega,procs,distribution_type::replicated ) );
    mpi_distribution<1> repl1( omega,procs,distribution_type::replicated );
    REQUIRE_NOTHROW( repl1.local_domain() );
    indexstructure<index_int,1> local_domain = repl1.local_domain();
    REQUIRE( local_domain.volume()==total_points );
  }
  {
    INFO( "2D" );
    const int points_per_proc = ipower(10,2);
    index_int total_points = points_per_proc*the_env.nprocs();
    coordinate<index_int,2> omega( endpoint<index_int,2>(total_points) );
    mpi_decomposition<2> procs( the_env );
    REQUIRE_NOTHROW( mpi_distribution<2>( omega,procs,distribution_type::replicated ) );
    mpi_distribution<2> repl2( omega,procs,distribution_type::replicated );
    REQUIRE_NOTHROW( repl2.local_domain() );
    indexstructure<index_int,2> local_domain = repl2.local_domain();
    REQUIRE( local_domain.volume()==total_points );
  }
}

TEST_CASE( "replicated scalars","[mpi][distribution][replication][05]" ) {
  {
    INFO( "1D" );
    mpi_decomposition<1> procs( the_env );
    REQUIRE_NOTHROW( replicated_scalar_distribution<1>( procs ) );
    auto repl1     = replicated_scalar_distribution<1>( procs );
    REQUIRE_NOTHROW( repl1.local_domain() );
    indexstructure<index_int,1> local_domain = repl1.local_domain();
    REQUIRE( local_domain.volume()==1 );
  }
  {
    INFO( "2D" );
    mpi_decomposition<2> procs( the_env );
    REQUIRE_NOTHROW( replicated_scalar_distribution<2>( procs ) );
    auto repl2     = replicated_scalar_distribution<2>( procs );
    REQUIRE_NOTHROW( repl2.local_domain() );
    indexstructure<index_int,2> local_domain = repl2.local_domain();
    REQUIRE( local_domain.volume()==1 );
  }
}

/****
 **** Operations on distribution
 ****/

TEST_CASE( "distribution shifting" ) {
  {
    INFO( "1D" );
    // processors
    mpi_decomposition<1> procs( the_env );
    // global domain
    coordinate<index_int,1> omega( procs.domain_layout()*16 );
    index_int total_points = omega.span();
    domain<1> dom(omega);
    // distributed domain
    mpi_distribution<1> dist( omega,procs );
    INFO( "original domain: " << dist.global_domain().as_string() );
    REQUIRE( dist.global_domain().first_index()==constant_coordinate<index_int,1>(0) );

    ioperator<index_int,1> right1(">>1");
    REQUIRE_NOTHROW( dom.operate( right1 ) );
    REQUIRE_NOTHROW( dist.operate( right1 ) );
    auto new_dist = dist.operate( right1 );
    REQUIRE_NOTHROW( new_dist.global_domain() );
    REQUIRE( new_dist.global_domain().first_index()==constant_coordinate<index_int,1>(1) );

    REQUIRE_NOTHROW( new_dist.global_domain() );
    auto new_global = new_dist.global_domain();
    INFO( "shifted global : " << new_global.as_string() );
    REQUIRE( new_global.volume()==total_points );

    INFO( "original local : " << dist.local_domain().as_string() );
    REQUIRE_NOTHROW( new_dist.local_domain() );
    auto new_local = new_dist.local_domain();
    INFO( "shifted local  : " << new_local.as_string() );
    REQUIRE( new_local.volume()==dist.local_domain().volume() );
  }
  {
    INFO( "2D" );
    // processors
    mpi_decomposition<2> procs( the_env );
    // global domain
    coordinate<index_int,2> omega( procs.domain_layout()*16 );
    index_int total_points = omega.span();
    domain<2> dom(omega);
    // distributed domain
    mpi_distribution<2> dist( omega,procs );
    INFO( "original domain: " << dist.global_domain().as_string() );
    REQUIRE( dist.global_domain().first_index()==constant_coordinate<index_int,2>(0) );

    ioperator<index_int,2> right1(">>1");
    REQUIRE_NOTHROW( dom.operate( right1 ) );
    REQUIRE_NOTHROW( dist.operate( right1 ) );
    auto new_dist = dist.operate( right1 );
    REQUIRE_NOTHROW( new_dist.global_domain() );
    REQUIRE( new_dist.global_domain().first_index()==constant_coordinate<index_int,2>(1) );

    REQUIRE_NOTHROW( new_dist.global_domain() );
    auto new_global = new_dist.global_domain();
    INFO( "shifted global : " << new_global.as_string() );
    REQUIRE( new_global.volume()==total_points );

    INFO( "original local : " << dist.local_domain().as_string() );
    REQUIRE_NOTHROW( new_dist.local_domain() );
    auto new_local = new_dist.local_domain();
    INFO( "shifted local  : " << new_local.as_string() );
    REQUIRE( new_local.volume()==dist.local_domain().volume() );
  }
}

TEST_CASE( "divided distributions","[mpi][distribution][operation][06]" ) {
  {
    INFO( "1D" );
    const int points_per_proc = ipower(10,1);
    index_int total_points = points_per_proc*the_env.nprocs();
    coordinate<index_int,1> omega( total_points );
    mpi_decomposition<1> procs( the_env );
    mpi_distribution<1> dist( omega,procs );
    INFO( "original domain: " << dist.global_domain().as_string() );

    ioperator<index_int,1> div2("/2");
    REQUIRE_NOTHROW( dist.operate( div2 ) );
    auto new_dist = dist.operate( div2 );
    REQUIRE_NOTHROW( new_dist.global_domain() );

    REQUIRE_NOTHROW( new_dist.global_domain() );
    auto new_global = new_dist.global_domain();
    INFO( "divided global : " << new_global.as_string() );
    REQUIRE( new_global.volume()==total_points/2 );

    INFO( "original local : " << dist.local_domain().as_string() );
    REQUIRE_NOTHROW( new_dist.local_domain() );
    auto new_local = new_dist.local_domain();
    INFO( "divided local  : " << new_local.as_string() );
    REQUIRE( new_local.volume()==points_per_proc/2 );
  }
  {
    INFO( "2D" );
    mpi_decomposition<2> procs( the_env );
    INFO( "Decomposition: " << the_env.as_string() );

    coordinate<index_int,2> omega( procs.domain_layout()*16 /* total_points */ );
    index_int total_points = omega.span();
    mpi_distribution<2> dist( omega,procs );
    INFO( "original domain: " << dist.global_domain().as_string() );

    ioperator<index_int,2> div2("/2");
    REQUIRE_NOTHROW( dist.operate( div2 ) );
    auto new_dist = dist.operate( div2 );
    REQUIRE_NOTHROW( new_dist.global_domain() );

    REQUIRE_NOTHROW( new_dist.global_domain() );
    auto new_global = new_dist.global_domain();
    INFO( "divided global : " << new_global.as_string() );
    REQUIRE( new_global.volume()==total_points/4 );

    INFO( "original local : " << dist.local_domain().as_string() );
    REQUIRE_NOTHROW( new_dist.local_domain() );
    auto new_local = new_dist.local_domain();
    INFO( "divided local  : " << new_local.as_string() );
    index_int check_total_points = the_env.allreduce_ii( new_local.volume() );
    REQUIRE( check_total_points==total_points/4 );
  }
}

TEST_CASE( "NUMA addressing" ) {
  auto my_pnum = the_env.procid();
  auto my_pcrd = 0;
  INFO( "proc: " << my_pnum );
  {
    INFO( "1D" );
    const int points_per_proc = ipower(10,1);
    index_int total_points = points_per_proc*the_env.nprocs();
    coordinate<index_int,1> omega( total_points );
    mpi_decomposition<1> procs( the_env );
    mpi_distribution<1> omega_p( omega,procs );

    for ( int p=0; p<the_env.nprocs(); p++ ) {
      auto pcoord = procs.coordinate_from_linear(p);
      if (p==my_pnum) {      
	REQUIRE_NOTHROW( omega_p.location_of_first_index(pcoord) );
	REQUIRE( omega_p.location_of_first_index(pcoord)==0 );
      } else {
	REQUIRE_THROWS( omega_p.location_of_first_index(pcoord) );
      }
    }
  }
  {
    INFO( "2D" );
    mpi_decomposition<2> procs( the_env );
    INFO( "Decomposition: " << procs.as_string() );

    coordinate<index_int,2> omega( procs.domain_layout()*16 /* total_points */ );
    index_int total_points = omega.span();
    mpi_distribution<2> omega_p( omega,procs );

    for ( int p=0; p<the_env.nprocs(); p++ ) {
      auto pcoord = procs.coordinate_from_linear(p);
      if (p==my_pnum) {      
	REQUIRE_NOTHROW( omega_p.location_of_first_index(pcoord) );
	REQUIRE( omega_p.location_of_first_index(pcoord)==0 );
      } else {
	REQUIRE_THROWS( omega_p.location_of_first_index(pcoord) );
      }
    }
  }
}

/*
 * Distribution intersection
 */
TEST_CASE( "shift right" ) {
  {
    INFO( "1D" );
    mpi_decomposition<1> procs( the_env );
    coordinate<index_int,1> omega( procs.domain_layout()*16 );
    index_int total_points = omega.span();
    domain<1> dom(omega);
    mpi_distribution<1> dist( omega,procs );

    // shift distribution to the right
    ioperator<index_int,1> right1(">>1");
    auto shifted = dist.operate( right1 );

    // p is a domain we depend on
    int p = the_env.procid();
    auto p_domain = shifted.local_domain();
    for ( int q=0; q<the_env.nprocs(); q++ ) {
      INFO( p << " intersection " << q );
      auto q_domain = dist.local_domain(q);
      REQUIRE_NOTHROW( p_domain.intersect(q_domain) );
      auto pq_intersection  = p_domain.intersect(q_domain);
      if ( q==p or q==p+1 )
	REQUIRE( not pq_intersection.is_empty() );
      else 
	REQUIRE( pq_intersection.is_empty() );
    }
  }
}


#if 0

TEST_CASE( "Operated distributions with modulo","[mpi][distribution][modulo][24]" ) {

  INFO( "mytid=" << mytid );

  int nlocal = 10, gsize = nlocal*ntids;
  auto d1 = 
    shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,-1,gsize) );
  // record information for the original distribution
  auto
    first = d1->first_index_r(mycoord),
    last = d1->last_index_r(mycoord);
  index_int
    localsize = d1->volume(mycoord);

  // the unshifted distribution
  CHECK( d1->volume(mycoord)==localsize );
  CHECK( arch.get_protocol()==protocol_type::MPI );
  CHECK( d1->get_protocol()==protocol_type::MPI );
  CHECK( d1->local_allocation()==nlocal );
  CHECK( d1->contains_element(mycoord,first) );
  CHECK( d1->contains_element(mycoord,last) );
  
  // now check information for the shifted distribution, modulo
  auto d1shift = 
    shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,-1,gsize) );
  auto shift_op = new multi_ioperator( ioperator(">>1") );
  CHECK( shift_op->is_modulo_op() );
  REQUIRE_NOTHROW( d1shift->operate( shift_op ) );

  //int fshift=MOD(first+1,gsize),lshift=MOD(last+1,gsize);
  auto fshift=(first+1)%gsize, lshift=(last+1)%gsize;
  CHECK( d1shift->volume(mycoord)==localsize );
  CHECK( d1shift->contains_element(mycoord,fshift) );
}

TEST_CASE( "dividing parstruct","[structure][ortho][25]" ) {

  int nlocal = 8, k,gsize = nlocal*ntids;
  parallel_structure level_dist, new_dist;
  ioperator coarsen; const char *path;
  // SECTION( "div:" ) { path = "div:2";
  //   coarsen = new ioperator(":2");
  // }
  SECTION( "div/" ) { path = "div/2";
    coarsen = ioperator("/2");
  }
  INFO( "path: " << path );

  REQUIRE_NOTHROW( level_dist = parallel_structure(decomp) );
  REQUIRE_NOTHROW( level_dist.create_from_global_size(gsize) );
  //  INFO( "original dist: " << level_dist.as_string() );
  CHECK( level_dist.volume(mycoord)==nlocal );

  { // see what we do with just a processor structure
    shared_ptr<multi_indexstruct> coarse,fine;
    REQUIRE_NOTHROW( fine = level_dist.get_processor_structure(mycoord) );
    INFO( "fine struct: " << fine->as_string() );
    REQUIRE_NOTHROW( coarse = fine->operate(coarsen) );
    INFO( "coarse struct: " << coarse->as_string() );
    CHECK( coarse->volume()==fine->volume()/2 );
  }
  // now for real with the distribution
  REQUIRE_NOTHROW( level_dist = level_dist.operate(coarsen) );
  INFO( "operated dist: " << level_dist.as_string() );
  fmt::print("operated dist: {}\n",level_dist.as_string() );

  CHECK( level_dist.volume(mycoord)==nlocal/2 );
}

TEST_CASE( "dividing distribution","[distribution][ortho][26]" ) {

  int nlocal = 8, k,gsize = nlocal*ntids;
  shared_ptr<distribution> level_dist, new_dist;
  ioperator coarsen; const char *path;
  // SECTION( "div:" ) { path = "div:2";
  //   coarsen = new ioperator(":2");
  // }
  SECTION( "div/" ) { path = "div/2";
    coarsen = ioperator("/2");
  }
  INFO( "path: " << path );
  for (int k=1; k<=3; k++) {
    INFO( "k=" << k );
    REQUIRE_NOTHROW( level_dist = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,k,-1,gsize) ) );
    INFO( "original dist: " << level_dist->as_string() );
    CHECK( level_dist->get_orthogonal_dimension()==k );
    CHECK( level_dist->local_allocation()==k*nlocal );

    REQUIRE_NOTHROW( level_dist = level_dist->operate(coarsen) );
    INFO( "operated dist: " << level_dist->as_string() );

    CHECK( level_dist->local_allocation()==k*nlocal/2 );
    INFO( "divided dist: " << level_dist->as_string() );
    CHECK( level_dist->get_orthogonal_dimension()==k );
    //CHECK( level_dist->has_type_contiguous() );
  }
}

TEST_CASE( "extending distributions","[distribution][extend][27]" ) {
  if (ntids<2) {
    printf("test 27 needs multiple processes ????\n");
    //  return;
  }

  int dim = 1;
  int nlocal=100,nglobal=nlocal*ntids;
  shared_ptr<distribution> d1,d2;
  REQUIRE_NOTHROW( d1 = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,nlocal,-1) ) );
  coordinate<index_int,d>
    my_first = d1->first_index_r(mycoord), my_last = d1->last_index_r(mycoord);

  int shift = 1;
  //  SECTION( "keep it contiguous" ) {
    shift = 1;
    //}
  // SECTION( "make it composite" ) {
  //   shift = 2;
  // }
  INFO( "mytid=" << mytid << "\nusing shift: " << shift );

  {
    coordinate<index_int,d> the_first(dim), the_last(dim);
    REQUIRE_NOTHROW( the_first = d1->first_index_r(mycoord) );
    REQUIRE_NOTHROW( the_last = d1->last_index_r(mycoord) );
    shared_ptr<multi_indexstruct> estruct,xstruct;

    coordinate<int,d> close(1);
    REQUIRE_NOTHROW( close = decomp.get_origin_processor() );
    if (mycoord==close)
      REQUIRE_NOTHROW( estruct = shared_ptr<multi_indexstruct>
		       ( new contiguous_multi_indexstruct ( the_first-shift ) ) );
    else if (mycoord==decomp.get_farpoint_processor())
      REQUIRE_NOTHROW( estruct = shared_ptr<multi_indexstruct>
		       ( new contiguous_multi_indexstruct( the_last+shift ) ) );
    else
      REQUIRE_NOTHROW( estruct = shared_ptr<multi_indexstruct>
		       ( new empty_multi_indexstruct(dim) ) );

    REQUIRE_NOTHROW( d2 = d1->extend(mycoord,estruct) );
//     fmt::print("going to print\n");
//     fmt::print("Extended structure: {}",d2->as_string());
  }
  // the print statement in d1->extend succeeds, the following doesn't
  return;

  if (mytid==0 || mytid==ntids-1)
    CHECK( d2->volume(mycoord)==(nlocal+1) );
  else
    CHECK( d2->volume(mycoord)==(nlocal+2) );
  return ;
  if (mytid==0) 
    CHECK( d2->first_index_r(mycoord)==my_first );
  else
    CHECK( d2->first_index_r(mycoord)==my_first-shift );
  if (mytid==ntids-1) 
    CHECK( d2->last_index_r(mycoord)==my_last );
  else
    CHECK( d2->last_index_r(mycoord)==my_last+shift );
}

TEST_CASE( "orthogonal dimension","[distribution][ortho][30]" ) {
  int nlocal=100,nglobal=nlocal*ntids;
  shared_ptr<distribution> d1; int k; const char *path;
  SECTION( "default k=1" ) { k=1; path = "k=1 by default";
    REQUIRE_NOTHROW( d1 = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,nlocal,-1) ) );
  }
  SECTION( "explicit k=1" ) { k=1; path = "k=1 explicit";
    REQUIRE_NOTHROW( d1 = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,k,nlocal,-1) ) );
  }
  SECTION( "k=2" ) { k=2; path = "k=2 explicit";
    REQUIRE_NOTHROW( d1 = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,k,nlocal,-1) ) );
  }
  CHECK( d1->volume(mycoord)==nlocal );
  CHECK( d1->local_allocation_p(mycoord)==k*nlocal );
}

TEST_CASE( "Cyclic distributions","[distribution][cyclic][40]" ) {
  shared_ptr<distribution> d;
  //  REQUIRE_THROWS( d = shared_ptr<distribution>( new mpi_cyclic_distribution(decomp,-1,-1) ) );
  //  REQUIRE_THROWS( d = shared_ptr<distribution>( new mpi_cyclic_distribution(decomp,1,ntids+1) ) );

  // each proc gets 2 elements: mytid,mytid+ntids
  REQUIRE_NOTHROW( d = shared_ptr<distribution>( new mpi_cyclic_distribution(decomp,-1,2*ntids) ) );
  CHECK( d->volume(mycoord)==2 );
  CHECK( d->first_index_r(mycoord).coord(0)==mytid );
  CHECK( d->last_index_r(mycoord).coord(0)==mytid+ntids );
}

TEST_CASE( "Block cyclic distributions","[distribution][cyclic][41]" ) {
  shared_ptr<distribution> d;
  REQUIRE_NOTHROW( d = shared_ptr<distribution>( new mpi_blockcyclic_distribution(decomp,5,4,-1) ) );
  CHECK( d->volume(mycoord)==20 );
  CHECK( d->first_index_r(mycoord).coord(0)==mytid*5 );
  CHECK( d->get_processor_structure(mycoord)->get_component(0)->contains_element( 5*ntids + mytid*5 ) );
}

index_int pfunc1(int p,index_int i) {
  return 3*p+i;
}

index_int pfunc2(int p,index_int i) {
  return 3*(p/2)+i;
}

TEST_CASE( "Function-specified distribution","[distribution][50]" ) {
  INFO( "mytid=" << mytid );
  shared_ptr<distribution> d1,d2;
  int nlocal = 3;

  CHECK_NOTHROW( d1 = shared_ptr<distribution>( new mpi_distribution(decomp,&pfunc1,nlocal ) ) );
  CHECK( d1->volume(mycoord)==nlocal );
  for (int i=0; i<nlocal; i++) {
    index_int iglobal = 3*mytid+i;
    CHECK( d1->contains_element(mycoord,coordinate<index_int,d>(vector<index_int>{iglobal})) );
    CHECK( d1->find_index(iglobal)==mytid );
  }

  CHECK_NOTHROW( d2 = shared_ptr<distribution>( new mpi_distribution(decomp,&pfunc2,nlocal ) ) );
  CHECK( d2->volume(mycoord)==nlocal );
  for (int i=0; i<nlocal; i++) {
    index_int iglobal = 3*(mytid/2)+i; // proc 0,1 have same data, likewise 2,3, 4,5
    CHECK( d2->contains_element(mycoord,coordinate<index_int,d>(vector<index_int>{iglobal})) );
    CHECK( d2->find_index(iglobal,mytid)==mytid );
    CHECK( d2->find_index(iglobal)==2*(mytid/2) ); // the first proc with my data is 2*(p/2)
  }
}

TEST_CASE( "Distribution transformations","[distribution][operate][abut][60]" ) {
  shared_ptr<distribution> block,newblock;
  index_int
    localsize = 10*(mytid+1),
    globalsize = 5*ntids*(ntids+1);
  REQUIRE_NOTHROW
    ( block =
      shared_ptr<distribution>
      ( make_shared<mpi_block_distribution>(decomp,localsize,globalsize) ) );
  INFO( fmt::format("Irregular block dist: {}",block->as_string()) );
  auto first = block->first_index_r(mycoord);

  index_int gsize = block->global_volume();
  CHECK( gsize==globalsize );
  coordinate<index_int,d> first_coord(1),last_coord(1);
  decltype( block->get_enclosing_structure() ) enc;
  REQUIRE_NOTHROW( enc = block->get_enclosing_structure() );
  //  REQUIRE( enc!=nullptr );
  REQUIRE_NOTHROW( first_coord = enc->first_index_r()+1 );
  REQUIRE_NOTHROW( last_coord = enc->last_index_r()+1 );

  const char *path;
  fmt::format("Distribution 60: transofrmations, bunch of stuff commented out");
  // SECTION( "operate pidx" ) { path = "operate pidx";
  //   parallel_indexstruct *pidx, *new_pidx;
  //   REQUIRE_NOTHROW( pidx = block->get_dimension_structure(0) );
  //   SECTION( "classic transformation" ) {
  //     REQUIRE_NOTHROW( new_pidx = pidx->operate(times2) );
  //   }
  //   SECTION( "sigma transformation by point" ) {
  //     sigma_operator *pidx_times2 = new sigma_operator(times2);
  //     CHECK( pidx_times2->is_point_operator() );
  //     REQUIRE_NOTHROW( new_pidx = pidx->operate(pidx_times2) );
  //   }
  //   SECTION( "sigma transformation by struct" ) {
  //     sigma_operator *pidx_times2 = new sigma_operator
  //   	( [times2] (indexstruct &i) -> shared_ptr<indexstruct>
  //   	  { return i.operate(times2); } );
  //     REQUIRE_NOTHROW( new_pidx = pidx->operate(pidx_times2) );
  //   }
  //   SECTION( "dynamic transformation" ) {
  //     sigma_operator *pidx_times2 = new sigma_operator
  //   	( [times2] (indexstruct &i) -> shared_ptr<indexstruct>
  //   	  { shared_ptr<indexstruct> opstruct = i.operate(times2);
  //   	    //fmt::print("operate on struct {} gives {}\n",i.as_string(),opstruct->as_string());
  //   	    return opstruct; } );
  //     distribution_sigma_operator *dist_times2 = new distribution_sigma_operator(pidx_times2);
  //     shared_ptr<indexstruct> newstruct;
  //     REQUIRE_NOTHROW( newstruct = dist_times2->operate(0,block,mycoord) );
  //     REQUIRE_NOTHROW( new_pidx = new parallel_indexstruct(decomp.domains_volume()) );
  //     REQUIRE_NOTHROW( new_pidx->set_processor_structure(mytid,newstruct) );
  //     CHECK( !new_pidx->is_known_globally() );
  //   }
  //   INFO( fmt::format("new structure: {}",new_pidx->as_string()) );
  //   CHECK( new_pidx->first_index_r(mytid)==pidx->first_index_r(mytid)*2 );
  //   CHECK( new_pidx->get_processor_structure(mytid)->stride()==1 );
  //   CHECK( new_pidx->local_size(mytid)==pidx->local_size(mytid)*2-1 );
  //   parallel_structure parstruct;
  //   REQUIRE_NOTHROW( parstruct = new parallel_structure(decomp,new_pidx) );
  //   //REQUIRE_NOTHROW( newblock = shared_ptr<distribution>( new mpi_distribution(/*decomp,*/parstruct) ) );
  // }
  // SECTION( "operate structure" ) { path = "operate structure";
  //   parallel_structure parstruct,*new_struct;
  //   REQUIRE_NOTHROW( parstruct = dynamic_cast<parallel_structure*>(block) );
  //   CHECK( parstruct!=nullptr );
  //   SECTION( "operate parstruct" ) {
  //     REQUIRE_NOTHROW( new_struct = parstruct->operate(times2) );
  //   }
  //   SECTION( "sigma transformation" ) {
  //     auto *pidx_times2 =
  // 	new multi_sigma_operator( sigma_operator(times2) );
  //     CHECK( pidx_times2->is_point_operator() );
  //     REQUIRE_NOTHROW( new_struct = parstruct->operate(pidx_times2) );
  //   }
  //   REQUIRE_NOTHROW( newblock = shared_ptr<distribution>( new mpi_distribution(new_struct) ) );
  // }
  ioperator times2("x2");
  SECTION( "operate distribution" ) { path = "operate distribution";
    auto pidx_times2 =
      multi_sigma_operator( sigma_operator(times2) );
    REQUIRE_NOTHROW( newblock = block->operate(pidx_times2) );
  }
  SECTION( "stretch distribution" ) { path = "stretch distribution";
    fmt::print("Stretch to {}\n",last_coord.as_string());
    auto double_last = last_coord*2;
    distribution_stretch_operator stretch2(double_last);
    REQUIRE_NOTHROW( newblock = block->operate(stretch2) );
    distribution_sigma_operator op;
    REQUIRE_NOTHROW( op = distribution_abut_operator(mycoord) );
    REQUIRE_NOTHROW( newblock = newblock->operate(op) );
  }

  INFO( fmt::format("Irregular block dist: {}",block->as_string()) );
  INFO( fmt::format("Operated block dist: {}",newblock->as_string()) );
  //CHECK( newblock->first_index_r(mycoord)==first*2 );
  CHECK( newblock->global_volume()==gsize*2 );
}

#if 0
shared_ptr<multi_indexstruct> transform_by_shift
    (shared_ptr<distribution> unbalance,coordinate<int,d> &me,shared_ptr<distribution> load) {
  if (!load->has_type_replicated())
    throw(string("Load description needs to be replicated"));
  if (load->volume(me)!=unbalance->domains_volume())
    throw(fmt::format("Load vector has {} items, for {} domains",
		      load->volume(me),unbalance->domains_volume()));
  if (!unbalance->is_known_globally())
    throw(fmt::format("Can only transform-shift globally known distributions"));

  // to work!

  auto decomp = unbalance->get_decomposition();
  shared_ptr<multi_indexstruct> old_pstruct,new_pstruct;
  try {
    old_pstruct = unbalance->get_processor_structure(me);
    new_pstruct = old_pstruct->operate
      ( new multi_ioperator( ioperator(">>2") ) );
    fmt::print("shift operation {} -> {}\n",
	       old_pstruct->as_string(),new_pstruct->as_string());
  } catch (string c) { fmt::print("{}: Error <<{}>>\n",me.as_string(),c);
    throw(string("Could not redistribute by shift")); }
  return new_pstruct;
}

TEST_CASE( "Distribution operation by simple shift","[distribution][operate][61]" ) {
  index_int nlocal = 10*mytid+1;
  auto
    block = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,nlocal,-1) ), 
    load = shared_ptr<distribution>( new mpi_replicated_distribution(decomp,ntids) );
  shared_ptr<distribution> newblock;
  block->set_name("blockdist61");
  auto first = block->first_index_r(mycoord);
  auto saved_mycoord = shared_ptr<coordinate<int,d>>
    ( new coordinate<int,d>(mycoord) );
  auto average =
    distribution_sigma_operator
    ( [load] (shared_ptr<distribution> d,coordinate<int,d> &p)
          -> shared_ptr<multi_indexstruct> { return transform_by_shift(d,p,load); } );
  CHECK( average.is_coordinate_based() );
  REQUIRE_NOTHROW( newblock = block->operate(average) );
  index_int check_local;
  REQUIRE_NOTHROW( check_local = newblock->volume(mycoord) );
  CHECK( check_local==nlocal );
  auto p_old = block->get_processor_structure(mycoord),
    p_new = newblock->get_processor_structure(mycoord);
  INFO( fmt::format("old block: {}, shifted block: {}",
		    p_old->as_string(),p_new->as_string()) );
  CHECK( p_new->first_index_r()==p_old->first_index_r()+2 );
}

shared_ptr<multi_indexstruct> transform_by_multi
    (shared_ptr<distribution> unbalance,coordinate<int,d> &me,shared_ptr<distribution> load) {
  if (!load->has_type_replicated())
    throw(string("Load description needs to be replicated"));
  if (load->volume(me)!=unbalance->domains_volume())
    throw(fmt::format("Load vector has {} items, for {} domains",
		      load->volume(me),unbalance->domains_volume()));
  if (!unbalance->is_known_globally())
    throw(fmt::format("Can only transform-shift globally known distributions"));

  // to work!

  auto decomp = unbalance->get_decomposition();
  shared_ptr<multi_indexstruct> old_pstruct,new_pstruct;
  try {
    old_pstruct = unbalance->get_processor_structure(me);
    new_pstruct = old_pstruct->operate
      ( new multi_ioperator( ioperator("x2") ) );
    fmt::print("shift operation {} -> {}\n",
	       old_pstruct->as_string(),new_pstruct->as_string());
  } catch (string c) { fmt::print("{}: Error <<{}>>\n",me.as_string(),c);
    throw(string("Could not redistribute by shift")); }
  return new_pstruct;
}

TEST_CASE( "Distribution operation by local blowup","[distribution][operate][abut][62]" ) {
  index_int nlocal = 10*mytid+1;
  shared_ptr<distribution> block =
    shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,nlocal,-1) ),
    newblock,
    load = shared_ptr<distribution>( new mpi_replicated_distribution(decomp,ntids) );
  block->set_name("blockdist61");
  auto first = block->first_index_r(mycoord);
  auto saved_mycoord = shared_ptr<coordinate<int,d>>
    ( new coordinate<int,d>(mycoord) );
  auto average =
    distribution_sigma_operator
    ( [load] (shared_ptr<distribution> d,coordinate<int,d> &p)
      -> shared_ptr<multi_indexstruct> {
      return transform_by_multi(d,p,load); } );
  REQUIRE_NOTHROW( newblock = block->operate(average) );
  REQUIRE_NOTHROW( newblock = newblock->operate( distribution_abut_operator(mycoord) ) );
  auto p_old = block->get_processor_structure(mycoord),
    p_new = newblock->get_processor_structure(mycoord);
  INFO( fmt::format("old block: {}, shifted block: {}",
		    p_old->as_string(),p_new->as_string()) );
  CHECK( p_new->volume()==2*p_old->volume() );

  shared_ptr<object> new_object;
  REQUIRE_NOTHROW( new_object = shared_ptr<object>( new mpi_object(newblock) ) );
}

TEST_CASE( "Make a distribution abutting","[distribution][operate][abut][63]" ) {
  index_int nlocal = 10;
  parallel_structure pstr;
  REQUIRE_NOTHROW( pstr = parallel_structure(decomp) );
  for (int p=0; p<ntids; p++) {
    coordinate<int,d> pcoord( vector<int>{p} );
    coordinate<index_int,d>
      first(vector<index_int>{p*nlocal}),
      last(vector<index_int>{(p+2)*nlocal-1});
    auto pstruct = shared_ptr<multi_indexstruct>
      ( new contiguous_multi_indexstruct(first,last) );
    pstr.set_processor_structure(pcoord,pstruct);
  }
  shared_ptr<distribution> messy;
  REQUIRE_NOTHROW( messy = shared_ptr<distribution>( new mpi_distribution(pstr) ) );
  shared_ptr<object> messy_object;
  REQUIRE_NOTHROW( messy_object = shared_ptr<object>( new mpi_object(messy) ) );
  shared_ptr<distribution> sizes;
  REQUIRE_NOTHROW( sizes = shared_ptr<distribution>( new mpi_gathered_distribution(decomp) ) );
  CHECK( sizes->volume(mycoord)==ntids );
  shared_ptr<object> size_summary;
  REQUIRE_NOTHROW( size_summary = shared_ptr<object>( new mpi_object(sizes) ) );
  shared_ptr<kernel> gather_sizes;
  REQUIRE_NOTHROW( gather_sizes = shared_ptr<kernel>( new mpi_gather_kernel
		   (messy_object,size_summary,[] ( kernel_function_args ) -> void {
		     //double *out_data = outvector->get_data(p);
		     //out_data[0] = invectors->at(0)->volume(p);
		   } ) ) );
  REQUIRE_NOTHROW( gather_sizes->analyze_dependencies() );
  REQUIRE_NOTHROW( gather_sizes->execute() );
  fmt::print("Premature exit from [63]\n");
  return;

  shared_ptr<distribution> clean;
  REQUIRE_NOTHROW( clean = messy->operate( distribution_abut_operator(mycoord) ) );
  for (int p=0; p<ntids; p++) {
    coordinate<int,d> pcoord( vector<int>{p} );
    auto pstruct = clean->get_processor_structure(pcoord);
    INFO( fmt::format("P={}, struct={}",p,pstruct->as_string()) );
    CHECK( pstruct->first_index_r().at(0)==2*p*nlocal );
    CHECK( pstruct->last_index_r().at(0)==2*(p+1)*nlocal );
  }
}

TEST_CASE( "Distribution stretch","[distribution][operate][stretch][64]" ) {
  shared_ptr<distribution> block, stretched;
  float factor;

  // SECTION( "regular blocked" ) {
  //   index_int nlocal = 10;
  //   REQUIRE_NOTHROW( block = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,nlocal,-1) ) );
  //   SECTION( "2" ) { factor = 2; };
  //   SECTION( "5" ) { factor = 5; };
  //   SECTION( "1/3" ) { factor = 1./3; };
  // }
  SECTION( "irregular blocked" ) {
    vector<index_int> nlocals(ntids);
    for (int id=0; id<ntids; id++ )
      nlocals[id] = 10+8*id;
    shared_ptr<parallel_indexstruct> pidx;
    REQUIRE_NOTHROW( pidx = make_shared<parallel_indexstruct>(nlocals) );
    REQUIRE_NOTHROW
      ( block = shared_ptr<distribution>( new mpi_distribution(decomp,pidx) ) );
    SECTION( "2" ) { factor = 2; };
    SECTION( "5" ) { factor = 5; };
    SECTION( "1/3" ) { factor = 1./3; };
  }
  return;
  INFO( fmt::format("Stretching {} by factor {}",block->as_string(),factor) );
  index_int gsize;
  REQUIRE_NOTHROW( gsize = factor*block->global_volume() );
  coordinate<index_int,d> big;
  REQUIRE_NOTHROW( big = coordinate<index_int,d>( vector<index_int>{gsize} ) );
  INFO( fmt::format("Stretching to {}",big.as_string()) );
  REQUIRE( big.get_dimensionality()==1 );

  distribution_sigma_operator stretch;
  REQUIRE_NOTHROW( stretch = distribution_stretch_operator(big) );
  REQUIRE_NOTHROW( stretched = block->operate(stretch) );
  INFO( fmt::format("Stretched distro: {}",stretched->as_string()) );
  CHECK( stretched->global_volume()==gsize );
  CHECK( stretched->is_known_globally() );

  shared_ptr<multi_indexstruct> old_pstruct,new_pstruct;
  REQUIRE_NOTHROW( old_pstruct = block->get_processor_structure(mycoord) );
  REQUIRE_NOTHROW( new_pstruct = stretched->get_processor_structure(mycoord) );
  index_int my_nlocal, new_nlocal = factor*my_nlocal;
  REQUIRE_NOTHROW( my_nlocal = old_pstruct->volume() );
  INFO( fmt::format
	("\nold pstruct: {}, volume={},\nintended volume={},\nnew pstruct: {}, volume={}",
	 old_pstruct->as_string(),my_nlocal,
	 new_nlocal,
	 new_pstruct->as_string(),new_pstruct->volume()) );
  CHECK( new_pstruct->volume()>=new_nlocal-1 );
  CHECK( new_pstruct->volume()<=new_nlocal+1 );

}

TEST_CASE( "Distribution operation by averaging","[distribution][operate][abut][65]" ) {
  index_int nlocal = 12;
  auto
    block = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,nlocal,-1) );
  index_int gsize = block->global_volume();
  coordinate<index_int,d> glast = block->global_size();
  block->set_name("blockdist61");

  // the load object is replicated
  auto load = shared_ptr<distribution>( new mpi_replicated_distribution(decomp,ntids) );
  auto stats_object = shared_ptr<object>( new mpi_object(load) );
  
  SECTION ( "perfectly balanced" ) {
    double *stats_data = stats_object->get_data(mycoord);
    for (int p=0; p<ntids; p++) stats_data[p] = 1.;
  }
  SECTION ( "linearly increasing" ) {
    double *stats_data = stats_object->get_data(mycoord);
    for (int p=0; p<ntids; p++) stats_data[p] = 1.+p;
  }

  // the diffusion operation
  MatrixXd adjacency;
  {
    auto partition_points = block->partitioning_points();
    int p = partition_points.size()-1;
    REQUIRE_NOTHROW( adjacency = AdjacencyMatrix1D(p) );
  }
  auto diffusion =
    distribution_sigma_operator
    ( [stats_object,adjacency]
      (shared_ptr<distribution> d) -> shared_ptr<distribution> {
      return transform_by_diffusion(d,stats_object,adjacency); } );
  shared_ptr<distribution> newblock;
  REQUIRE_NOTHROW( newblock = block->operate(diffusion) );
  INFO( fmt::format("old block: {}, avg block: {}, stretch block : {}",
		    block->get_processor_structure(mycoord)->as_string(),
		    newblock->get_processor_structure(mycoord)->as_string(),
		    newblock->get_processor_structure(mycoord)->as_string()
		    ) );
  index_int checksize;
  REQUIRE_NOTHROW( checksize = newblock->global_volume() );
  CHECK( checksize==gsize );

  shared_ptr<object> new_object;
  REQUIRE_NOTHROW( new_object = shared_ptr<object>( new mpi_object(newblock) ) );
}

TEST_CASE( "Masked distribution creation","[distribution][mask][70]" ) {
  if( ntids<2 ) { printf("masking requires two procs\n"); return; }

  index_int localsize = 5;
  processor_mask *mask;

  fmt::MemoryWriter path;
  SECTION( "create mask by adding" ) { path.write("adding odd");
    REQUIRE_NOTHROW( mask = new processor_mask(decomp) );
    for (int p=1; p<ntids; p+=2 ) {
      coordinate<int,d> c(1); c.set(0,p);
      REQUIRE_NOTHROW( mask->add(c) );
    }
  }
  SECTION( "create mask by subtracting" ) { path.write("subtracting odd");
    REQUIRE_NOTHROW( mask = new processor_mask(decomp,ntids) );
    for (int p=0; p<ntids; p+=2) {
      coordinate<int,d> c(1); c.set(0,p);
      REQUIRE_NOTHROW( mask->remove(p) );
    }
  }
  INFO( "masked created by: " << path.str() );

  auto block = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,localsize,-1) ),
    masked_block = shared_ptr<distribution>( new mpi_distribution( *block ) );
  REQUIRE_NOTHROW( masked_block->add_mask(mask) );

  auto 
    whole_vector = shared_ptr<object>( new mpi_object(block) ),
    masked_vector = shared_ptr<object>( new mpi_object(masked_block) );
  REQUIRE_NOTHROW( whole_vector->allocate() );
  REQUIRE_NOTHROW( masked_vector->allocate() );
  CHECK( whole_vector->get_allocated_space()==localsize );
  if (mytid%2==1)
    CHECK( masked_vector->get_allocated_space()==localsize );
  else
    CHECK( masked_vector->get_allocated_space()==0 );
  double *data;
  CHECK_NOTHROW( data = whole_vector->get_data(mycoord) );
  {
    CHECK( block->lives_on(mycoord) );
    if (mytid%2==1) {
      CHECK( masked_block->lives_on(mycoord) );
      REQUIRE_NOTHROW( data = masked_vector->get_data(mycoord) );
    } else {
      CHECK( !masked_block->lives_on(mycoord) );
      REQUIRE_THROWS( data = masked_vector->get_data(mycoord) );
    }
  }
}

TEST_CASE( "processor sets","[processor][80]" ) {
  coordinate_set<int,1> set;
  // add a vector
  REQUIRE_NOTHROW
    ( set.add( coordinate<int,d>( vector<int>{1,2,3} ) ) );
  // check that it's there
  CHECK( set.size()==1 );
  CHECK( set.contains( coordinate<int,d>( vector<int>{1,2,3} ) ) );
  // add another vector and check
  REQUIRE_NOTHROW
    ( set.add( coordinate<int,d>( vector<int>{3,2,1} ) ) );
  CHECK( set.size()==2 );
  CHECK( set.contains( coordinate<int,d>( vector<int>{3,2,1} ) ) );
  // we can not add a vector of a different dimension
  CHECK_THROWS( set.add( coordinate<int,d>( vector<int>{3,2} ) ) );
  // we can not test a vector of a different dimension
  CHECK_THROWS( set.contains( coordinate<int,d>( vector<int>{3,2} ) ) );
  // everything still copacetic?
  CHECK( set.contains( coordinate<int,d>( vector<int>{1,2,3} ) ) );
  CHECK( set.size()==2 );

  // see if we can iterate
  int count=0,sum=0;
  for ( auto p : set ) {
    switch (count) {
    case 0 :
      CHECK( p==coordinate<int,d>( vector<int>{1,2,3} ) );
      sum += count;
      break;
    case 1 :
      CHECK( p==coordinate<int,d>( vector<int>{3,2,1} ) );
      sum += count;
      break;
    }
    count++;
  }
  CHECK( count==2 );
  CHECK( sum==1 );
}
#endif

TEST_CASE( "range over decomposition, count","[range][81]" ) {
  int domain_count{0};
  for ( auto d : decomp ) {
    domain_count++;
  }
  CHECK( domain_count==ntids );
}

TEST_CASE( "range over decomposition, coverage","[range][82]" ) {
  vector<int> check(ntids,-1);
  string path;
  SECTION( "default ranging" ) { path = "default";
  }
  SECTION( "two-sided ranging" ) { path = "twosided";
    decomp.set_range_twoside();
  }
  INFO( format("Ranging strategy: {}",path) );
  for ( auto c : decomp ) {
    int l;
    REQUIRE_NOTHROW( l = decomp.linearize(c) );
    check.at(l) = 1;
  }
  int s{0};
  for ( auto c : check )
    s++;
  CHECK( s==ntids );
}

TEST_CASE( "access data","[91]" ) {

}

#if 0
TEST_CASE( "multidimensional distributions","[multi][distribution][100]" ) {
  int ntids_i,ntids_j;
  if (ntids!=4) { printf("100 grid example needs exactly 4 procs\n"); return; }
  for (int n=sqrt(ntids); n>=1; n--)
    if (ntids%n==0) {
      ntids_j = n; ntids_i = ntids/n; break; }
  if (ntids_i==1) { printf("Could not split processor grid\n"); return; }
  CHECK( ntids_i>1 );
  CHECK( ntids_j>1 );
  CHECK( ntids==ntids_i*ntids_j );
  int mytid_j = mytid%ntids_j, mytid_i = mytid/ntids_j;

  coordinate<int,d> layout;
  REQUIRE_NOTHROW( layout = arch.get_proc_layout(2) );
  mpi_decomposition mdecomp;
  SECTION( "default splitting of processor grid" ) {
    REQUIRE_NOTHROW( mdecomp = mpi_decomposition(arch,layout) );
  }
  // SECTION( "explicit splitting of processor grid" ) {
  //   vector<int> grid; grid.push_back(2); grid.push_back(2);
  //   REQUIRE_NOTHROW( mdecomp = mpi_decomposition(arch,grid) );
  // }

  coordinate<int,d> mycoord; 
  REQUIRE_NOTHROW( mycoord = mdecomp.coordinate_from_linear(mytid) );
  INFO( "p: " << mytid << ", pcoord: " << mycoord.coord(0) << "," << mycoord.coord(1) );
  CHECK( mytid==mycoord.linearize(mdecomp) );
  CHECK( mycoord.get_dimensionality()==2 );
  INFO( "mytid=" << mytid << ", s/b " << mytid_i << "," << mytid_j );
  CHECK( mycoord.coord(0)==mytid_i );
  CHECK( mycoord.coord(1)==mytid_j );
  
  int nlocal = 10; index_int g;
  vector<index_int> domain_layout;
  g = ntids_i*(nlocal+1);
  domain_layout.push_back(g);

  g = ntids_j*(nlocal+2);
  domain_layout.push_back(g);
  coordinate<index_int,d> domain_size(domain_layout);

  shared_ptr<distribution> d;
  REQUIRE_NOTHROW( d = shared_ptr<distribution>
		   ( make_shared<mpi_block_distribution>(mdecomp,domain_size) ) );
  CHECK( d->get_dimensionality()==2 );
  shared_ptr<multi_indexstruct> local_domain;
  REQUIRE_NOTHROW( local_domain = d->get_processor_structure(mycoord) );
  CHECK( local_domain->get_dimensionality()==2 );
  CHECK( local_domain->get_component(0)->volume()==nlocal+1 );
  CHECK( local_domain->get_component(1)->volume()==nlocal+2 );

}

TEST_CASE( "multidimensional distributions error test","[multi][distribution][101]" ) {
  int ntids_i,ntids_j;
  if (ntids!=4) { printf("101 grid example needs exactly 4 procs\n"); return; }

  decomposition mdecomp;
  SECTION( "explicit splitting of processor grid" ) {
    coordinate<int,d> endpoint = coordinate<int,d>(2);
    endpoint.set(0,2); endpoint.set(1,2);
    REQUIRE_NOTHROW( mdecomp = mpi_decomposition(arch,endpoint) );
  }
  // we should not test this because we can overdecompose, true?
  // SECTION( "incorrect splitting of processor grid" ) {
  //   coordinate<int,d> *endpoint = new coordinate<int,d>(2);
  //   endpoint->set(0,2); endpoint->set(1,1);
  //   REQUIRE_THROWS( mdecomp = mpi_decomposition(arch,endpoint) );
  // }
  SECTION( "pencil splitting of processor grid" ) {
    coordinate<int,d> endpoint = coordinate<int,d>(2);
    endpoint.set(0,1); endpoint.set(1,4);
    REQUIRE_NOTHROW( mdecomp = mpi_decomposition(arch,endpoint) );
  }
}

TEST_CASE( "pencil distribution","[multi][distribution][pencil][102]" ) {
}

#endif
#endif
