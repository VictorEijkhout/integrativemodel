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
 **** unit tests for operating on parallel structures
 ****
 ****************************************************************/

#include <stdlib.h>
#include <math.h>

#include "catch2/catch_all.hpp"

#include "mpi_base.h"
#include "mpi_static_vars.h"
#include "unittest_functions.h"
#include "imp_functions.h"

using std::string;

using std::make_shared;
using std::shared_ptr;

using fmt::print;
using fmt::format;

TEST_CASE( "copy parallel indexstruct","[indexstruct][copy][51]" ) {

  int localsize = 12,gsize = ntids*localsize;
  parallel_indexstruct *pstr,*qstr;
  REQUIRE_NOTHROW( pstr = new parallel_indexstruct(ntids) );
  REQUIRE_NOTHROW( pstr->create_from_global_size( gsize ) );
  CHECK( pstr->local_size(mytid)==localsize );
  
  REQUIRE_NOTHROW( qstr = new parallel_indexstruct( *pstr ) );
  //  REQUIRE_NOTHROW( pstr->create_from_global_size( gsize+ntids ) );
  //  CHECK( pstr->local_size(mytid)==localsize+1 );
  CHECK( qstr->local_size(mytid)==localsize );
}

TEST_CASE( "shift parallel indexstruct","[indexstruct][operate][shift][52]" ) {
  
  int localsize = 12,gsize = ntids*localsize;
  shared_ptr<parallel_indexstruct> pstr,qstr;
  REQUIRE_NOTHROW( pstr = shared_ptr<parallel_indexstruct>( new parallel_indexstruct(ntids) ) );
  REQUIRE_NOTHROW( pstr->create_from_global_size( gsize ) );
  CHECK( pstr->local_size(mytid)==localsize );
  
  REQUIRE_NOTHROW( qstr = pstr->operate( ioperator(">>1") ) );
  CHECK( pstr->local_size(mytid)==localsize );
  CHECK( qstr->local_size(mytid)==localsize );
  CHECK( pstr->first_index(mytid)==(mytid*localsize) );
  CHECK( qstr->first_index(mytid)==(mytid*localsize+1) );

}

TEST_CASE( "divide parallel indexstruct","[indexstruct][operate][divide][53]" ) {
  int localsize = 15,gsize = ntids*localsize;
  shared_ptr<parallel_indexstruct> pstr,qstr;
  REQUIRE_NOTHROW( pstr = shared_ptr<parallel_indexstruct>( new parallel_indexstruct(ntids) ) );
  REQUIRE_NOTHROW( pstr->create_from_global_size( 2*gsize ) );

  REQUIRE_NOTHROW( qstr = pstr->operate( ioperator("/2") ) );
  CHECK( pstr->local_size(mytid)==(2*localsize) );
  CHECK( qstr->local_size(mytid)==localsize );
  CHECK( pstr->first_index(mytid)==(2*mytid*localsize) );
  CHECK( qstr->first_index(mytid)==(mytid*localsize) );
}

TEST_CASE( "copy and operate distributions","[distribution][operate][copy][54]" ) {
  INFO( "mytid=" << mytid );
  int localsize = 10;
  auto
    d1 = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,localsize,-1) );
  shared_ptr<distribution> d2;

  SECTION( "plain copy" ) {
    shared_ptr<parallel_indexstruct> pidx;
    REQUIRE_NOTHROW( d2 = shared_ptr<distribution>( new mpi_distribution( d1 ) ) );
    REQUIRE_NOTHROW( pidx = d2->get_dimension_structure(0) );
    REQUIRE( pidx->size()==ntids );
    index_int s1,s2;
    REQUIRE_NOTHROW( s1 = d1->volume(mycoord) );
    REQUIRE_NOTHROW( s2 = d2->volume(mycoord) );
    CHECK( s1==s2 );
  }

  SECTION( "operated copy" ) {
    auto is = ioperator(">>1");
    parallel_structure ps,po, *pp;
    REQUIRE_NOTHROW( pp = dynamic_cast<parallel_structure*>(d1.get()) );
    REQUIRE( pp!=nullptr );
    REQUIRE_NOTHROW( ps = *pp );
    REQUIRE_NOTHROW( po = ps.operate(is) );

    // identical
    REQUIRE_NOTHROW( d2 = shared_ptr<distribution>( new mpi_distribution(ps) ) );
    CHECK( d2->volume(mycoord)==d1->volume(mycoord) );
    CHECK( d2->first_index_r(mycoord)==d1->first_index_r(mycoord) );

    // shifted from index structure
    REQUIRE_NOTHROW( d2 = shared_ptr<distribution>( new mpi_distribution(po) ) );
    CHECK( d2->volume(mycoord)==d1->volume(mycoord) );
    CHECK( d2->first_index_r(mycoord)==(d1->first_index_r(mycoord)+1) );

    // shifted by operate
    REQUIRE_NOTHROW( d2 = d1->operate( is ) );
    CHECK( d2->volume(mycoord)==d1->volume(mycoord) );
    CHECK( d2->first_index_r(mycoord)==(d1->first_index_r(mycoord)+1) );
  }

  SECTION( "operate the other way" ) { // VLE when do we wrap around?
    REQUIRE_NOTHROW( d2 = d1->operate( ioperator("<<1") ) );
    CHECK( d2->volume(mycoord)==d1->volume(mycoord) );
    CHECK( d2->first_index_r(mycoord)==(d1->first_index_r(mycoord)-1) );
  }

  SECTION( "divide operate" ) {
    auto div2 = ioperator("/2");
    REQUIRE_NOTHROW( d2 = d1->operate(div2) );
    CHECK( d2->volume(mycoord)==(d1->volume(mycoord)/2) );
    CHECK( d2->first_index_r(mycoord)==(d1->first_index_r(mycoord)/2) );
  }

  SECTION( "multiply" ) {
    auto mul2 = ioperator("*2");
    REQUIRE_NOTHROW( d2 = d1->operate(mul2) );
    CHECK( d2->first_index_r(mycoord)==d1->first_index_r(mycoord)*2 );
    CHECK( d2->last_index_r(mycoord)==d1->last_index_r(mycoord)*2 );
    CHECK( d2->volume(mycoord)==d1->volume(mycoord) );
    CHECK( (d2->last_index_r(mycoord)-d2->first_index_r(mycoord))==
	   (d1->last_index_r(mycoord)-d1->first_index_r(mycoord))*2 );
  }
}

TEST_CASE( "merge parallel indexstruct","[indexstruct][merge][55]" ) {

  int localsize = 11,gsize = ntids*localsize;
  parallel_structure pstr,qstr,zstr;
  REQUIRE_NOTHROW( pstr = parallel_structure(decomp) );
  REQUIRE_NOTHROW( pstr.create_from_global_size( gsize ) );
  CHECK( pstr.get_dimension_structure(0)->local_size(mytid)==localsize );
  
  SECTION( "shift right" ) {
    SECTION( "by base" ) {
      REQUIRE_NOTHROW( qstr = pstr.operate_base( ioperator("shift",1) ) );
    }
    SECTION( "by struct" ) {
      REQUIRE_NOTHROW( qstr = pstr.operate( ioperator("shift",1) ) );
    }
    CHECK( pstr.get_dimension_structure(0)->local_size(mytid)==localsize );
    CHECK( qstr.get_dimension_structure(0)->local_size(mytid)==localsize );
    CHECK( pstr.get_dimension_structure(0)->first_index(mytid)==(mytid*localsize) );
    CHECK( qstr.get_dimension_structure(0)->first_index(mytid)==(mytid*localsize+1) );

    REQUIRE_NOTHROW( zstr = pstr.struct_union( qstr ) );
    CHECK( zstr.get_dimension_structure(0)->local_size(mytid)==localsize+1 );
    CHECK( zstr.get_dimension_structure(0)->first_index(mytid)==(mytid*localsize) );
  }

  SECTION( "shift left" ) {
    SECTION( "by base" ) {
      REQUIRE_NOTHROW( qstr = pstr.operate_base( ioperator("shift",-1) ) );
    }
    SECTION( "by struct" ) {
      REQUIRE_NOTHROW( qstr = pstr.operate( ioperator("shift",-1) ) );
    }
    CHECK( pstr.get_dimension_structure(0)->local_size(mytid)==localsize );
    CHECK( qstr.get_dimension_structure(0)->local_size(mytid)==localsize );
    CHECK( pstr.get_dimension_structure(0)->first_index(mytid)==(mytid*localsize) );
    CHECK( qstr.get_dimension_structure(0)->first_index(mytid)==(mytid*localsize-1) );

    REQUIRE_NOTHROW( zstr = pstr.struct_union( qstr ) );
    CHECK( zstr.get_dimension_structure(0)->local_size(mytid)==localsize+1 );
    CHECK( zstr.get_dimension_structure(0)->first_index(mytid)==(mytid*localsize-1) );
  }
}

TEST_CASE( "merge distributions","[indexstruct][merge][56]" ) {

  int localsize = 11,gsize = ntids*localsize;
  shared_ptr<distribution> pstr,qstr,zstr;
  REQUIRE_NOTHROW( pstr = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,-1,gsize) ) );
  CHECK( pstr->volume(mycoord)==localsize );
  
  SECTION( "shift right" ) {
    REQUIRE_NOTHROW( qstr = pstr->operate_base( ioperator("shift",1) ) );
    CHECK( pstr->volume(mycoord)==localsize );
    CHECK( qstr->volume(mycoord)==localsize );
    CHECK( pstr->first_index_r(mycoord)==(mycoord_coord*localsize) );
    CHECK( qstr->first_index_r(mycoord)==(mycoord_coord*localsize+1) );

    REQUIRE_NOTHROW( zstr = pstr->distr_union( qstr ) );
    CHECK( zstr->volume(mycoord)==localsize+1 );
    CHECK( zstr->first_index_r(mycoord)==(mycoord_coord*localsize) );
  }

  SECTION( "shift left" ) {
    REQUIRE_NOTHROW( qstr = pstr->operate_base( ioperator("shift",-1) ) );
    CHECK( pstr->volume(mycoord)==localsize );
    CHECK( qstr->volume(mycoord)==localsize );
    CHECK( pstr->first_index_r(mycoord)==(mycoord_coord*localsize) );
    CHECK( qstr->first_index_r(mycoord)==(mycoord_coord*localsize-1) );

    REQUIRE_NOTHROW( zstr = pstr->distr_union( qstr ) );
    CHECK( zstr->volume(mycoord)==localsize+1 );
    CHECK( zstr->first_index_r(mycoord)==(mycoord_coord*localsize-1) );
  }
}

TEST_CASE( "divide distributions","[distribution][operate][57]" ) {

  {
    INFO( "Need an even number of processors" );
    int twomod = ntids%2;
    REQUIRE( twomod==0 );
  }

  INFO( "mytid=" << mytid );

  auto twoper = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,2,-1) );
  shared_ptr<distribution> oneper,duplic;

  REQUIRE_NOTHROW( oneper = twoper->operate( ioperator("/2") ) );
  CHECK( oneper->volume(mycoord)==1 );
  CHECK( oneper->first_index_r(mycoord)==mycoord_coord );

  REQUIRE_NOTHROW( duplic = oneper->operate( ioperator("/2") ) );
  CHECK( duplic->volume(mycoord)==1 );
  CHECK( duplic->first_index_r(mycoord)==(mycoord_coord/2) );
}

// TEST_CASE( "operate 2d distributions","[distribution][grid][operate][copy][2d][70][hide]" ) {
//   index_int nlocal=3;
//   shared_ptr<distribution> d1;
//   shared_ptr<distribution> d2;

//   REQUIRE( (ntids%2)==0 ); // sorry, need even number
//   mpi_environment *gridenv;
//   REQUIRE_NOTHROW( gridenv = new mpi_environment( *env ) );
//   REQUIRE_NOTHROW( gridenv->set_grid_2d( ntids/2,2 ) );
//   CHECK_NOTHROW( d1 = shared_ptr<distribution>(make_shared<mpi_block_distribution>
// 	     ( gridenv,  nlocal,nlocal,-1,-1 ) ) );

//   SECTION( "operated copy" ) {
//     gridoperator *is = new gridoperator(">>1",">>1");
//     parallel_indexstruct *ps,*po;
//     index_int s;

//     // shifted by operate
//     REQUIRE_NOTHROW( d2 = d1->operate( d1,is ) );
//     REQUIRE_NOTHROW( s = d2->volume(mycoord) );
//     CHECK( s==nlocal*nlocal );

//     REQUIRE_NOTHROW( delete is );
//   }

//   delete gridenv;
// }

