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
 **** unit tests for the sparse matrix package
 **** (most tests do not actually rely on MPI)
 ****
 ****************************************************************/

#include <stdlib.h>
#include <math.h>

#include "catch2/catch_all.hpp"

#include "mpi_base.h"
#include "mpi_ops.h"
#include "mpi_static_vars.h"

using fmt::format;
using fmt::print;

using std::make_shared;
using std::shared_ptr;

using std::string;
using std::vector;

TEST_CASE( "element","[element][1]" ) {
  sparse_element *e;
  REQUIRE_NOTHROW( e = new sparse_element(3,5.41) );
  CHECK( e->get_index()==3 );
  CHECK( e->get_value()==Approx(5.41) );

  sparse_element *f;
  REQUIRE_NOTHROW( f = new sparse_element(2,7.3) );
  CHECK( *f<*e );
  CHECK( !(*e<*f) );
}

TEST_CASE( "row","[2]" ) {
  sparse_row *r;
  REQUIRE_NOTHROW( r = new sparse_row() );
  SECTION( "regular" ) {
    REQUIRE_NOTHROW( r->add_element(1,8.5) );
    REQUIRE_NOTHROW( r->add_element(2,7.5) );
  }
  SECTION( "reverse" ) {
    REQUIRE_NOTHROW( r->add_element(2,7.5) );
    REQUIRE_NOTHROW( r->add_element(1,8.5) );
  }
  CHECK( r->size()==2 );
  CHECK( r->at(0) < r->at(1) );
  CHECK( r->row_sum()==Approx(16.) );

  shared_ptr<indexstruct> i;
  REQUIRE_NOTHROW( i = r->all_indices() );
  CHECK( i->local_size()==2 );
  CHECK( i->contains_element(1) );
  CHECK( i->contains_element(2) );
}

TEST_CASE( "irow","[3]" ) {
  sparse_rowi *r;
  REQUIRE_NOTHROW( r = new sparse_rowi(5) );
  SECTION( "regular" ) {
    REQUIRE_NOTHROW( r->add_element(11,8.5) );
    REQUIRE_NOTHROW( r->add_element(12,7.5) );
  }
  SECTION( "reverse" ) {
    REQUIRE_NOTHROW( r->add_element(12,7.5) );
    REQUIRE_NOTHROW( r->add_element(11,8.5) );
  }
  CHECK( r->size()==2 );
  CHECK( r->at(0) < r->at(1) );
  CHECK( r->row_sum()==Approx(16.) );

  shared_ptr<indexstruct> i;
  REQUIRE_NOTHROW( i = r->all_indices() );
  CHECK( i->local_size()==2 );
  CHECK( i->contains_element(11) );
  CHECK( i->contains_element(12) );
}

TEST_CASE( "sparse inprod","[4]" ) {
  INFO( "mytid=" << mytid );
  int nlocal = 10;
  sparse_row r;
  REQUIRE_NOTHROW( r = sparse_row() );
  for (index_int i=0; i<nlocal; i++ ) {
    auto iglobal = i+mytid*nlocal;
    REQUIRE_NOTHROW( r.add_element( iglobal,1.*iglobal ) );
  }
  shared_ptr<distribution> d;
  REQUIRE_NOTHROW
    ( d = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,nlocal,-1) ) );
  REQUIRE( d->has_type_locally_contiguous() );
  REQUIRE( d->volume(mycoord)==nlocal );
  auto o = shared_ptr<object>( make_shared<mpi_object>(d) );
  REQUIRE_NOTHROW( o->allocate() );
  data_pointer data;
  REQUIRE_NOTHROW( data = o->get_data(mycoord) );
  for (index_int i=0; i<nlocal; i++ )
    data.at(i) = i;
  double inprod;
  REQUIRE_NOTHROW( inprod = r.inprod(o,mycoord) );

  int nsum = 0;
  for (int i=0; i<nlocal; i++) nsum += i*(mytid*nlocal+i);
  CHECK( inprod==nsum );
}

TEST_CASE( "uni-processor matrix","[10]" ) {
  sparse_matrix m; int nrows;

  string how;
  { how = "indexed";
    REQUIRE_NOTHROW( m = sparse_matrix(indexed_indexstruct( {1,3,7} ) ) );
    CHECK( m.local_size()==3 );
  }
  INFO( format("mycoord={}",mycoord.as_string()) );
  INFO( format("{}",m.as_string()) );

  REQUIRE_NOTHROW( m.add_element(1,2,3.) );
  REQUIRE_NOTHROW( m.add_element(3,5,8.) );
  REQUIRE_NOTHROW( m.add_element(3,3,7.) );
  REQUIRE_NOTHROW( m.add_element(7,3,9.) );

  CHECK( m.nnzeros()==4 );
  CHECK( m.local_size()==3 );
  CHECK( m.has_element(3,5) );
  CHECK( !m.has_element(3,4) );

  shared_ptr<indexstruct> idx;
  REQUIRE_NOTHROW( idx = m.row_indices() );
  CHECK( idx->local_size()==3 );
  CHECK( idx->first_index()==1 );
  CHECK( idx->last_index()==7 );

  int s;
  REQUIRE_NOTHROW( s = m.row_sum(1) );
  CHECK( s==Approx(3.) );
  REQUIRE_THROWS( s = m.row_sum(2) );
  REQUIRE_NOTHROW( s = m.row_sum(3) );
  CHECK( s==Approx(15.) );
  REQUIRE_THROWS( s = m.row_sum(4) );

  shared_ptr<indexstruct> i;
  REQUIRE_NOTHROW( i = m.all_columns() );
  INFO( "columns: " << i->as_string() );
  CHECK( i->local_size()==3 );
  CHECK( i->contains_element(2) );
  CHECK( i->contains_element(3) );
  CHECK( i->contains_element(5) );
}

TEST_CASE( "index pattern collapsing","[pattern][11]" ) {
  index_int localsize=SMALLBLOCKSIZE+2,gsize=ntids*localsize;
  CHECK( decomp.get_dimensionality()==1 );
  shared_ptr<distribution> d;
  REQUIRE_NOTHROW
    ( d = shared_ptr<distribution>( make_shared<mpi_block_distribution>
				    (decomp,gsize) ) );
  {
    processor_coordinate pcoord;
    REQUIRE_NOTHROW( pcoord = d->proc_coord() );
    CHECK( pcoord==mycoord );
  
    shared_ptr<indexstruct> pstruct;
    REQUIRE_NOTHROW( pstruct = d->get_processor_structure(pcoord)->get_component(0) );
    CHECK( pstruct->is_contiguous() );
  }
  mpi_sparse_matrix p;
  REQUIRE_NOTHROW( p = mpi_sparse_matrix(d,d->global_volume()) );

  REQUIRE_NOTHROW( p.set_jrange(0,gsize-1) );
  shared_ptr<indexstruct> columns;
  int shift;

  SECTION( "shift 1 makes contiguous" ) {
    shift = 1;
    for (index_int i=d->first_index_r(mycoord)[0]; i<=d->last_index_r(mycoord)[0]; i++) {
      INFO( format("inserting {},{} with jrange {}-{}",i,i+shift,0,gsize-1) );
      CHECK_NOTHROW( p.add_element(i,i) );
      if (i+shift<gsize) {
	CHECK_NOTHROW( p.add_element(i,i+shift) );
      } else {
	CHECK_THROWS( p.add_element(i,i+shift) );
      }
    }

    REQUIRE_NOTHROW( columns = p.all_columns_from(d->get_processor_structure(mycoord)) );
    INFO( "matrix column: " << columns->as_string() );
    CHECK( columns->is_contiguous() );
    CHECK( columns->first_index()==d->first_index_r(mycoord)[0] );
    if (mytid==ntids-1) {
      INFO( "last proc" );
      CHECK( columns->last_index()==d->last_index_r(mycoord)[0] );
      CHECK( columns->local_size()==localsize );
    } else {
      INFO( "arbitrary proc" );
      CHECK( columns->last_index()==(d->last_index_r(mycoord)[0]+shift) );
      CHECK( columns->local_size()==(localsize+1) );
    }
  }

  SECTION( "shift 2 makes gap" ) {
    shift = 2;
    for (index_int i=d->first_index_r(mycoord)[0]; i<=d->last_index_r(mycoord)[0]; i++) {
      INFO( "row " << i );
      CHECK_NOTHROW( p.add_element(i,i) );
      if (i+shift<gsize) {
	CHECK_NOTHROW( p.add_element(i,i+shift) );
      } else {
	CHECK_THROWS( p.add_element(i,i+shift) );
      }
    }

    REQUIRE_NOTHROW( columns = p.all_columns_from(d->get_processor_structure(mycoord)) );
    CHECK( columns->first_index()==d->first_index_r(mycoord)[0] );
    if (mytid==ntids-1) {
      INFO( "last proc" );
      CHECK( columns->last_index()==d->last_index_r(mycoord)[0] );
      CHECK( columns->local_size()==localsize );
    } else {
      INFO( "arbitrary proc" );
      CHECK( columns->last_index()==(d->last_index_r(mycoord)[0]+shift) );
      CHECK( columns->local_size()==(localsize+shift) );
    }
  }
}

TEST_CASE( "Sparse matrix tests","[sparse][12]" ) {
  INFO( "mytid: " << mytid );
  int nlocal = 10;
  sparse_matrix pat; sparse_matrix mat;

  SECTION( "pattern from indexstruct" ) {
    REQUIRE_NOTHROW( pat = sparse_matrix( contiguous_indexstruct(0,nlocal-1) ) );
    REQUIRE_NOTHROW( pat.set_jrange(0,nlocal-1) );
    REQUIRE_THROWS ( pat.add_element(nlocal,2) );
    REQUIRE_NOTHROW( pat.add_element(0,0) );
    REQUIRE_NOTHROW( pat.add_element(0,3) );
    REQUIRE_NOTHROW( pat.add_element(0,1) );
    bool has{false};
    REQUIRE_NOTHROW( has = pat.has_element(-1,0) );
    CHECK( !has );
    REQUIRE_NOTHROW( has = pat.has_element(nlocal,0) );
    CHECK( !has );
    REQUIRE_NOTHROW( pat.has_element(0,2*nlocal) );
    CHECK( pat.has_element(0,0) );
    CHECK( pat.has_element(0,1) );
    CHECK( !pat.has_element(0,2) );
    CHECK( pat.has_element(0,3) );
    CHECK( !pat.has_element(0,4) );
    REQUIRE_NOTHROW( pat.add_element(nlocal-1,nlocal-1) );
  }
  SECTION( "zero-based set of rows" ) {
    parallel_indexstruct idx;
    REQUIRE_NOTHROW( idx = parallel_indexstruct( ntids ) );
    REQUIRE_NOTHROW( idx.create_from_uniform_local_size(nlocal) );
    REQUIRE_NOTHROW( pat = sparse_matrix( idx,mytid ) );
    index_int f,l;
    REQUIRE_NOTHROW( f = idx.first_index(mytid) );
    REQUIRE_NOTHROW( l = idx.last_index(mytid) );
    if (mytid!=1)
      REQUIRE_THROWS ( pat.add_element(nlocal,2) );
    else
      REQUIRE_NOTHROW ( pat.add_element(nlocal,2) );
    if (mytid==0)
      REQUIRE_NOTHROW( pat.add_element(0,0) );
    else
      REQUIRE_THROWS( pat.add_element(0,0) );
  }
  SECTION( "sparse matrix from indexstruct" ) {
    double v;
    REQUIRE_NOTHROW( mat = sparse_matrix( contiguous_indexstruct(0,nlocal-1) ) );
    REQUIRE_THROWS ( mat.add_element(nlocal,2,2.) );
    REQUIRE_NOTHROW( mat.add_element(0,0,0.) );
    REQUIRE_NOTHROW( mat.add_element(0,3,3.) );
    REQUIRE_NOTHROW( mat.add_element(0,1,1.) );
    bool has{true};
    REQUIRE_NOTHROW( has = mat.has_element(-1,0) );
    CHECK( !has );
    REQUIRE_NOTHROW( has = mat.has_element(nlocal,0) );
    CHECK( !has );
    bool has_j_range{false};
    SECTION ( "no jrange" ) {
    }
    SECTION ( "j range defined" ) {
      REQUIRE_NOTHROW( mat.set_jrange(0,3*nlocal) );
    }
    REQUIRE_NOTHROW( has = mat.has_element(0,2*nlocal) );
    CHECK( has==has_j_range );

    CHECK( mat.has_element(0,0) );
    CHECK( mat.has_element(0,1) );
    // REQUIRE_NOTHROW( v = mat.get_element(0,1) );
    // CHECK( v==Approx(1.) );
    CHECK( !mat.has_element(0,2) );

    CHECK( mat.has_element(0,3) );
    // REQUIRE_NOTHROW( v = mat.get_element(0,3) );
    // CHECK( v==Approx(3.) );

    CHECK( !mat.has_element(0,4) );
    REQUIRE_NOTHROW( mat.add_element(nlocal-1,nlocal-1,nlocal-1.) );
  }
}

TEST_CASE( "single element matrix multiply","[20]" ) {
  int nlocal=1; double s;
  sparse_matrix m;

  INFO( format("mytid: {}",mytid) );
  REQUIRE_NOTHROW( m = sparse_matrix( contiguous_indexstruct(mytid) ) );

  REQUIRE_NOTHROW( m.add_element( mytid,mytid,1. ) );
  INFO( format("matrix: {}",m.as_string()) );

  auto
    d = shared_ptr<distribution>( make_shared<mpi_block_distribution>
				  (decomp,1,-1) );
  auto in = shared_ptr<object>( make_shared<mpi_object>(d) ),
    out = shared_ptr<object>( make_shared<mpi_object>(d) );
  in->allocate();
  auto data = in->get_data(mycoord);
  for (index_int i=0; i<nlocal; i++ )
    data.at(i) = 1.;

  REQUIRE_NOTHROW( m.multiply(in,out,mycoord) );
  auto val = out->get_data(mycoord);
  CHECK( val.at(0)==Approx( 1. ) );
}

TEST_CASE( "square sparse matprod","[21]" ) {
  /*
   * in this test we still do a sequential product
   * so the input should be local
   */
  int nlocal,nglobal; double s;
  sparse_matrix m;
  shared_ptr<distribution> d;
  string path;
  SECTION( "single element" ) { path = "single element per proc";
    nlocal = 1; nglobal = ntids;
    REQUIRE_NOTHROW( d = shared_ptr<distribution>
		     ( make_shared<mpi_block_distribution>(decomp,nlocal,-1) ) );
    REQUIRE_NOTHROW( m = mpi_sparse_matrix(d) );
    //s = 1.;
    REQUIRE_NOTHROW( m.add_element( mytid,mytid,1. ) );
  }
  SECTION( "one row on each processor" ) { path = "one row per proc";
    nlocal = 1; nglobal = ntids*nlocal;
    REQUIRE_NOTHROW( d = shared_ptr<distribution>
		     ( make_shared<mpi_block_distribution>(decomp,nlocal,-1) ) );
    REQUIRE_NOTHROW( m = mpi_sparse_matrix(d,nglobal) );
    //s = nglobal;
    for (index_int i=mytid*nlocal; i<(mytid+1)*nlocal; i++ )
      REQUIRE_NOTHROW( m.add_element( mytid,i,1.) );
  }
  INFO( format("Sparse matprod, {}",path) );

  auto in = shared_ptr<object>( new mpi_object(d) ),
    out = shared_ptr<object>( new mpi_object(d) );
  in->allocate(); auto data = in->get_data(mycoord);
  for (index_int i=0; i<nlocal; i++ )
    data.at(i) = mytid+1;

  REQUIRE_NOTHROW( m.multiply(in,out,mycoord) );
  auto val = out->get_data(mycoord);
  CHECK( val.at(0)==Approx( (mytid+1)*nlocal*(nlocal+1)/2 ) );
}

TEST_CASE( "sparse matprod","[25]" ) {
  int ncols = 10;
  sparse_matrix m;
  auto d = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,1,-1) );
  // matrix with just one row on each processor, ncols unique columns
  REQUIRE_NOTHROW( m = mpi_sparse_matrix(d,ncols*ntids) );
  {
    shared_ptr<indexstruct> mcolumns;
    //REQUIRE_NOTHROW( mcolumns = m.all_columns() ); // VLE does not yet work in MPI mode
  }
  double s{0};
  for (index_int i=0; i<ncols; i+=2 ) {
    double v = i+2.;
    m.add_element( mytid,ncols*mytid+i,v );
    s += (mytid+1)*v;
  }

  auto
    din = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,ncols,-1) ),
    dout = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,1,-1) );
  auto in = shared_ptr<object>( new mpi_object(din) ),
    out = shared_ptr<object>( new mpi_object(dout) );
  REQUIRE_NOTHROW( in->allocate() );
  data_pointer data,val;
  REQUIRE_NOTHROW( data = in->get_data(mycoord) );
  for (index_int i=0; i<ncols; i++ )
    data.at(i) = mytid+1;

  REQUIRE_NOTHROW( m.multiply(in,out,mycoord) );
  REQUIRE_NOTHROW( val = out->get_data(mycoord) );
  CHECK( val.at(0)==Approx(s) );
}

TEST_CASE( "mpi sparse matprod","[31]" ) {
  index_int ncols = 10, gsize = ncols*ntids;
  shared_ptr<sparse_matrix> m;
  auto
    din = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,ncols,-1) ),
    dout = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,1,-1) );

  // matrix with just one row on each processor, ncols unique columns
  REQUIRE_NOTHROW( m = shared_ptr<sparse_matrix>( new mpi_sparse_matrix(dout,gsize) ) );
  double s{0};
  // every row of the matrix is completely full
  for (int tid=0; tid<ntids; tid++ ) {
    for (index_int i=0; i<ncols; i++ ) {
      index_int gi = tid*ncols+i;
      REQUIRE( gi<gsize );
      double v = gi;
      REQUIRE_NOTHROW( m->add_element( mytid,gi,v ) );
      s += gi * (tid+1);
    }
  }

  auto in = shared_ptr<object>( new mpi_object(din) ),
    out = shared_ptr<object>( new mpi_object(dout) );
  REQUIRE_NOTHROW( in->set_name("31-in") );
  REQUIRE_NOTHROW( out->set_name("31-out") );
  REQUIRE_NOTHROW( in->allocate() );
  data_pointer data,val;
  REQUIRE_NOTHROW( data = in->get_data(mycoord) );
  CHECK( in->volume(mycoord)==ncols );
  for (index_int i=0; i<ncols; i++ ) {
    data.at(i) = mytid+1;
  }

  shared_ptr<kernel> spmvp;
  REQUIRE_NOTHROW( spmvp = shared_ptr<kernel>( new mpi_spmvp_kernel(in,out,m) ) );
  REQUIRE( spmvp->get_dependencies().size()==1 );
  //print("spmvp (kernel ptr) dependency type: {}\n",(int)(spmvp->last_dependency().get_type()));
  REQUIRE( spmvp->last_dependency().has_type_pattern() );

  algorithm multiply;
  REQUIRE_NOTHROW( multiply = mpi_algorithm(decomp) );
  REQUIRE_NOTHROW( multiply.add_kernel( shared_ptr<kernel>( new mpi_origin_kernel(in) ) ) );
  REQUIRE_NOTHROW( multiply.add_kernel( spmvp ) );
  REQUIRE_NOTHROW( multiply.analyze_dependencies(true) );

  shared_ptr<object> beta;
  REQUIRE_NOTHROW( beta = spmvp->get_beta_object() );
  shared_ptr<multi_indexstruct> mybeta;
  REQUIRE_NOTHROW( mybeta = beta->get_processor_structure(mycoord) );
  INFO( format("My beta struct: {}",mybeta->as_string() ) );

  REQUIRE_NOTHROW( multiply.execute() );
  REQUIRE_NOTHROW( val = out->get_data(mycoord) );
  CHECK( val.at(0)==Approx(s) );
}

TEST_CASE( "mpi matrix construction by one-sided","[41]" ) {
  index_int ncols = 1, gsize = ncols*ntids;
  shared_ptr<sparse_matrix> m;
  auto
    d = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,ncols,-1) );

  // matrix with just one row on each processor, ncols unique columns
  REQUIRE_NOTHROW( m = shared_ptr<sparse_matrix>( new mpi_sparse_matrix(d,gsize) ) );
  REQUIRE_NOTHROW( m->set_global() );

}
