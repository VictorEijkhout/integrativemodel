/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-9
 ****
 **** Unit tests for the OpenMP product backend of IMP
 **** based on the CATCH framework (https://github.com/philsquared/Catch)
 ****
 **** unit tests for the sparse matrix package
 **** (most tests do not actually rely on OMP)
 ****
 ****************************************************************/

#include <stdlib.h>
#include <math.h>

#include "catch.hpp"

#include "omp_base.h"
#include "omp_static_vars.h"

using fmt::format;
using fmt::print;

using std::shared_ptr;
using std::string;

#include <vector>
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
  sparse_row r;
  //  REQUIRE_NOTHROW( r = new sparse_row() );
  SECTION( "regular" ) {
    REQUIRE_NOTHROW( r.add_element(1,8.5) );
    REQUIRE_NOTHROW( r.add_element(2,7.5) );
  }
  SECTION( "reverse" ) {
    REQUIRE_NOTHROW( r.add_element(2,7.5) );
    REQUIRE_NOTHROW( r.add_element(1,8.5) );
  }
  CHECK( r.size()==2 );
  CHECK( ( r.at(0) < r.at(1) ) );
  CHECK( r.row_sum()==Approx(16.) );

  shared_ptr<indexstruct> i;
  REQUIRE_NOTHROW( i = r.all_indices() );
  CHECK( i->local_size()==2 );
  CHECK( i->contains_element(1) );
  CHECK( i->contains_element(2) );
}

TEST_CASE( "row iterating","[2]" ) {
  sparse_row r;
  REQUIRE_NOTHROW( r.add_element(2,8.5) );
  REQUIRE_NOTHROW( r.add_element(5,7.5) );
  INFO( format("row with 2 and 5: {}",r.as_string()) );
  int cnt{0};
  for ( auto e : r ) {
    INFO( format("element {}: {}",cnt,e.as_string()) );
    auto i = e.get_index();
    auto v = e.get_value();
    if (cnt==0) {
      CHECK( i==2 );
      CHECK( v==Approx(8.5) );
    } else {
      CHECK( i==5 );
      CHECK( v==Approx(7.5) );
    }
    cnt++;
  }
  CHECK( cnt==2 );
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
  CHECK( (r->at(0) < r->at(1) ) );
  CHECK( r->row_sum()==Approx(16.) );

  shared_ptr<indexstruct> i;
  REQUIRE_NOTHROW( i = r->all_indices() );
  CHECK( i->local_size()==2 );
  CHECK( i->contains_element(11) );
  CHECK( i->contains_element(12) );
}

TEST_CASE( "sparse inprod","[4]" ) {
  int nlocal = 10;
  auto d = shared_ptr<distribution>( new omp_block_distribution(decomp,nlocal,-1) );
  CHECK( d->global_volume()==ntids*nlocal );
  auto longvector = shared_ptr<object>( new omp_object(d) );
  longvector->set_name("unit4vector");
  REQUIRE_NOTHROW( longvector->allocate() );
  REQUIRE_NOTHROW( longvector->has_type_blocked() );
  CHECK( longvector->global_volume()==ntids*nlocal );
  {
    //shared_ptr<vector<double>> data; // use get_numa_data_pointer?
    auto p = processor_coordinate_zero(1);
    decltype(longvector->get_data(p)) data;
    CHECK( p.get_dimensionality()==1 );
    CHECK( longvector->get_dimensionality()==1 );
    REQUIRE_NOTHROW( data = longvector->get_data(p) );
    for (index_int i=0; i<ntids*nlocal; i++ )
      data.at(i) = 1.;
  }

  sparse_row onerow;
  double inprod;
  processor_coordinate mycoord;
  for (int mytid=0; mytid<ntids; mytid++) {
    INFO( "thread: " << mytid );
    REQUIRE_NOTHROW( mycoord = decomp.coordinate_from_linear(mytid) );
    REQUIRE_NOTHROW( onerow = sparse_row() );
    for (index_int i=0; i<nlocal; i+=2 )
      REQUIRE_NOTHROW( onerow.add_element( mytid*nlocal+i,mytid*nlocal+i+1. ) );
    REQUIRE_NOTHROW( inprod = onerow.inprod(longvector,mycoord) );
    double s = 0;
    for (int i=0; i<nlocal; i+=2)
      s += mytid*nlocal+i+1.;
    CHECK( inprod==s );
  }
}

TEST_CASE( "matrix","[10]" ) {
  sparse_matrix m; int nrows;
  
  string how;
  SECTION( "contiguous row numbers" ) { how = "contiguous";
    nrows = 8;
    REQUIRE_NOTHROW( m = sparse_matrix(8) );
    CHECK( m.local_size()==8 );
  }
  SECTION( "indexed row numbers" ) { how = "indexed";
    nrows = 4;
    REQUIRE_NOTHROW( m = sparse_matrix( indexed_indexstruct( {1,3,6,7} ) ) );
    CHECK( m.local_size()==4 );
  }

  INFO( format("matrix {}: {}",how,m.as_string()) );

  REQUIRE_NOTHROW( m.add_element(1,2,3.) );
  REQUIRE_NOTHROW( m.add_element(3,5,8.) );
  REQUIRE_NOTHROW( m.add_element(3,3,7.) );
  REQUIRE_NOTHROW( m.add_element(7,3,9.) );
  
  CHECK( m.nnzeros()==4 );
  CHECK( m.local_size()==nrows );
  CHECK( m.has_element(3,5) );
  CHECK( !m.has_element(3,4) );

  shared_ptr<indexstruct> idx;
  REQUIRE_NOTHROW( idx = m.row_indices() );
  if (nrows==8) {
    CHECK( idx->local_size()==nrows );
    CHECK( idx->first_index()==0 );
    CHECK( idx->last_index()==7 );
  } else {
    CHECK( idx->first_index()==1 );
    CHECK( idx->last_index()==7 );
  }

  int s;
  REQUIRE_NOTHROW( s = m.row_sum(1) );
  CHECK( s==Approx(3.) );
  if (nrows==8) {
  } else {
    REQUIRE_THROWS( s = m.row_sum(2) );
    REQUIRE_THROWS( s = m.row_sum(4) );
  }
  REQUIRE_NOTHROW( s = m.row_sum(3) );
  CHECK( s==Approx(15.) );

  {
    shared_ptr<sparse_rowi> r;
    shared_ptr<indexstruct> indices;
    // a row that has been filled in
    REQUIRE_NOTHROW( r = m.get_row_by_global_number(1) );
    CHECK( r->get_row_number()==1 );
    REQUIRE_NOTHROW( indices = r->all_indices() );
    CHECK( indices->local_size()==1 );
    // an empty row
    REQUIRE_NOTHROW( r = m.get_row_by_global_number(6) );
    CHECK( r!=nullptr );
    CHECK( r->get_row_number()==6 );
    REQUIRE_NOTHROW( indices = r->all_indices() );
    CHECK( indices->local_size()==0 );
  }
  shared_ptr<indexstruct> i;
  REQUIRE_NOTHROW( i = m.all_columns() );
  CHECK( i->local_size()==3 );
  CHECK( i->contains_element(2) );
  CHECK( i->contains_element(3) );
  CHECK( i->contains_element(5) );
}

TEST_CASE( "sparse matprod","[11]" ) {
  int nlocal = 10;
  sparse_matrix m;
  auto
    din = shared_ptr<distribution>( new omp_block_distribution(decomp,nlocal,-1) ),
    dout = shared_ptr<distribution>( new omp_block_distribution(decomp,1,-1) );
  int nglobal_in = din->global_volume(), nglobal_out = dout->global_volume();

  const char *path;
  SECTION( "bare matrix" ) { path = "direct";
    REQUIRE_NOTHROW( m = sparse_matrix(nglobal_out,nglobal_in) );
  }
  // SECTION( "omp matrix class" ) { path = "derived";
  //   REQUIRE_NOTHROW( m = omp_sparse_matrix(dout,nlocal*ntids) );
  // }
  INFO( path << " matrix creation" );
  // matrix with just one row on each processor
  for (int mytid=0; mytid<ntids; mytid++)
    for (index_int icol=0; icol<nlocal; icol+=2 ) {
      index_int i=mytid,j=mytid*nlocal+icol;
      //print("adding element {},{}\n",i,j);
      REQUIRE_NOTHROW( m.add_element( i,j,i+2. ) );
    }
  return;

  auto in = shared_ptr<object>( new omp_object(din) ),
    out = shared_ptr<object>( new omp_object(dout) );
  in->allocate();
  auto data = in->get_data(processor_coordinate_zero(1));
  for (int mytid=0; mytid<ntids; mytid++)
    for (index_int i=0; i<nlocal; i++ )
      data.at(mytid*nlocal+i) = mytid+1;

  for (int mytid=0; mytid<ntids; mytid++) {
    INFO( "multiply on " << mytid );
      processor_coordinate mycoord;
      REQUIRE_NOTHROW( mycoord = decomp.coordinate_from_linear(mytid) );
    REQUIRE_NOTHROW( m.multiply(in,out,mycoord) );
    auto valdata = out->get_data(processor_coordinate_zero(1));
    int n2 = nlocal/2;
    CHECK( valdata.at(mytid)==Approx( (mytid+1)*n2*(n2+1) ) );
  }
}

