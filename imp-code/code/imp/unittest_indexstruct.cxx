/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2023
 ****
 **** Unit tests for the IMP
 **** based on the CATCH framework (https://github.com/philsquared/Catch)
 ****
 **** unit tests for coordinates and indexstructs
 **** (tests do not actually rely on MPI)
 ****
 ****************************************************************/

#include <cmath>

#include "catch2/catch_all.hpp"

#include "imp_coord.h"
#include "indexstruct.hpp"

using fmt::memory_buffer;
using fmt::format,fmt::print,fmt::format_to,fmt::to_string;
using std::shared_ptr,std::make_shared;
using std::string;
using std::vector,std::array;

TEST_CASE( "coordinates","[01]" ) {
  coordinate<int,1> justfive; justfive.set( 5 );
  CHECK( justfive.dimensionality()==1 );
  CHECK( justfive[0]==5 );

  coordinate<int,2> twofives( {5,5} );
  CHECK( twofives.dimensionality()==2 );
  CHECK( twofives[0]==5 );

  coordinate<int,2> fours( {4,4} );
  coordinate<int,2> five1( {5,1} );
  REQUIRE( twofives==twofives );
  REQUIRE( not (twofives==five1) );
  REQUIRE( (twofives!=fours) );
  REQUIRE( (twofives!=five1) );
}

TEST_CASE( "linear location","[02]" ) {
  coordinate<int,1> five; five.set(5);
  coordinate<int,1> two; two.set(2);
  REQUIRE_THROWS( two.linear_location_of(five) );
  int loc;
  REQUIRE_NOTHROW( loc = five.linear_location_of(two) );
  REQUIRE( loc==2 );
}

TEST_CASE( "contiguous indexstruct, construction","[indexstruct][03]" ) {

  REQUIRE_NOTHROW( contiguous_indexstruct<index_int,1>(0,5) );
  contiguous_indexstruct<index_int,1> i1(0,5);

  coordinate<index_int,1> one; one.set(1);
  SECTION( "Basic" ) {
    CHECK( i1.is_contiguous() );
    CHECK( !i1.is_indexed() );
    CHECK( ( i1.first_index()==0 ) );
    CHECK( ( ( i1.last_actual_index()==5 ) ) );
    CHECK( ( ( i1.last_index()==6 ) ) );
    CHECK( i1.volume()==6 );
    CHECK( i1.stride()==one );
    REQUIRE_NOTHROW( i1.linear_location_of(0) );
    CHECK( i1.linear_location_of(0)==0 );
    CHECK( i1.linear_location_of(5)==5 );
    REQUIRE_THROWS( i1.linear_location_of(6) );
  }
  SECTION( "Pimpl" ) {
    indexstructure<index_int,1> ii(i1);
    CHECK( ii.is_contiguous() );
    CHECK( !ii.is_indexed() );
    CHECK( ( ii.first_index()==0 ) );
    CHECK( ( ( ii.last_actual_index()==5 ) ) );
    CHECK( ii.volume()==6 );
    CHECK( ii.stride()==one );
    CHECK( ii.linear_location_of(0)==0 );
    CHECK( ii.linear_location_of(5)==5 );
    REQUIRE_THROWS( ii.linear_location_of(6) );
  }
}

TEST_CASE( "Contiguous indexstruct accretion","[04]" ) {

  REQUIRE_NOTHROW( indexstructure<index_int,1>( contiguous_indexstruct<index_int,1>(2,5) ) );
  indexstructure<index_int,1> i1( contiguous_indexstruct<index_int,1>(2,5) );
  CHECK( i1.volume()==4 );
  CHECK( ( i1.first_index()==2 ) );
  CHECK( ( i1.last_actual_index()==5 ) );

  coordinate<index_int,1> c3; c3.set(3);
  INFO( format("adding 3: {}",c3.as_string()) );
  REQUIRE_NOTHROW( i1.add_element(c3) );
  INFO( format("result: {}",i1.as_string()) );
  CHECK( i1.is_contiguous() );
  CHECK( i1.volume()==4 );
  CHECK( ( i1.first_index()==2 ) );
  CHECK( ( i1.last_actual_index()==5 ) );

  coordinate<index_int,1> c5; c5.set(5);
  INFO( format("adding 5: {}",c5.as_string()) );
  REQUIRE_NOTHROW( i1.add_element(c5) );
  INFO( format("result: {}",i1.as_string()) );
  CHECK( i1.is_contiguous() );
  CHECK( i1.volume()==4 );
  CHECK( ( i1.first_index()==2 ) );
  CHECK( ( i1.last_actual_index()==5 ) );

  coordinate<index_int,1> c6; c6.set(6);
  INFO( format("adding 6: {}",c6.as_string()) );
  REQUIRE_NOTHROW( i1.add_element(c6) );
  INFO( format("result: {}",i1.as_string()) );
  CHECK( i1.is_contiguous() );
  CHECK( i1.volume()==5 );
  CHECK( ( i1.first_index()==2 ) );
  CHECK( ( i1.last_actual_index()==6 ) );

  coordinate<index_int,1> c1; c1.set(1);
  REQUIRE_NOTHROW( i1.add_element(c1) );
  INFO( format("result: {}",i1.as_string()) );
  CHECK( i1.is_contiguous() );
  CHECK( i1.volume()==6 );
  CHECK( ( i1.first_index()==1 ) );
  CHECK( ( i1.last_actual_index()==6 ) );

  coordinate<index_int,1> c9; c9.set(9);
  REQUIRE_NOTHROW( i1.add_element(c9) );
  INFO( format("result: {}",i1.as_string()) );
  CHECK( !i1.is_strided() );

}

TEST_CASE( "more contiguous","[05]" ) {
  indexstructure<index_int,1> i1( contiguous_indexstruct<index_int,1>(1,7) );
  CHECK( i1.is_contiguous() );
  CHECK( !i1.is_indexed() );
  CHECK( ( i1.first_index()==1 ) );
  CHECK( ( i1.last_actual_index()==7 ) );
  coordinate<index_int,1> one; one.set(1);
  CHECK( i1.stride()==one );
  REQUIRE_THROWS( i1.linear_location_of(0) );
}

TEST_CASE( "strided","[06]" ) {
  indexstructure i1( strided_indexstruct<index_int,1>(2,6,2) );
  CHECK( i1.is_strided() );
  CHECK( !i1.is_contiguous() );
  CHECK( !i1.is_indexed() );
  CHECK( ( i1.first_index()==2 ) );
  CHECK( ( i1.last_actual_index()==6 ) );
  CHECK( i1.volume()==3 );
  coordinate<index_int,1> one; one.set(1);
  CHECK( i1.stride()==one*2 );
  CHECK( i1.linear_location_of(2)==0 );
  REQUIRE_THROWS( i1.linear_location_of(3) );
  CHECK( i1.linear_location_of(4)==1 );
  CHECK( i1.linear_location_of(6)==2 );
  REQUIRE_THROWS( i1.linear_location_of(7) );
  REQUIRE_THROWS( i1.linear_location_of(8) );
}


TEST_CASE( "basic stride tests","[07]" ) {
  indexstructure i1( strided_indexstruct<index_int,1>(4,7,2) );
  INFO( "i1: " << i1.as_string() );
  CHECK( !i1.is_contiguous() );
  CHECK( i1.is_strided() );
  CHECK( !i1.is_indexed() );
  CHECK( ( i1.first_index()==4 ) );
  CHECK( ( i1.last_actual_index()==6 ) );
  CHECK( i1.volume()==2 );
  coordinate<index_int,1> one; one.set(1);
  CHECK( i1.stride()==one*2 );
  CHECK( i1.contains(i1) );
  CHECK( i1.equals(i1) );

  indexstructure i2( contiguous_indexstruct<index_int,1>(4,6) );
  INFO( "i2: " << i2.as_string() );
  CHECK( !i1.equals(i2) );
  CHECK( !i2.equals(i1) );
  CHECK( i2.get_ith_element(1)==5 );
  CHECK_THROWS( i2.get_ith_element(3) );
  
  indexstructure i3( strided_indexstruct<index_int,1>(4,6,2) );
  INFO( "i3: " << i3.as_string() );
  CHECK( i3.equals(i1) );
  CHECK( i1.equals(i3) );
  CHECK( i3.get_ith_element(1)==6 );
  CHECK_THROWS( i3.get_ith_element(3) );

  CHECK( i3.contains_element(4) );
  CHECK( !i3.contains_element(5) );
  CHECK( i3.contains_element(6) );
  CHECK( !i3.contains_element(7) );
}

TEST_CASE( "strided containment" ) {
  indexstructure i1( strided_indexstruct<index_int,1>(4,7,2) );
  indexstructure i2( contiguous_indexstruct<index_int,1>(4,8) );
  indexstructure i3( strided_indexstruct<index_int,1>(4,8,2) );
  indexstructure i4( strided_indexstruct<index_int,1>(4,8,4) );
  coordinate<index_int,1> one; one.set(1);
  SECTION( "containment" ) {
    CHECK( i2.contains(i3) );
    CHECK( !i3.contains(i2) );
    CHECK( !i2.equals(i3) );
    CHECK( i2.contains(i4) );
    CHECK( i3.contains(i4) );
    CHECK( !i4.contains(i2) );
    CHECK( !i4.contains(i3) );
  }
  SECTION( "translation forward" ) {
    REQUIRE_NOTHROW( i1.translate_by(1) );
    CHECK( i1.is_strided() );
    CHECK( !i1.is_contiguous() );
    CHECK( !i1.is_indexed() );
    CHECK( ( i1.first_index()==5 ) );
    CHECK( ( i1.last_actual_index()==7 ) );
    CHECK( i1.volume()==2 );
    CHECK( i1.stride()==one*2 );
  }
  SECTION( "translation backward" ) {
    REQUIRE_NOTHROW( i1.translate_by(-2) );
    CHECK( !i1.is_contiguous() );
    CHECK( i1.is_strided() );
    CHECK( !i1.is_indexed() );
    CHECK( ( i1.first_index()==2 ) );
    CHECK( ( i1.last_actual_index()==4 ) );
    CHECK( i1.volume()==2 );
    CHECK( i1.stride()==one*2 );
  }

  SECTION( "translation through zero" ) {
    REQUIRE_NOTHROW( i1.translate_by(-5) );
    CHECK( i1.is_strided() );
    CHECK( !i1.is_contiguous() );
    CHECK( !i1.is_indexed() );
    CHECK( ( i1.first_index()==-1 ) );
    CHECK( ( i1.last_actual_index()==1 ) );
    CHECK( i1.volume()==2 );
    CHECK( i1.stride()==one*2 );
  }
}

TEST_CASE( "Find","[10]" ) {
  SECTION( "find in contiguous" ) {
    indexstructure i1( contiguous_indexstruct<index_int,1>(8,12) );
    indexstructure i2( strided_indexstruct<index_int,1>(10,12,2) );
    index_int loc;
    REQUIRE_NOTHROW( loc = i1.linear_location_of(i2) );
    CHECK( loc==2 );
    REQUIRE_THROWS( loc = i2.linear_location_of(i1) );
  }
  SECTION( "find in strided" ) {
    indexstructure i1( strided_indexstruct<index_int,1>(8,12,2) );
    indexstructure i2( strided_indexstruct<index_int,1>(10,12,2) );
    indexstructure i3( strided_indexstruct<index_int,1>(11,12,2) );
    index_int loc;
    REQUIRE_NOTHROW( loc = i1.linear_location_of(i2) );
    CHECK( loc==1 );
    REQUIRE_THROWS( loc = i1.linear_location_of(i3) );
    REQUIRE_THROWS( loc = i2.linear_location_of(i1) );
  }
}

TEST_CASE( "indexed indexstruct","[indexstruct][2]" ) {

  SECTION( "basic construction" ) {
    SECTION( "correct" ) {
      int len=3; vector<index_int> idx{1,2,4};
      indexstructure i1{ indexed_indexstruct<index_int,1>(idx) };
      CHECK( !i1.is_contiguous() );
      CHECK( i1.is_indexed() );
      CHECK( ( i1.first_index()==1 ) );
      CHECK( ( i1.last_actual_index()==4 ) );
      CHECK( i1.volume()==len );
    }

    SECTION( "unsorted throws an error" ) {
      int len=4; vector<index_int> idx{1,2,6,4};
      REQUIRE_THROWS( indexstructure{ indexed_indexstruct<index_int,1>(idx) } );
    }

    SECTION( "negative indices allowed" ) {
      int len=3; vector<index_int> idx{-1,2,4};
      indexstructure i1{ indexed_indexstruct<index_int,1>(idx) };
      CHECK( !i1.is_contiguous() );
      CHECK( i1.is_indexed() );
      CHECK( ( i1.first_index()==-1 ) );
      CHECK( ( i1.last_actual_index()==4 ) );
      CHECK( i1.volume()==len );

      indexstructure<index_int,1> ii(i1);
      //REQUIRE_NOTHROW( ii = indexstructure(i1) );
      CHECK( !ii.is_contiguous() );
      CHECK( ii.is_indexed() );
      CHECK( ( ii.first_index()==-1 ) );
      CHECK( ( ii.last_actual_index()==4 ) );
      CHECK( ii.volume()==len );
    }

    SECTION( "gradual construction" ) {
      int len=3; vector<index_int> idx{4,9,20};
      indexstructure i1{ indexed_indexstruct<index_int,1>(idx) };
      CHECK( i1.volume()==3 );
      CHECK( ( i1.first_index()==4 ) );
      CHECK( ( i1.last_actual_index()==20 ) );
      REQUIRE_NOTHROW( i1.add_element(9) );
      CHECK( i1.volume()==3 );
      CHECK( ( i1.first_index()==4 ) );
      CHECK( ( i1.last_actual_index()==20 ) );
      REQUIRE_NOTHROW( i1.add_element(30) );
      CHECK( i1.volume()==4 );
      CHECK( ( i1.first_index()==4 ) );
      CHECK( ( i1.last_actual_index()==30 ) );
      REQUIRE_NOTHROW( i1.add_element(1) );
      CHECK( i1.volume()==5 );
      CHECK( ( i1.first_index()==1 ) );
      CHECK( ( i1.last_actual_index()==30 ) );
      REQUIRE_NOTHROW( i1.add_element(10) );
      CHECK( i1.volume()==6 );
      CHECK( ( i1.first_index()==1 ) );
      CHECK( ( i1.last_actual_index()==30 ) );
    }

    SECTION( "simplificy to strided" ) {
      indexstructure i1{ indexed_indexstruct<index_int,1>( vector<index_int>{2,6,8} ) };
      indexstructure i2{ indexed_indexstruct<index_int,1>( vector<index_int>{3,5,7} ) };
      
      SECTION( "can not make strided" ) {
	// i1 is not strided
	REQUIRE_NOTHROW( i1.make_strided() );
      }
      SECTION( "can make strided" ) {
	// i2 is strided
	REQUIRE_NOTHROW( i2.make_strided() );
      }
    }

    SECTION( "construction and simplify" ) {
      indexstructure i1{ indexed_indexstruct<index_int,1>( vector<index_int>{2,6,8} ) };
      INFO( "i1: " << i1.as_string() );
      indexstructure i2{ indexed_indexstruct<index_int,1>( vector<index_int>{3,5,7} ) };
      INFO( "i2: " << i2.as_string() );
      index_int stride;
      REQUIRE( not i1.is_strided_between_indices(0,2,stride) );

      // SECTION( "try forcing 1" ) {
      // 	REQUIRE_NOTHROW( i1.force_simplify() );
      // 	CHECK( i1.is_indexed() );
      // }
      // SECTION( "force 2" ) {
      // 	REQUIRE_NOTHROW( i1.add_element(4) );
      // 	REQUIRE_NOTHROW( i1.force_simplify() );
      // 	CHECK( i1.is_strided() );
      // 	CHECK( i1.stride()==2 );
      // 	REQUIRE_NOTHROW( i1.struct_union(i2) );
      // 	REQUIRE( i1.volume()==7 );
      // 	REQUIRE_NOTHROW( i1.force_simplify() );
      // 	CHECK( i1.is_strided() );
      // 	CHECK( i1.stride()==1 );
      // 	CHECK( i1.is_contiguous() );
      // }

      // SECTION( "boundary case of empty" ) {
      // 	indexstructure i1{ empty_indexstruct<index_int,1>() };
      // 	REQUIRE_NOTHROW( i1.force_simplify() );
      // }

      // i1 = empty_indexstruct<index_int,1>() );
      // REQUIRE_NOTHROW( i1 = i1.struct_union
      // 		       ( empty_indexstruct<index_int,1>() ) ) );
      // REQUIRE_NOTHROW( i2 = i1.force_simplify() );
    }
  }
}

// TEST_CASE( "more simplify" ) {
//   indexstructure i1( indexed_indexstruct<index_int,1>( vector<index_int>{2,4, 10,12, 15} ) );
//   // indexstruct<index_int,1> i2,i3,i1a;
//   // REQUIRE_NOTHROW( i1a = i1.make_clone() );
//   INFO( format("starting with indexed {}",i1.as_string()) );
//   REQUIRE_NOTHROW( i1.add_in_element(11) );
//   REQUIRE( i1.volume()==6 );
//   INFO( format("add element 11 gives {}",i1.as_string()) );
//   CHECK( i1.is_indexed() );

//   indexstructure i2(i1);
//   REQUIRE_NOTHROW( i2.force_simplify() );
//   INFO( format("simplify to composite {}",i2.as_string()) );
//   CHECK( i2.is_composite() );
//   REQUIRE( i2.volume()==6 );

//   indexstructure i3(i2);
//   REQUIRE_NOTHROW( i3.convert_to_indexed() );
//   INFO( format("back to indexed {}",i3.as_string()) );
//   REQUIRE( i3.volume()==6 );
//   CHECK( i1.equals(i3) );
// }

TEST_CASE( "striding and operations" ) {
  int len=5; vector<index_int> idx{1,2,4,7,9};
  indexstructure i1{ indexed_indexstruct<index_int,1>(idx) };
  
  SECTION( "basic stride tests" ) {
    CHECK( !i1.is_contiguous() );
    CHECK( i1.is_indexed() );
    CHECK( ( i1.first_index()==1 ) );
    CHECK( ( i1.last_actual_index()==9 ) );
    CHECK( i1.volume()==len );

    CHECK( !i1.contains_element(0) );
    CHECK( i1.contains_element(1) );
    CHECK( i1.contains_element(4) );
    CHECK( !i1.contains_element(5) );
    CHECK( !i1.contains_element(6) );
    CHECK( i1.contains_element(7) );

    CHECK( i1.linear_location_of(1)==0 );
    CHECK( i1.linear_location_of(7)==3 );
    REQUIRE_THROWS( i1.linear_location_of(0) );
    REQUIRE_THROWS( i1.linear_location_of(8) );
    CHECK_THROWS( i1.get_ith_element(5) );
    CHECK_NOTHROW( i1.get_ith_element(4) );
    CHECK( i1.get_ith_element(3)==7 );
  }

  SECTION( "translation forward" ) {
    REQUIRE_NOTHROW( i1.translate_by(1) );
    CHECK( !i1.is_contiguous() );
    CHECK( i1.is_indexed() );
    CHECK( ( i1.first_index()==2 ) );
    CHECK( ( i1.last_actual_index()==10 ) );
    CHECK( i1.volume()==len );
  }

  SECTION( "translation through zero" ) {
    REQUIRE_NOTHROW( i1.translate_by(-2) );
    CHECK( !i1.is_contiguous() );
    CHECK( i1.is_indexed() );
    CHECK( ( i1.first_index()==-1 ) );
    CHECK( ( i1.last_actual_index()==7 ) );
    CHECK( i1.volume()==len );
  }
}


// VLE Finding in indexed is just too hard
// TEST_CASE( "find in indexed","[20]" ) {
//   int len=5; vector<index_int> idx{1,2,4,7,9};
//   indexstructure i1{ indexed_indexstruct<index_int,1>(idx) };
//   indexstructure i2{ strided_indexstruct<index_int,1>(2,4,2) };
//   indexstructure i3{ strided_indexstruct<index_int,1>(7,8,2) };

//   index_int loc;
//   REQUIRE_NOTHROW( loc = i1.linear_location_of(i2) );
//   CHECK( loc==1 );
//   REQUIRE_NOTHROW( loc = i1.linear_location_of(i3) );
//   CHECK( loc==3 );
//   indexstructure ii(i1);
//   //REQUIRE_NOTHROW( ii = indexstructure(i1) );
//   REQUIRE_NOTHROW( loc = ii.linear_location_of(i2) );
//   CHECK( loc==1 );
//   indexstructure ii2(i2);
//   REQUIRE_NOTHROW( loc = ii.linear_location_of(ii2) );
//   CHECK( loc==1 );
//   REQUIRE_NOTHROW( loc = ii.linear_location_of(indexstructure(i3)) );
//   CHECK( loc==3 );
// }

TEST_CASE( "contiguous 2d" ) {
  REQUIRE_NOTHROW( contiguous_indexstruct<int,2>( array<int,2>{0,0},array<int,2>{4,5} ) );
  REQUIRE_NOTHROW( indexstructure{ contiguous_indexstruct<int,2>( array<int,2>{0,0},array<int,2>{4,5} ) } );
}

TEST_CASE( "composite indexstruct","[indexstruct][composite][8]" ) {
  // indexstruct<index_int,1> i1,i2,ifinal;
  // shared_ptr<composite_indexstruct<index_int,1>> icomp;

  REQUIRE_NOTHROW( indexstructure<index_int,1>( contiguous_indexstruct<index_int,1>(3,5) ) );
  indexstructure<index_int,1> i1( contiguous_indexstruct<index_int,1>(3,5) );
  REQUIRE_NOTHROW( indexstructure<index_int,1>( contiguous_indexstruct<index_int,1>(10,12) ) );
  indexstructure<index_int,1> i2( contiguous_indexstruct<index_int,1>(10,12) );

  REQUIRE_NOTHROW( indexstructure<index_int,1>( composite_indexstruct<index_int,1>() ) );
  indexstructure<index_int,1> icomp{ composite_indexstruct<index_int,1>() };
  SECTION( "right away" ) {
    REQUIRE_NOTHROW( icomp.push_back(i1) );
    REQUIRE_NOTHROW( icomp.push_back(i2) );
  }
  SECTION( "reverse away" ) {
    REQUIRE_NOTHROW( icomp.push_back(i2) );
    REQUIRE_NOTHROW( icomp.push_back(i1) );
  }
  REQUIRE_NOTHROW( icomp.make_clone() );
  auto ifinal = icomp.make_clone();
  //    INFO( format("ifinal: {}",streamed(ifinal)) );
  CHECK( !ifinal.is_contiguous() );
  CHECK( ( ifinal.first_index()==3 ) );
  CHECK( ( ifinal.last_actual_index()==12 ) );
  CHECK( ifinal.volume()==6 );
}


TEST_CASE( "tricky composite simplify with indexed","[9]" ) {
  indexstructure<index_int,1> i1{ composite_indexstruct<index_int,1>() };

  indexstructure<index_int,1> left_cont{ contiguous_indexstruct<index_int,1>(0,9) };
  REQUIRE_NOTHROW( i1.push_back(left_cont) );

  indexstructure<index_int,1> right_cont{ contiguous_indexstruct<index_int,1>(11,19) };
  REQUIRE_NOTHROW( i1.push_back(right_cont) );
  #if 0
auto more_cont =contiguous_indexstruct<index_int,1>(23,30) );
REQUIRE_NOTHROW( i1.push_back(more_cont) );
auto gaps = indexstruct<index_int,1>
  ( new indexed_indexstruct<index_int,1>( vector<index_int>{10,20,22,40} ) );
REQUIRE_NOTHROW( i1.push_back(gaps) );
CHECK( i1.get_structs().size()==4 );
REQUIRE_NOTHROW( i2 = i1.force_simplify() );
INFO( format("Simplifying {} to {}",i1.as_string(),i2.as_string()) );
CHECK( i2.is_composite() );
auto i2comp = dynamic_cast<composite_indexstruct<index_int,1>*>(i2.get());
if (i2comp==nullptr) CHECK( 0 );
CHECK( i2comp.get_structs().size()==3 );
#endif
}

TEST_CASE( "tricky composite simplify with strided" ) {
  indexstructure<index_int,1> i1{ composite_indexstruct<index_int,1>() };
  indexstructure<index_int,1> left_cont{ contiguous_indexstruct<index_int,1>(0,9) };
  REQUIRE_NOTHROW( i1.push_back(left_cont) );
  indexstructure<index_int,1> right_cont{ contiguous_indexstruct<index_int,1>(11,19) };
  REQUIRE_NOTHROW( i1.push_back(right_cont) );
  indexstructure<index_int,1> more_cont{ contiguous_indexstruct<index_int,1>(23,29) };
  REQUIRE_NOTHROW( i1.push_back(more_cont) );
  SECTION( "fully incorporate" ) {
    indexstructure<index_int,1> gaps{ strided_indexstruct<index_int,1>( 10,30,10 ) };
    REQUIRE_NOTHROW( i1.push_back(gaps) );
    CHECK( i1.get_structs().size()==4 );
    INFO( format("Struct after push: {}",i1.as_string()) );
    REQUIRE_NOTHROW( i1.force_simplify() );
    INFO( format(".. simplified to {}",i1.as_string()) );
    CHECK( i1.is_composite() );
    CHECK( i1.get_structs().size()==2 );
  }
  SECTION( "shift left" ) {
    indexstructure<index_int,1> gaps{ strided_indexstruct<index_int,1>( 10,50,10 ) };
    REQUIRE_NOTHROW( i1.push_back(gaps) );
    INFO( format("Struct after push: {}",i1.as_string()) );
    CHECK( i1.get_structs().size()==4 );
    REQUIRE_NOTHROW( i1.force_simplify() );
    INFO( format("Simplifying to {}",i1.as_string()) );
    CHECK( i1.is_composite() );
    CHECK( i1.get_structs().size()==3 );
  }
}

#if 0
  SECTION( "composite over simplify" ) {
    auto i1 = shared_ptr<composite_indexstruct<index_int,1>>( new composite_indexstruct<index_int,1>() );
    bool has_index{false};
    auto right_cont =contiguous_indexstruct<index_int,1>(11,19) );
    REQUIRE_NOTHROW( i1.push_back(right_cont) );
    auto left_cont = contiguous_indexstruct<index_int,1>(2,9) );
    REQUIRE_NOTHROW( i1.push_back(left_cont) );
    SECTION( "just two members" ) {
    }
    SECTION( "three can also be incorporated" ) {
      SECTION( "contiguous" ) {
	auto more_cont =contiguous_indexstruct<index_int,1>(21,29) );
	REQUIRE_NOTHROW( i1.push_back(more_cont) );
      }
      SECTION( "indexed" ) {
	auto more_cont =indexed_indexstruct<index_int,1>( vector<index_int>{21} ) );
	REQUIRE_NOTHROW( i1.push_back(more_cont) );
      }
    }
    SECTION( "index" ) {
      has_index = true;
      auto more =indexed_indexstruct<index_int,1>( vector<index_int>{0,40,100} ) );
      REQUIRE_NOTHROW( i1.push_back(more) );
    }
    indexstruct<index_int,1> i2;
    REQUIRE_NOTHROW( i2 = i1.over_simplify() );
    INFO( format("{} --simplify-. {}",i1.as_string(),i2.as_string()) );
    if (has_index) {
      CHECK( i2.is_composite() );
    } else {
      CHECK( i2.is_contiguous() );
    }
  }
}

#endif

#if 0

TEST_CASE( "enumerating indexstructs","[10]" ) {
  int count,cnt=0;
  
  SECTION( "contiguous" ) {
    indexstruct<index_int,1> idx;
    REQUIRE_NOTHROW( idx = contiguous_indexstruct<index_int,1>(13,15) ) );
    int value = 13, count = 0;
    SECTION( "traditional" ) {
      REQUIRE_NOTHROW( idx.begin() );
      REQUIRE_NOTHROW( idx.end() );
      try {
	for (auto i=idx.begin(); i!=idx.end(); ++i) {
	  CHECK( *i==value );
	  CHECK( idx.get_ith_element(count)==(*i) );
	  value++; count++;
	}
      } catch( string c ) {
	print("Contiguous enumeration loop failed: {}\n",c);
      }
    }
    SECTION( "ranged" ) {
      try {
	for (auto i : *(idx.get())) {
	  CHECK( i==value );
	  CHECK( idx.get_ith_element(count)==i );
	  value++; count++;
	}
      } catch( string c ) {
	print("Contiguous ranging loop failed: {}\n",c);
      }
    }
    CHECK( count==3 );
  }
  SECTION( "strided" ) {
    indexstruct<index_int,1> idx;
    REQUIRE_NOTHROW( idx = strided_indexstruct<index_int,1>(3,10,2) ) );
    CHECK( ( idx.first_index()==3 ) );
    CHECK( ( idx.last_actual_index()==9 ) );
    int value = 3, count = 0;
    SECTION( "traditional" ) {
      for (auto i=idx.begin(); i!=idx.end(); ++i) {
	CHECK( *i==value );
	CHECK( idx.get_ith_element(count)==(*i) );
	value += 2; count++;
      }
    }
    SECTION( "ranged" ) {
      for (auto i : *idx) {
	CHECK( i==value );
	CHECK( idx.get_ith_element(count)==i );
	value += 2; count++;
      }
    }
  }
  SECTION( "indexed" ) {
    vector<index_int> ar{2,3,5,8};
    indexstruct<index_int,1> idx;
    REQUIRE_NOTHROW( idx = indexed_indexstruct<index_int,1>(ar) ) );
    count = 0;
    SECTION( "traditional" ) {
      for (auto i=idx.begin(); i!=idx.end(); ++i) {
	CHECK( *i==ar[count] );
	CHECK( idx.get_ith_element(cnt++)==(*i) );
	count++;
      }
    }
    SECTION( "ranged" ) {
      for (auto i : *idx) {
	CHECK( i==ar[count] );
	CHECK( idx.get_ith_element(cnt++)==i );
	count++;
      }
    }
    CHECK( count==4 );
  }
  SECTION( "composite" ) {
    auto
      i1 = contiguous_indexstruct<index_int,1>(1,10) ),
      i2 = contiguous_indexstruct<index_int,1>(21,30) );
    indexstruct<index_int,1> icomp;
    REQUIRE_NOTHROW( icomp = indexstruct<index_int,1>{ i1.make_clone() } );
    REQUIRE_NOTHROW( icomp = icomp.struct_union(i2) );
    CHECK( icomp.is_composite() );
    CHECK( icomp.volume()==20 );
    cnt = 0; const char *path;
    SECTION( "traditional" ) { path = "traditional";
      for ( indexstruct<index_int,1> i=icomp.begin(); i!=icomp.end(); ++i ) {
	//printf("%d\n",*i);
	cnt++;
      }
    }
    SECTION( "ranged" ) { path = "ranged";
      for ( auto i : *icomp ) {
	//printf("%d\n",i);
	cnt++;
      }
    }
    INFO( "path: " << path );
    CHECK(cnt==20);
  }
}

TEST_CASE( "indexstruct intersections","[indexstruct][intersect][20]" ) {
  
  indexstruct<index_int,1> i1,i2,i3,i4;
  coordinate<index_int,1> one; one.set(1);
  SECTION( "first cont" ) {
    i1 = indexstruct<index_int,1>{ new contiguous_indexstruct<index_int,1>(1,10) };
    indexstructure<index_int,1> I1( contiguous_indexstruct<index_int,1>(1,10) );
    SECTION( "cont-cont" ) {
      i2 = indexstruct<index_int,1>{ new contiguous_indexstruct<index_int,1>(5,12) };
      REQUIRE_NOTHROW( i3 = i1.intersect(i2) );
      REQUIRE( i3!=nullptr );
      CHECK( i3.is_contiguous() );
      CHECK( ( i3.first_index()==5 ) );
      CHECK( ( i3.last_actual_index()==10 ) );
      CHECK( !i1.contains(i2) );
      CHECK( i1.contains(i3) );
      CHECK( i2.contains(i3) );

      indexstructure<index_int,1> I2( contiguous_indexstruct<index_int,1>(5,12) ), I3;
      REQUIRE_NOTHROW( I3 = I1.intersect(I2) );
      CHECK( I3.is_contiguous() );
      CHECK( ( I3.first_index()==5 ) );
      CHECK( ( I3.last_actual_index()==10 ) );

      i2 = indexstruct<index_int,1>{ new contiguous_indexstruct<index_int,1>(10,12) };
      REQUIRE_NOTHROW( i3 = i1.intersect(i2) );
      REQUIRE( i3!=nullptr );
      CHECK( i3.is_contiguous() );
      CHECK( ( i3.first_index()==10 ) );
      CHECK( ( i3.last_actual_index()==10 ) );
      CHECK( !i1.contains(i2) );
      CHECK( i1.contains(i3) );
      CHECK( i2.contains(i3) );
      REQUIRE_THROWS( i4 = i2.relativize_to(i1) );
      REQUIRE_NOTHROW( i4 = i3.relativize_to(i1) ); // [10,10] in [1,10] is [9,9]
      CHECK( i4.is_contiguous() );
      CHECK( ( i4.first_index()==9 ) );
      CHECK( ( i4.last_actual_index()==9 ) );
      REQUIRE_NOTHROW( i4 = i3.relativize_to(i2) );
      CHECK( i4.is_contiguous() );
      CHECK( ( i4.first_index()==0 ) );
      CHECK( ( i4.last_actual_index()==0 ) );

      i2 = indexstruct<index_int,1>{ new contiguous_indexstruct<index_int,1>(11,12) };
      REQUIRE_NOTHROW( i3 = i1.intersect(i2) );
      REQUIRE( i3.is_empty() );

      i2 = indexstruct<index_int,1>{ new strided_indexstruct<index_int,1>(10,12,2) };
      i3 = strided_indexstruct<index_int,1>(8,14,2) );
      REQUIRE_NOTHROW( i4 = i2.relativize_to(i3) );
      CHECK( i4.stride()==one );
      CHECK( i4.volume()==2 );
      CHECK( ( i4.first_index()==1 ) );
      CHECK( ( i4.last_actual_index()==2 ) );
    }
    SECTION( "cont-idx" ) {
      int len=3; vector<index_int> idx{4,8,11};
      i2 = indexstruct<index_int,1>{ new indexed_indexstruct<index_int,1>(idx) };
      REQUIRE_NOTHROW( i3 = i1.intersect(i2) );
      REQUIRE( !i3.is_empty() );
      CHECK( i3.is_indexed() );
      CHECK( ( i3.first_index()==4 ) );
      CHECK( ( i3.last_actual_index()==8 ) );
      CHECK( !i1.contains(i2) );
      CHECK( i1.contains(i3) );
      CHECK( i2.contains(i3) );
      REQUIRE_THROWS( i2.relativize_to(i1) );

      len=3; vector<index_int> idxs{4,8,10}; 
      i2 = indexstruct<index_int,1>{ new indexed_indexstruct<index_int,1>(idxs) };
      REQUIRE_NOTHROW( i3 = i1.intersect(i2) ); // [1,10] & [4,8,10] => i3 = [4,8,10]
      REQUIRE( !i3.is_empty() );
      CHECK( i3.is_indexed() );
      CHECK( ( i3.first_index()==4 ) );
      CHECK( ( i3.last_actual_index()==10 ) );
      CHECK( i1.contains(i2) );
      CHECK( i1.contains(i3) );
      CHECK( i2.contains(i3) );

      CHECK( i3.is_indexed() );

      REQUIRE_NOTHROW( i4 = i3.relativize_to(i1) ); // [4,8,10] in [1:10] i3 is indexed

      CHECK( i4.is_indexed() );
      CHECK( ( i4.first_index()==3 ) );
      CHECK( ( i4.last_actual_index()==9 ) );
    }
  }
  SECTION( "stride-stride" ) {
    i1 = indexstruct<index_int,1>{ new strided_indexstruct<index_int,1>(10,20,2) };
    CHECK( i1.volume()==6 );
    i2 = indexstruct<index_int,1>{ new strided_indexstruct<index_int,1>(12,14,2) };
    coordinate<index_int,1> one; one.set(1);
    REQUIRE_NOTHROW( i3 = i1.intersect(i2) );
    CHECK( i3.volume()==2 );
    CHECK( i3.is_strided() );
    CHECK( i3.stride()==one*2 );

    i2 = indexstruct<index_int,1>{ new strided_indexstruct<index_int,1>(12,22,10) };
    REQUIRE_NOTHROW( i3 = i1.intersect(i2) );
    CHECK( i3.volume()==1 );

    i2 = indexstruct<index_int,1>{ new strided_indexstruct<index_int,1>(12,20,4) };
    REQUIRE_NOTHROW( i3 = i1.intersect(i2) );
    CHECK( i3.volume()==3 );
    CHECK( i3.is_strided() );

    i2 = indexstruct<index_int,1>{ new strided_indexstruct<index_int,1>(12,20,5) };
    REQUIRE_NOTHROW( i3 = i1.intersect(i2) );
    CHECK( i3.volume()==1 );

    i2 = indexstruct<index_int,1>{ new strided_indexstruct<index_int,1>(13,20,4) };
    REQUIRE_NOTHROW( i3 = i1.intersect(i2) );
    CHECK( i3!=nullptr );
    CHECK( i3.is_empty() );
  }
  SECTION( "strided-indexed" ) {
    i1 = indexstruct<index_int,1>{ new strided_indexstruct<index_int,1>(10,20,10) };
    CHECK( i1.volume()==2 );
    i2 = indexstruct<index_int,1>{
      new indexed_indexstruct<index_int,1>( vector<index_int>{10,16,20} ) };
    CHECK( i2.volume()==3 );
    REQUIRE_NOTHROW( i3 = i1.relativize_to(i2) );
    CHECK( i3.volume()==2 );
    CHECK( ( i3.first_index()==0 ) );
    CHECK( ( i3.last_actual_index()==2 ) );
  }
  SECTION( "idx-idx" ) {
    indexstruct<index_int,1> i5{nullptr};
    i2 = indexstruct<index_int,1>{
      new indexed_indexstruct<index_int,1>( vector<index_int>{4,8,11} ) };
    i3 = indexstruct<index_int,1>{
      new indexed_indexstruct<index_int,1>( vector<index_int>{3,8,10,11,12} ) };
    REQUIRE_NOTHROW( i4 = i2.intersect(i3) );
    REQUIRE( i4!=nullptr );
    CHECK( i4.volume()==2 );
    CHECK( i4.is_indexed() );
    CHECK( ( i4.first_index()==8 ) );
    CHECK( ( i4.last_actual_index()==11 ) );
    CHECK( !i2.contains(i3) );
    CHECK( i2.contains(i4) );
    CHECK( i3.contains(i4) );

    indexstructure<index_int,1> I2(i2), I3(i3), I4;
    REQUIRE_NOTHROW( I4 = I2.intersect(I3) );
    CHECK( I4.volume()==2 );
    CHECK( I4.is_indexed() );
    CHECK( ( I4.first_index()==8 ) );
    CHECK( ( I4.last_actual_index()==11 ) );

    REQUIRE_THROWS( i5 = i3.relativize_to(i2) );
    REQUIRE_THROWS( i5 = i2.relativize_to(i3) );
    CHECK( i2.contains(i4) );
    CHECK( i2.is_indexed() );
    CHECK( i4.is_indexed() );
    REQUIRE_NOTHROW( i5 = i4.relativize_to(i2) ); // [8,11] in indexed:[4,8,11]
    CHECK( i5.is_indexed() );
    CHECK( ( i5.first_index()==1 ) );
    CHECK( ( i5.last_actual_index()==2 ) );
    REQUIRE_NOTHROW( i5 = i4.relativize_to(i3) ); // [8,11] in [3,8,10,11,12]
    CHECK( i5.is_indexed() );
    CHECK( ( i5.first_index()==1 ) );
    CHECK( ( i5.last_actual_index()==3 ) );
  }
}

TEST_CASE( "indexstruct differences","[minus][21]" ) {
  indexstruct<index_int,1> *it;
  indexstruct<index_int,1> i1,i2,i3;
  SECTION( "cont-cont non-overlapping" ) {
    i1 = contiguous_indexstruct<index_int,1>(5,15) );
    i2 = contiguous_indexstruct<index_int,1>(20,30) );
    SECTION( "one way" ) {
      REQUIRE_NOTHROW( i3 = i1.minus(i2) );
      INFO( i3.as_string() );
      CHECK( i3.is_contiguous() );
      CHECK( i3.equals(i1) );
    }
    SECTION( "other way" ) {
      REQUIRE_NOTHROW( i3 = i2.minus(i1) ); 
      CHECK( i3.is_contiguous() );
      CHECK( i3.equals(i2) );
    }
  }
  SECTION( "cont-cont containment" ) {
    i1 = contiguous_indexstruct<index_int,1>(5,30) );
    i2 = contiguous_indexstruct<index_int,1>(15,20) );
    SECTION( "one way" ) {
      REQUIRE_NOTHROW( i3 = i1.minus(i2) );
      CHECK( i3.volume()==20 );
      CHECK( ( i3.first_index()==5 ) );
      CHECK( ( i3.last_actual_index()==30 ) );
    }
    SECTION( "other way" ) {
      REQUIRE_NOTHROW( i3 = i2.minus(i1) );
      CHECK( i3.is_empty() );
    }
  }
  SECTION( "cont-cont for real" ) {
    i1 = contiguous_indexstruct<index_int,1>(5,20) );
    i2 = contiguous_indexstruct<index_int,1>(15,30) );
    SECTION( "one way" ) {
      REQUIRE_NOTHROW( i3 = i1.minus(i2) );
      INFO( i3.as_string() );
      CHECK( i3.is_contiguous() );
      CHECK( ( i3.first_index()==5 ) );
      CHECK( ( i3.last_actual_index()==14 ) );
    }
    SECTION( "other way" ) {
      REQUIRE_NOTHROW( i3 = i2.minus(i1) );
      CHECK( i3.is_contiguous() );
      CHECK( ( i3.first_index()==21 ) );
      CHECK( ( i3.last_actual_index()==30 ) );
    }
  }
  SECTION( "contiguous minus contiguous, creating gap" ) {
    i1 = contiguous_indexstruct<index_int,1>(1,40) );
    i2 = contiguous_indexstruct<index_int,1>(11,20) );
    REQUIRE_NOTHROW( i3 = i1.minus(i2) );
    INFO( "resulting i3: " << i3.as_string() );
    CHECK( i3.volume()==30 );
    CHECK( !i3.contains_element(11) );
    CHECK( i3.is_composite() );
    composite_indexstruct<index_int,1> *i4;
    REQUIRE_NOTHROW( i4 = dynamic_cast<composite_indexstruct<index_int,1>*>(i3.get()) );
    CHECK( i4!=nullptr );
    CHECK( i4.get_structs().size()==2 );
  }
  SECTION( "strided minus contiguous, creating gap" ) {
    i1 = strided_indexstruct<index_int,1>(1,41,4) );
    CHECK( i1.volume()==11 );
    i2 = contiguous_indexstruct<index_int,1>(12,16) ); // this only cuts 1
    REQUIRE_NOTHROW( i3 = i1.minus(i2) );
    INFO( "resulting i3: " << i3.as_string() );
    CHECK( i3.is_composite() );
    CHECK( !i3.contains_element(13) );
    CHECK( i3.volume()==10 );
    composite_indexstruct<index_int,1> *i4;
    REQUIRE_NOTHROW( i4 = dynamic_cast<composite_indexstruct<index_int,1>*>(i3.get()) );
    CHECK( i4!=nullptr );
    CHECK( i4.get_structs().size()==2 );
  }
  SECTION( "strided minus contiguous, hitting nothing" ) {
    i1 = strided_indexstruct<index_int,1>(1,41,4) );
    CHECK( i1.volume()==11 );
    i2 = contiguous_indexstruct<index_int,1>(14,16)); // this hits nothing: falls between 13-17
    REQUIRE_NOTHROW( i3 = i1.minus(i2) );
    INFO( "resulting i3: " << i3.as_string() );
    CHECK( i3.volume()==11 );
    CHECK( i3.is_strided() );
  }
  SECTION( "indexed cont" ) {
    i1 = strided_indexstruct<index_int,1>( 4,16,4 ) );
    i1 = i1.convert_to_indexed() ;
    CHECK( i1.is_indexed() );
    CHECK( ( i1.first_index()==4 ) );
    CHECK( ( i1.last_actual_index()==16 ) );
    i2  = contiguous_indexstruct<index_int,1>( 13,17 ) );
    REQUIRE_NOTHROW( i3 = i1.minus(i2) );
    CHECK( i3.is_indexed() );
    CHECK( ( i3.first_index()==4 ) );
    CHECK( ( i3.last_actual_index()==12 ) );

    indexstructure<index_int,1> I1(strided_indexstruct<index_int,1>( 4,16,4 )), I2(i2), I3;
    REQUIRE_NOTHROW( I3 = I1.minus(I2) );
    CHECK( I3.volume()==3 );
    CHECK( ( I3.first_index()==4 ) );
    CHECK( ( I3.last_actual_index()==12 ) );

  }
  SECTION( "indexed-indexed" ) {
    i1 = strided_indexstruct<index_int,1>( 4,16,4 ) );
    i1 = indexstruct<index_int,1>( i1.convert_to_indexed() );
    i2 = strided_indexstruct<index_int,1>( 5,17,4 ) );
    i2 = indexstruct<index_int,1>( i2.convert_to_indexed() );
    REQUIRE_NOTHROW( i3 = i1.minus(i2) );
    CHECK( i3.equals(i1) );
    i2 = strided_indexstruct<index_int,1>( 16,20,2 ) );
    i2 = indexstruct<index_int,1>( i2.convert_to_indexed() );
    REQUIRE_NOTHROW( i3 = i1.minus(i2) );
    CHECK( i3.volume()==3 );
    CHECK( ( i3.first_index()==4 ) );
    CHECK( ( i3.last_actual_index()==12 ) );
  }
}

TEST_CASE( "indexstruct<index_int,1> unions","[indexstruct<index_int,1>][union][22]" ) {
  indexstruct<index_int,1> i1,i2,i3;

  SECTION( "convert from stride 1" ) {
    i1 = indexstruct<index_int,1>{ new contiguous_indexstruct<index_int,1>(2,2+SMALLBLOCKSIZE-1) };
    CHECK( i1.is_contiguous() );
    CHECK_NOTHROW( i2 = i1.convert_to_indexed() );
    CHECK( i2.is_indexed() );
    CHECK( ( i2.first_index()==2 ) );
    CHECK( ( i2.last_actual_index()==2+SMALLBLOCKSIZE-1 ) );
    CHECK( i2.volume()==i1.volume() );
  }
  SECTION( "convert from stride 2" ) {
    i1 = indexstruct<index_int,1>{ new strided_indexstruct<index_int,1>(2,2+2*SMALLBLOCKSIZE-2,2) };
    CHECK_NOTHROW( i2 = i1.convert_to_indexed() );
    CHECK( i2.is_indexed() );
    CHECK( ( i2.first_index()==2 ) );
    CHECK( ( i2.last_actual_index()==2+2*SMALLBLOCKSIZE-2 ) );
    CHECK( i2.volume()==i1.volume() );
  }
  SECTION( "cont-cont" ) {
    i1 = indexstruct<index_int,1>{ new contiguous_indexstruct<index_int,1>(1,10) };
    indexstructure<index_int,1> I1(contiguous_indexstruct<index_int,1>(1,10));
    SECTION( "1" ) {
      i2 = indexstruct<index_int,1>{ new contiguous_indexstruct<index_int,1>(5,12) };
      indexstructure<index_int,1> I2(contiguous_indexstruct<index_int,1>(5,12)), I3;
      REQUIRE_NOTHROW( i3 = i1.struct_union(i2) );
      REQUIRE_NOTHROW( I3 = I1.struct_union(I2) );
      REQUIRE( i3!=nullptr );
      CHECK( i3.is_contiguous() );
      CHECK( ( i3.first_index()==1 ) );
      CHECK( ( i3.last_actual_index()==12 ) );
    }
    SECTION( "2" ) {
      i2 = indexstruct<index_int,1>{ new contiguous_indexstruct<index_int,1>(11,13) };
      REQUIRE_NOTHROW( i3 = i1.struct_union(i2) );
      REQUIRE( i3!=nullptr );
      CHECK( i3.is_contiguous() );
      CHECK( ( i3.first_index()==1 ) );
      CHECK( ( i3.last_actual_index()==13 ) );
    }
    SECTION( "3" ) {
      SECTION( "extend right" ) {
	i1 = indexstruct<index_int,1>{ new contiguous_indexstruct<index_int,1>(1,10) };
	i2 = indexstruct<index_int,1>{ new contiguous_indexstruct<index_int,1>(13) };
	REQUIRE_NOTHROW( i3 = i1.struct_union(i2) ); // [1--10]+[13]

	REQUIRE( i3!=nullptr );
	CHECK( !i3.is_indexed() );
	CHECK( i3.is_composite() );
	index_int i1l,i2l;
	REQUIRE_NOTHROW( i1l = i1.volume() );
	REQUIRE_NOTHROW( i2l = i2.volume() );
	CHECK( i3.volume()==(i1l+i2l) );
	CHECK( ( i3.first_index()==1 ) );
	CHECK( ( i3.last_actual_index()==13 ) );
      }
      SECTION( "extend left" ) {
	i1 = indexstruct<index_int,1>{ new contiguous_indexstruct<index_int,1>(1,10) };
	i2 = indexstruct<index_int,1>{ new contiguous_indexstruct<index_int,1>(14) };
	REQUIRE_NOTHROW( i3 = i2.struct_union(i1) ); // [1--10]+[14]
	//print("i3 {}\n",i3.as_string());
	REQUIRE( i3!=nullptr );
	CHECK( !i3.is_indexed() );
	CHECK( i3.is_composite() );
	CHECK( i3.volume()==(i1.volume()+i2.volume()) );
	CHECK( ( i3.first_index()==1 ) );
	CHECK( ( i3.last_actual_index()==14 ) );
      }
    }
  }
  SECTION( "cont-idx giving indexed" ) {
    i1 = indexstruct<index_int,1>{ new contiguous_indexstruct<index_int,1>(5,8) };
    i2 = strided_indexstruct<index_int,1>(8,12,2) );
    CHECK( i2.volume()==3 );
    REQUIRE_NOTHROW( i2 = i2.convert_to_indexed() );
    CHECK( i2.volume()==3 );
    REQUIRE_NOTHROW( i3 = i1.struct_union(i2) ); // [5-8] & [8,10,12] overlap 1
    INFO( "i3 should be [5-8] & [8,10,12], is: " << i3.as_string() );
    CHECK( !i3.is_contiguous() );
    CHECK( i1.volume()==4 );
    CHECK( i2.volume()==3 );
    CHECK( i3.volume()==6 );
    CHECK( ( i3.first_index()==5 ) );
    CHECK( ( i3.last_actual_index()==12) );
  }
  SECTION( "cont-idx extending" ) {
    i1 = indexstruct<index_int,1>{ new contiguous_indexstruct<index_int,1>(5,11) };
    i2 = strided_indexstruct<index_int,1>(8,12,2) );
    indexstructure<index_int,1> I3,
      I1(contiguous_indexstruct<index_int,1>(5,11)),
      I2(strided_indexstruct<index_int,1>(8,12,2));
    CHECK( i2.volume()==3 );
    REQUIRE_NOTHROW( i2 = i2.convert_to_indexed() );
    CHECK( i2.is_indexed() );
    CHECK( i2.volume()==3 );
    REQUIRE_NOTHROW( i3 = i1.struct_union(i2.get()) );
    REQUIRE_NOTHROW( I3 = I1.struct_union(I2) );
    CHECK( i1.volume()==7 );
    CHECK( i3.is_contiguous() );
    CHECK( i3.volume()==8 );
    CHECK( ( i3.first_index()==5 ) );
    CHECK( ( i3.last_actual_index()==12) );
    // VLE do we need this? CHECK( I3.is_contiguous() );
    CHECK( I3.volume()==8 );
    CHECK( ( I3.first_index()==5 ) );
    CHECK( ( I3.last_actual_index()==12) );
  }
  SECTION( "cont-idx extending2" ) {
    i1 = indexstruct<index_int,1>{ new contiguous_indexstruct<index_int,1>(5,11) };
    i2 = strided_indexstruct<index_int,1>(6,12,2) );
    CHECK( i2.volume()==4 );
    REQUIRE_NOTHROW( i2 = i2.convert_to_indexed() );
    CHECK( i2.is_indexed() );
    CHECK( i2.volume()==4 );
    REQUIRE_NOTHROW( i3 = i1.struct_union(i2.get()) );
    CHECK( i1.volume()==7 );
    CHECK( i3.is_contiguous() );
    CHECK( i3.volume()==8 );
    CHECK( ( i3.first_index()==5 ) );
    CHECK( ( i3.last_actual_index()==12) );
  }
  SECTION( "idx-cont" ) {
    i1 = indexstruct<index_int,1>{ new strided_indexstruct<index_int,1>(8,12,2) };
    //    i1.convert_to_indexed();
    i2 = indexstruct<index_int,1>{ new contiguous_indexstruct<index_int,1>(5,8) };
    REQUIRE_NOTHROW( i3 = i1.struct_union(i2) );
    CHECK( i3.volume()==6 );
    CHECK( ( i3.first_index()==5 ) );
    CHECK( ( i3.last_actual_index()==12) );
  }
  SECTION( "tricky composite stuff" ) {
    shared_ptr<composite_indexstruct<index_int,1>> icomp;
    REQUIRE_NOTHROW
      ( icomp = shared_ptr<composite_indexstruct<index_int,1>>{ new composite_indexstruct<index_int,1>() } );
    CHECK( icomp.is_composite() );
    REQUIRE_NOTHROW
      ( icomp.push_back( contiguous_indexstruct<index_int,1>(1,10) ) ) );
    REQUIRE_NOTHROW
      ( icomp.push_back( contiguous_indexstruct<index_int,1>(31,40) ) ) );
    i1 = icomp.make_clone();
    CHECK( i1.is_composite() );
    vector< indexstruct<index_int,1> > members;
    indexstructure<index_int,1> I1;
    REQUIRE_NOTHROW( I1 = indexstructure<index_int,1>(composite_indexstruct<index_int,1>()) );
    CHECK( I1.is_composite() );
    REQUIRE_NOTHROW( I1.push_back( contiguous_indexstruct<index_int,1>(1,10) ) );
    return;
    REQUIRE_NOTHROW( I1.push_back( contiguous_indexstruct<index_int,1>(31,40) ) );
    SECTION( "can not merge" ) {
      i2 = indexstruct<index_int,1>{ new strided_indexstruct<index_int,1>(11,15,2) };
      REQUIRE_NOTHROW( i1 = i1.struct_union(i2) );
      REQUIRE_NOTHROW( members = dynamic_cast<composite_indexstruct<index_int,1>*>(i1.get()).get_structs() );
      CHECK( members.size()==3 );
    }
    SECTION( "can merge" ) {
      i2 = indexstruct<index_int,1>{ new contiguous_indexstruct<index_int,1>(11,15) };
      REQUIRE_NOTHROW( i1 = i1.struct_union(i2) );
      REQUIRE_NOTHROW( members = dynamic_cast<composite_indexstruct<index_int,1>*>(i1.get()).get_structs() );
      CHECK( members.size()==2 );
    }
    CHECK( i1.contains_element(10) );
    CHECK( i1.contains_element(11) );
  }
}

TEST_CASE( "struct disjoint","[23]" ) {
  indexstruct<index_int,1> i1,i2;
  indexstructure<index_int,1> I1,I2;
  SECTION( "disjoint strided" ) {
    REQUIRE_NOTHROW( i1 = strided_indexstruct<index_int,1>(1,10,2) ) );
    REQUIRE_NOTHROW( i2 = contiguous_indexstruct<index_int,1>(10,20) ) );
    CHECK( i1.disjoint(i2) );
    REQUIRE_NOTHROW( I1 = indexstructure<index_int,1>(strided_indexstruct<index_int,1>(1,10,2)) );
    REQUIRE_NOTHROW( I2 = indexstructure<index_int,1>(contiguous_indexstruct<index_int,1>(10,20)) );
    CHECK( I1.disjoint(I2) );
  }
  SECTION( "disjoint strided, interleaved" ) {
    REQUIRE_NOTHROW( i1 = strided_indexstruct<index_int,1>(1,10,2) ) );
    REQUIRE_NOTHROW( i2 = strided_indexstruct<index_int,1>(8,20,2) ) );
    CHECK( i1.disjoint(i2) );
  }
  SECTION( "disjoint strided, hard to tell" ) {
    REQUIRE_NOTHROW( i1 = strided_indexstruct<index_int,1>(0,10,5) ) );
    REQUIRE_NOTHROW( i2 = strided_indexstruct<index_int,1>(7,9,2) ) );
    CHECK_THROWS( i1.disjoint(i2) );
  }
  SECTION( "disjoint indexed & strided, range" ) {
    REQUIRE_NOTHROW( i1 = indexstruct<index_int,1>
		     ( new indexed_indexstruct<index_int,1>(vector<index_int>{1,3,6}) ) );
    REQUIRE_NOTHROW( i2 = contiguous_indexstruct<index_int,1>(7,10) ) );
    SECTION( "one way" ) { CHECK( i1.disjoint(i2) ); }
    SECTION( "oth way" ) { CHECK( i2.disjoint(i1) ); }
  }
  SECTION( "disjoint indexed, range" ) {
    REQUIRE_NOTHROW( i1 = indexstruct<index_int,1>
		     ( new indexed_indexstruct<index_int,1>(vector<index_int>{1,3,6}) ) );
    REQUIRE_NOTHROW( i2 = indexstruct<index_int,1>
		     ( new indexed_indexstruct<index_int,1>(vector<index_int>{7,8,10}) ) );
    CHECK( i1.disjoint(i2) );
  }
  SECTION( "disjoint indexed, hard to tell" ) {
    REQUIRE_NOTHROW( i1 = indexstruct<index_int,1>
		     ( new indexed_indexstruct<index_int,1>(vector<index_int>{1,3,16}) ) );
    REQUIRE_NOTHROW( i2 = indexstruct<index_int,1>
		     ( new indexed_indexstruct<index_int,1>(vector<index_int>{7,8,10}) ) );
    CHECK_THROWS( i1.disjoint(i2) );
    REQUIRE_NOTHROW( I1 = indexstructure<index_int,1>(indexed_indexstruct<index_int,1>(vector<index_int>{1,3,16})) );
    REQUIRE_NOTHROW( I2 = indexstructure<index_int,1>(indexed_indexstruct<index_int,1>(vector<index_int>{7,8,10})) );
    CHECK_THROWS( I1.disjoint(I2) );
  }
}

TEST_CASE( "struct containment","[24]" ) {
  indexstruct<index_int,1> i1,i2,i3;
  indexstructure<index_int,1> I1,I2,I3;
  SECTION( "cont" ) {
    i1 = contiguous_indexstruct<index_int,1>(1,10) );

    i2 = contiguous_indexstruct<index_int,1>(2,10) );
    REQUIRE( i1.contains(i2) );
    REQUIRE( !i2.contains(i1) );

    i2 = contiguous_indexstruct<index_int,1>(2,11) );
    REQUIRE( !i1.contains(i2) );
    REQUIRE( !i2.contains(i1) );

    i2 = strided_indexstruct<index_int,1>(3,11,3) ); // 3,6,9
    REQUIRE( i1.contains(i2) );
    REQUIRE( !i2.contains(i1) );

    i2 = strided_indexstruct<index_int,1>(4,11,3) ); // 4,7,10
    REQUIRE( i1.contains(i2) );
    REQUIRE( !i2.contains(i1) );
  }
  SECTION( "stride" ) {
    i1 = strided_indexstruct<index_int,1>(10,20,3) );

    i2 = contiguous_indexstruct<index_int,1>(13) );
    REQUIRE( i1.contains(i2) );
  }
  SECTION( "idx" ) {
    int len=5; vector<index_int> idx{1,2,4,6,9};
    i1 = indexed_indexstruct<index_int,1>(idx) );

    i2 = strided_indexstruct<index_int,1>(4,4,3) );
    REQUIRE( i1.contains(i2) );

    i2 = strided_indexstruct<index_int,1>(4,6,2) );
    REQUIRE( i1.contains(i2) );

    i2 = contiguous_indexstruct<index_int,1>(4,4) );
    REQUIRE( i1.contains(i2) );
  }
  SECTION( "composite" ) {
    auto i1 = contiguous_indexstruct<index_int,1>(0,10) );
    REQUIRE_NOTHROW( i1 = i1.struct_union
		     ( contiguous_indexstruct<index_int,1>(20,29) ) ) );
    CHECK( i1.is_composite() );

    i2 = contiguous_indexstruct<index_int,1>(2,5) );
    CHECK( i1.contains(i2) );

    I1 = indexstructure<index_int,1>(contiguous_indexstruct<index_int,1>(0,10));
    I1 = I1.struct_union(indexstructure<index_int,1>(contiguous_indexstruct<index_int,1>(20,29)));
    I2 = indexstructure<index_int,1>(contiguous_indexstruct<index_int,1>(2,5));
    CHECK( I1.contains(I2) );

    i2 = contiguous_indexstruct<index_int,1>(22,25) );
    CHECK( i1.contains(i2) );
    I2 = indexstructure<index_int,1>(contiguous_indexstruct<index_int,1>(22,25));
    CHECK( I1.contains(I2) );

    i2 = contiguous_indexstruct<index_int,1>(8,25) );
    CHECK( !i1.contains(i2) );
    I2 = indexstructure<index_int,1>(contiguous_indexstruct<index_int,1>(8,25));
    CHECK( !I1.contains(I2) );

  }
}

TEST_CASE( "struct split","[split][25]" ) {
  SECTION( "contiguous" ) {
    auto i1 = contiguous_indexstruct<index_int,1>(10,20) );
    indexstructure<index_int,1> I1(contiguous_indexstruct<index_int,1>(10,20));
    indexstruct<index_int,1> c;
    SECTION( "non intersect" ) {
      auto i2 = contiguous_indexstruct<index_int,1>(30,40) );
      REQUIRE_THROWS( c = i1.split(i2) );
    }
    SECTION( "contains" ) {
      auto i2 = contiguous_indexstruct<index_int,1>(0,40) );
      REQUIRE_THROWS( c = i1.split(i2) );
    }
    SECTION( "right" ) {
      auto i2 = contiguous_indexstruct<index_int,1>(15,30) );
      REQUIRE_NOTHROW( c = i1.split(i2) );
      composite_indexstruct<index_int,1> *cc = dynamic_cast<composite_indexstruct<index_int,1>*>(c.get());
      CHECK( cc!=nullptr );
      auto ss = cc.get_structs();
      CHECK( ss.size()==2 );
      auto l = contiguous_indexstruct<index_int,1>(10,14) );
      CHECK( ss.at(0).equals(l) );
      auto r = contiguous_indexstruct<index_int,1>(15,20) );
      CHECK( ss.at(1).equals(r) );
    }
    SECTION( "left" ) {
      auto i2 = contiguous_indexstruct<index_int,1>(0,15) );
      REQUIRE_NOTHROW( c = i1.split(i2) );
      composite_indexstruct<index_int,1> *cc = dynamic_cast<composite_indexstruct<index_int,1>*>(c.get());
      CHECK( cc!=nullptr );
      auto ss = cc.get_structs();
      CHECK( ss.size()==2 );
      auto l = contiguous_indexstruct<index_int,1>(10,15) );
      CHECK( ss.at(0).equals(l) );
      auto r = contiguous_indexstruct<index_int,1>(16,20) );
      CHECK( ss.at(1).equals(r) );
    }
  }
  SECTION( "strided" ) {
    auto i1 = strided_indexstruct<index_int,1>(10,20,2) );
    indexstructure<index_int,1> I1(strided_indexstruct<index_int,1>(10,20,2));
    indexstruct<index_int,1> c;
    SECTION( "non intersect" ) {
      auto i2 = contiguous_indexstruct<index_int,1>(30,40) );
      indexstructure<index_int,1> I2(contiguous_indexstruct<index_int,1>(30,40)), c;
      REQUIRE_THROWS( c = i1.split(i2) );
      REQUIRE_THROWS( c = I1.split(I2) );
    }
    SECTION( "contains" ) {
      auto i2 = contiguous_indexstruct<index_int,1>(0,40) );
      REQUIRE_THROWS( c = i1.split(i2) );
    }
    SECTION( "right" ) {
      auto i2 = contiguous_indexstruct<index_int,1>(15,30) );
      REQUIRE_NOTHROW( c = i1.split(i2) );
      INFO( format("split to: {}",c.as_string()) );
      composite_indexstruct<index_int,1> *cc = dynamic_cast<composite_indexstruct<index_int,1>*>(c.get());
      CHECK( cc!=nullptr );
      auto ss = cc.get_structs();
      CHECK( ss.size()==2 );
      {
	auto l = strided_indexstruct<index_int,1>(10,14,2) );
	CHECK( ss.at(0).equals(l) );
      }
      {
	auto r = strided_indexstruct<index_int,1>(16,20,2) );
	INFO( format("Is {}, should be {}",ss.at(1).as_string(),r.as_string()) );
	CHECK( ss.at(1).equals(r) );
      }
    }
    SECTION( "left" ) {
      auto i2 = contiguous_indexstruct<index_int,1>(0,15) );
      REQUIRE_NOTHROW( c = i1.split(i2) );
      composite_indexstruct<index_int,1> *cc = dynamic_cast<composite_indexstruct<index_int,1>*>(c.get());
      CHECK( cc!=nullptr );
      auto ss = cc.get_structs();
      CHECK( ss.size()==2 );
      auto l = strided_indexstruct<index_int,1>(10,14,2) );
      CHECK( ss.at(0).equals(l) );
      auto r = strided_indexstruct<index_int,1>(16,20,2) );
      CHECK( ss.at(1).equals(r) );
    }
  }
}
#endif

TEST_CASE( "struct multiplying","[31]" ) {
  {
    INFO("1D");
    indexstructure i1(contiguous_indexstruct<index_int,1>(5,10));
    INFO( format("start with contiguous: {}",i1.as_string()) );
    ioperator<index_int,1> op;
    coordinate<index_int,1> one; one.set(1);
    SECTION( "old notation" ) {
      REQUIRE_NOTHROW( op = ioperator<index_int,1>("*2") );
    }
    // SECTION( "new notation" ) {
    //   REQUIRE_NOTHROW( op = ioperator<index_int,1>("*",2) );
    // }
    REQUIRE_NOTHROW( i1.operate(op) );
    auto i2 = i1.operate(op);
    // printf("i2 s/b strided: <<%s>>\n",i2.as_string().data());
    INFO( format("multiplied by 2: {}",i2.as_string()) );
    CHECK( i2.is_strided() );
    CHECK( ( i2.first_index()==10 ) );
    CHECK( ( i2.last_actual_index()==20 ) );
    CHECK( i2.volume()==i1.volume() );
    CHECK( i2.stride()==one*2 );
  }
}

index_int itimes2(index_int i) { return 2*i; }
indexstruct<index_int,1> *itimes2i(index_int i) { return new contiguous_indexstruct<index_int,1>(2*i,2*i); }

TEST_CASE( "structs and operations","[indexstruct<index_int,1>][operate][30]" ) {
  ioperator<index_int,1> op; 

  SECTION( "multiply by constant" ) {
  }
  SECTION( "multiply range by constant" ) {
    indexstructure i1{ contiguous_indexstruct<index_int,1>(5,10) };
    REQUIRE_NOTHROW( op = ioperator<index_int,1>("x2") );
    REQUIRE_NOTHROW( i1.operate(op) );
    auto i2 = i1.operate(op);
    CHECK( i2.is_contiguous() );
    CHECK( ( i2.first_index()==10 ) );
    CHECK( ( i2.last_actual_index()==21 ) );
    CHECK( i2.volume()==i1.volume()*2 );
    CHECK( i2.is_strided() );
  }
  SECTION( "multiply by function" ) {
    REQUIRE_NOTHROW( op = ioperator<index_int,1>(&itimes2) );
    {
      coordinate<index_int,1> opped;
      REQUIRE_NOTHROW( opped = op.operate( coordinate<index_int,1>(1) ) );
      CHECK( opped==coordinate<index_int,1>(2) );
    }
    indexstructure i1{ contiguous_indexstruct<index_int,1>(5,10) };
    REQUIRE_NOTHROW( i1.operate(op) );
    auto i2 = i1.operate(op);
    // CHECK( i2.is_strided() );
    CHECK( ( i2.first_index()==10 ) );
    CHECK( ( i2.last_actual_index()==20 ) );
  }
#if 0
  SECTION( "shift strided" ) {
    i1 = strided_indexstruct<index_int,1>(1,10,2) );
    CHECK( ( i1.first_index()==1 ) );
    CHECK( ( i1.last_actual_index()==9 ) );
    CHECK( i1.volume()==5 );
    SECTION( "bump" ) {
      REQUIRE_NOTHROW( op = ioperator<index_int,1>("<=1") );
    }
    SECTION( "mod" ) {
      REQUIRE_NOTHROW( op = ioperator<index_int,1>("<<1") );
    }
    REQUIRE_NOTHROW( i2 = i1.operate(op) );
    CHECK( ( i2.first_index()==0 ) );
    CHECK( ( i2.last_actual_index()==8 ) );
    CHECK( i2.volume()==5 );
  }
  SECTION( "test truncating by itself" ) {
    i1 = contiguous_indexstruct<index_int,1>( 1,10 ) );
    REQUIRE_NOTHROW( i2 = i1.truncate_left(4) );
    CHECK( i2.is_contiguous() );
    CHECK( ( i2.first_index()==4 ) );
  }
  SECTION( "shift strided with truncate" ) {
    i1 = strided_indexstruct<index_int,1>(0,10,2) );
    REQUIRE_NOTHROW( op = ioperator<index_int,1>("<=1") );
    REQUIRE_NOTHROW( i2 = i1.operate(op,0,100) );
    CHECK( ( i2.first_index()==1 ) );
    CHECK( ( i2.last_actual_index()==9 ) );
    CHECK( i2.volume()==5 );
    REQUIRE_NOTHROW( I1 = indexstructure<index_int,1>(i1) );
    REQUIRE_NOTHROW( I2 = I1.operate(op,0,100) );
    CHECK( I2.volume()==5 );
  }
#endif
}

TEST_CASE( "division operation 1D","[operate][31]" ) {
  indexstructure<index_int,1> i1,i2; ioperator<index_int,1> op;

  SECTION( "simple division" ) {
    REQUIRE_NOTHROW( op = ioperator<index_int,1>("/2") );
  
    SECTION( "contiguous1" ) {
      indexstructure<index_int,1> i1{ contiguous_indexstruct<index_int,1>(0,10) };
      REQUIRE_NOTHROW( i2 = i1.operate(op) );
      CHECK( ( i2.first_index()==0 ) );
      CHECK( ( i2.last_actual_index()==5 ) );
    }
    SECTION( "contiguous2" ) {
    indexstructure<index_int,1> i1{ contiguous_indexstruct<index_int,1>(0,9) };
    REQUIRE_NOTHROW( i2 = i1.operate(op) );
    CHECK( ( i2.first_index()==0 ) );
    CHECK( ( i2.last_actual_index()==4 ) );
    }
  }
  SECTION( "contiguous division" ) {
    REQUIRE_NOTHROW( op = ioperator<index_int,1>(":2") );
    
    SECTION( "contiguous1" ) {
      indexstructure<index_int,1> i1{ contiguous_indexstruct<index_int,1>(0,10) };
      REQUIRE_NOTHROW( i2 = i1.operate(op) );
      CHECK( ( i2.first_index()==0 ) );
      CHECK( ( i2.last_actual_index()==4 ) );
    }
    SECTION( "contiguous2" ) {
      indexstructure<index_int,1> i1{ contiguous_indexstruct<index_int,1>(0,9) };
      REQUIRE_NOTHROW( i2 = i1.operate(op) );
      CHECK( ( i2.first_index()==0 ) );
      CHECK( ( i2.last_actual_index()==4 ) );
    }
  }
}

TEST_CASE( "division operation 2D","[operate][31]" ) {
  indexstructure<index_int,2> i1,i2; ioperator<index_int,2> op;

  SECTION( "simple division" ) {
    REQUIRE_NOTHROW( op = ioperator<index_int,2>("/2") );
  
    SECTION( "exact divisible" ) {
      coordinate<index_int,2> layout( array<index_int,2>{1,1} );
      indexstructure<index_int,2> i1
	{ contiguous_indexstruct<index_int,2>( layout*0,layout*10 ) };
      INFO( "original struct" << i1.as_string() );
      REQUIRE_NOTHROW( i2 = i1.operate(op) );
      INFO( "divided struct" << i2.as_string() );
      CHECK( ( i2.first_index()==layout*0 ) );
      CHECK( ( i2.last_actual_index()==layout*5 ) );
    }
    SECTION( "round down" ) {
      coordinate<index_int,2> layout( array<index_int,2>{1,1} );
      indexstructure<index_int,2> i1
	{ contiguous_indexstruct<index_int,2>( layout*0,layout*9 ) };
      INFO( "original struct" << i1.as_string() );
      REQUIRE_NOTHROW( i2 = i1.operate(op) );
      INFO( "divided struct" << i2.as_string() );
      auto i2last = i2.last_actual_index();
      CHECK( ( i2.first_index()==layout*0 ) );
      CHECK( ( i2last==layout*4 ) );
    }
  }
  // SECTION( "contiguous division" ) {
  //   REQUIRE_NOTHROW( op = ioperator<index_int,2>(":2") );
    
  //   SECTION( "contiguous1" ) {
  //     indexstructure<index_int,2> i1{ contiguous_indexstruct<index_int,2>(0,10) };
  //     REQUIRE_NOTHROW( i2 = i1.operate(op) );
  //     CHECK( ( i2.first_index()==0 ) );
  //     CHECK( ( i2.last_actual_index()==4 ) );
  //   }
  //   SECTION( "contiguous2" ) {
  //     indexstructure<index_int,2> i1{ contiguous_indexstruct<index_int,2>(0,9) };
  //     REQUIRE_NOTHROW( i2 = i1.operate(op) );
  //     CHECK( ( i2.first_index()==0 ) );
  //     CHECK( ( i2.last_actual_index()==4 ) );
  //   }
  // }
}

#if 0
TEST_CASE( "copy indexstruct<index_int,1>","[indexstruct<index_int,1>][copy][42]" ) {
  indexstruct<index_int,1> i1,i2;

  i1 = indexstruct<index_int,1>{ new contiguous_indexstruct<index_int,1>(0,10) };
  REQUIRE_NOTHROW( i2 = i1.make_clone() );
  // make a copy
  CHECK( i1.volume()==i2.volume() );
  CHECK( ( i1.first_index()==i2.first_index() ) );
  CHECK( ( i1.last_actual_index()==i2.last_actual_index() ) );

  // shift the original
  REQUIRE_NOTHROW( i1 = i1.translate_by(1) );
  CHECK( i1.volume()==i2.volume() );
  CHECK( ( i1.first_index()==i2.first_index()+1 ) );
  CHECK( ( i1.last_actual_index()==i2.last_actual_index()+1 ) );
}

TEST_CASE( "big shift operators","[operator][shift][43]" ) {
  ioperator<index_int,1> i;
  i = ioperator<index_int,1>("shift",5);
  CHECK( i.operate(6)==11 );
  CHECK_NOTHROW( i.operate(-11)==-6 );
  i = ioperator<index_int,1>("shift",-3);
  CHECK( i.operate(6)==3 );
  CHECK( i.operate(1)==-2 );
  CHECK_NOTHROW( i.operate(-6)==-2 );  
}

TEST_CASE( "arbitrary shift of indexstruct<index_int,1>","[indexstruct<index_int,1>][shift][44]" ) {
  indexstruct<index_int,1> i1,i2;
  indexstructure<index_int,1> I1,I2;
  i1 = contiguous_indexstruct<index_int,1>(5,7) );
  REQUIRE_NOTHROW( I1 = indexstructure<index_int,1>(i1) );
  SECTION( "shift by" ) {
    REQUIRE_NOTHROW( i2 = i1.operate( ioperator<index_int,1>("shift",7) ) );
    CHECK( ( i2.first_index()==12 ) );
    CHECK( ( i2.last_actual_index()==14 ) );
    REQUIRE_NOTHROW( I2 = I1.operate( ioperator<index_int,1>("shift",7) ) );
    CHECK( ( I2.first_index()==12 ) );
    CHECK( ( I2.last_actual_index()==14 ) );
  }
  // SECTION( "shift to" ) {
  //   REQUIRE_NOTHROW( i2 = i1.operate( ioperator<index_int,1>("shiftto",7) ) );
  //   CHECK( ( i2.first_index()==7 ) );
  //   CHECK( ( i2.last_actual_index()==9 ) );
  // }
}

TEST_CASE( "sigma operator stuff","[50]" ) {
  sigma_operator<index_int,1> sop;
  ioperator<index_int,1> times2("*2");
  auto cont = contiguous_indexstruct<index_int,1>(10,20) );
  indexstruct<index_int,1> sstruct; const char *path;
  SECTION( "point" ) { path = "operate by point";
    REQUIRE_NOTHROW( sop = sigma_operator<index_int,1>(times2) );
  }
  SECTION( "struct" ) { path = "operate by struct";
    REQUIRE_NOTHROW
      ( sop = sigma_operator<index_int,1>
	( [times2] ( const indexstruct<index_int,1> &i) . indexstruct<index_int,1>
	  { return i.operate(times2); } ) );
  }
  INFO( "path: " << path );
  REQUIRE_NOTHROW( sstruct = cont.operate(sop) );
  CHECK( ( sstruct.first_index()==20 ) );
  CHECK( ( sstruct.last_actual_index()==40 ) );
  indexstructure<index_int,1> Cont,Sstruct;
  REQUIRE_NOTHROW( Cont = indexstructure<index_int,1>(cont) );
  REQUIRE_NOTHROW( Sstruct = Cont.operate(sop) );
  CHECK( ( Sstruct.first_index()==20 ) );
  CHECK( ( Sstruct.last_actual_index()==40 ) );
}

#if 0
TEST_CASE( "create multidimensional by component","[multi][indexstruct<index_int,1>][100]" ) {
  shared_ptr<multi_indexstruct<index_int,1>> mi,mt;
  multi_indexstructure<index_int,1> Mi,Mt;

  // we can not create zero-dimensional
  REQUIRE_THROWS( mi = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>(0) ) );
  REQUIRE_THROWS( Mi = multi_indexstructure<index_int,1>(0) );
  // create two-dimensional
  REQUIRE_NOTHROW( mi = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>(2) ) );
  REQUIRE_NOTHROW( Mi = multi_indexstructure<index_int,1>(2) );
  // set two component
  REQUIRE_NOTHROW( mi.set_component
		   ( 0, contiguous_indexstruct<index_int,1>(2,10) ) ) );
  REQUIRE_NOTHROW( mi.set_component
		   ( 1, contiguous_indexstruct<index_int,1>(40,41) ) ) );
  CHECK( mi.volume()==9*2 );
  // can set twice
  REQUIRE_NOTHROW( mi.set_component
		   ( 1, contiguous_indexstruct<index_int,1>(30,41) ) ) );
  CHECK( mi.volume()==9*12 );
  // can not set outside dimension bounds
  REQUIRE_THROWS( mi.set_component
		  ( 2, contiguous_indexstruct<index_int,1>(40,41) ) ) );
  CHECK( mi.volume()==9*12 );
  domain_coordinate val(1);
  REQUIRE_NOTHROW( val = mi.local_size_r() );
  CHECK( val.at(0)==9 );
  CHECK( val.at(1)==12 );
  REQUIRE_NOTHROW( val = mi.first_index_r() );
  CHECK( val.at(0)==2 );
  CHECK( val.at(1)==30 );
  REQUIRE_NOTHROW( val = mi.last_actual_index_r() );
  CHECK( val.at(0)==10 );
  CHECK( val.at(1)==41 );

  mt = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>(2) );
  mt.set_component(0,contiguous_indexstruct<index_int,1>(4,9)) );
  mt.set_component(1,contiguous_indexstruct<index_int,1>(40,40)) );
  REQUIRE_NOTHROW( val = mi.linear_location_of(mt) );
  CHECK( val.at(0)==2 );
  CHECK( val.at(1)==10 );
}

TEST_CASE( "create multidimensional block","[multi][indexstruct<index_int,1>][101]" ) {
  int dim = 2; index_int nlocal = 10,v=1;
  multi_indexstruct<index_int,1> m(dim);
  vector<index_int> sizes(dim);
  for (int id=0; id<dim; id++) {
    sizes[id] = nlocal+id; v *= nlocal+id;
  }
  SECTION( "from sizes" ) {
    m = multi_indexstruct<index_int,1>(sizes);
  }
  SECTION( "from structs" ) {
    vector<indexstruct<index_int,1>> blocks(dim);
    for (int id=0; id<dim; id++)
      blocks[id] = contiguous_indexstruct<index_int,1>(0,nlocal+id-1) );
    m = multi_indexstruct<index_int,1>(blocks);
  }
  REQUIRE( m.volume()==v );
}

TEST_CASE( "multidimensional containment","[multi][111]" ) {
  shared_ptr<multi_indexstruct<index_int,1>> m1,m1a,m2,m2a;
  SECTION( "simple:" ) {
    REQUIRE_NOTHROW
      ( m1 = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>
	( vector<indexstruct<index_int,1>>{
	  contiguous_indexstruct<index_int,1>(0,9) ) } ) ) );
    INFO( "m1:" << m1.as_string() );
    SECTION( "contains simple" ) {
      REQUIRE_NOTHROW
	( m2 = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>
	  ( vector<indexstruct<index_int,1>>{
	    contiguous_indexstruct<index_int,1>(1,5) ) } ) ) );
      INFO( "m2:" << m2.as_string() );
      CHECK( m1.contains(m2) );
    }
    SECTION( "not contains simple" ) {
      REQUIRE_NOTHROW
	( m2 = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>
	  ( vector<indexstruct<index_int,1>>{
	    contiguous_indexstruct<index_int,1>(1,15) ) } ) ) );
      INFO( "m2:" << m2.as_string() );
      CHECK( !m1.contains(m2) );
    }
  }
  SECTION( "multi:" ) {
    REQUIRE_NOTHROW
      ( m1 = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>
	( vector<indexstruct<index_int,1>>{
	  contiguous_indexstruct<index_int,1>(0,9) ) } ) ) );
    REQUIRE_NOTHROW
      ( m1a = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>
	( vector<indexstruct<index_int,1>>{
	  contiguous_indexstruct<index_int,1>(20,29) ) } )
						  ) );
    REQUIRE_NOTHROW( m1 = m1.struct_union(m1a) );
    INFO( "m1:" << m1.as_string() );
    SECTION( "contains simple" ) {
      REQUIRE_NOTHROW
	( m2 = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>
	  ( vector<indexstruct<index_int,1>>{
	    contiguous_indexstruct<index_int,1>(1,5) ) } ) ) );
      INFO( "m2:" << m2.as_string() );
      CHECK( m1.contains(m2) );
    }
    SECTION( "not contains simple" ) {
      REQUIRE_NOTHROW
	( m2 = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>
	  ( vector<indexstruct<index_int,1>>{
	    contiguous_indexstruct<index_int,1>(5,25) ) } ) ) );
      INFO( "m2:" << m2.as_string() );
      CHECK( !m1.contains(m2) );
    }
  }
}

TEST_CASE( "True multidimensional containment","[multi][112]" ) {
  shared_ptr<multi_indexstruct<index_int,1>> m1,m1a,m2,m2a;
  SECTION( "simple:" ) {
    REQUIRE_NOTHROW
      ( m1 = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>
	( vector<indexstruct<index_int,1>>{
	  contiguous_indexstruct<index_int,1>(0,9) ),
	  contiguous_indexstruct<index_int,1>(0,9) )
	    } ) ) );
    INFO( "m1:" << m1.as_string() );
    SECTION( "contains simple" ) {
      REQUIRE_NOTHROW
	( m2 = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>
	  ( vector<indexstruct<index_int,1>>{
	    contiguous_indexstruct<index_int,1>(1,5) ),
	    contiguous_indexstruct<index_int,1>(1,5) )
	      } ) ) );
      INFO( "m2:" << m2.as_string() );
      CHECK( m1.contains(m2) );
    }
    SECTION( "not contains simple" ) {
      REQUIRE_NOTHROW
	( m2 = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>
	  ( vector<indexstruct<index_int,1>>{
	    contiguous_indexstruct<index_int,1>(1,15) ),
	    contiguous_indexstruct<index_int,1>(1,15) )
	      } ) ) );
      INFO( "m2:" << m2.as_string() );
      CHECK( !m1.contains(m2) );
    }
  }
  SECTION( "multi:" ) {
    REQUIRE_NOTHROW
      ( m1 = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>
	( vector<indexstruct<index_int,1>>{
	  contiguous_indexstruct<index_int,1>(0,9) ),
	  contiguous_indexstruct<index_int,1>(0,9) )
	    } ) ) );
    REQUIRE_NOTHROW
      ( m1a = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>
	( vector<indexstruct<index_int,1>>{
	  contiguous_indexstruct<index_int,1>(20,29) ),
	  contiguous_indexstruct<index_int,1>(20,29) )
	    } ) )
	);
    REQUIRE_NOTHROW( m1 = m1.struct_union(m1a) );
    INFO( "m1:" << m1.as_string() );
    SECTION( "contains simple" ) {
      REQUIRE_NOTHROW
	( m2 = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>
	  ( vector<indexstruct<index_int,1>>{
	    contiguous_indexstruct<index_int,1>(1,5) ),
	    contiguous_indexstruct<index_int,1>(1,5) )
	      } ) ) );
      INFO( "m2:" << m2.as_string() );
      CHECK( m1.contains(m2) );
    }
    SECTION( "not contains simple" ) {
      REQUIRE_NOTHROW
	( m2 = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>
	  ( vector<indexstruct<index_int,1>>{
	    contiguous_indexstruct<index_int,1>(5,25) ),
	    contiguous_indexstruct<index_int,1>(5,25) )
	      } ) ) );
      INFO( "m2:" << m2.as_string() );
      CHECK( !m1.contains(m2) );
    }
  }
}

TEST_CASE( "multidimensional union","[multi][union][indexstruct<index_int,1>][113]" ) {
  shared_ptr<multi_indexstruct<index_int,1>> mi1,mi2,mi3; int diff;
  REQUIRE_NOTHROW( mi1 = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>(1) ) );
  REQUIRE_NOTHROW( mi2 = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>(2) ) );
  REQUIRE_THROWS( mi1.struct_union(mi2) );

  REQUIRE_NOTHROW( mi1 = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>(2) ) );
  REQUIRE_NOTHROW( mi1.set_component
		   ( 0, contiguous_indexstruct<index_int,1>(10,20) ) ) );
  REQUIRE_NOTHROW( mi1.set_component
		   ( 1, contiguous_indexstruct<index_int,1>(30,40) ) ) );

  SECTION( "fit in dimension 1: multis are merged" ) {
    REQUIRE_NOTHROW( mi2.set_component
		     ( 0, contiguous_indexstruct<index_int,1>(10,20) ) ) );
    REQUIRE_NOTHROW( mi2.set_component
		     ( 1, contiguous_indexstruct<index_int,1>(31,41) ) ) );
    CHECK( mi1.can_union_in_place(mi2,diff) );
    CHECK( mi2.can_union_in_place(mi1,diff) );
    REQUIRE_NOTHROW( mi3 = mi1.struct_union(mi2).force_simplify() );
    INFO( format("Union {} and {} gives {}",
		      mi1.as_string(),mi2.as_string(),mi3.as_string()) );
    CHECK( !mi3.is_multi() );
    CHECK( mi3.first_index_r()==domain_coordinate( vector<index_int>{10,30} ) );
    CHECK( mi3.last_actual_index_r()==domain_coordinate( vector<index_int>{20,41} ) );
  }

  SECTION( "fit in dimension 2: multis are merged" ) {
    REQUIRE_NOTHROW( mi2.set_component
		     ( 0, contiguous_indexstruct<index_int,1>(12,22) ) ) );
    REQUIRE_NOTHROW( mi2.set_component
		     ( 1, contiguous_indexstruct<index_int,1>(30,40) ) ) );
    CHECK( mi1.can_union_in_place(mi2,diff) );
    CHECK( mi2.can_union_in_place(mi1,diff) );
    REQUIRE_NOTHROW( mi3 = mi1.struct_union(mi2).force_simplify() );
    CHECK( !mi3.is_multi() );
    CHECK( mi3.first_index_r()==domain_coordinate( vector<index_int>{10,30} ) );
    CHECK( mi3.last_actual_index_r()==domain_coordinate( vector<index_int>{22,40} ) );
  }

  SECTION( "unfit: first comes from one and last second original" ) {
    // m1 = [ (10,30) - (20,40) ]
    // m2 = [ (12,31) - (22,41) ]  so we store pointers to both
    // first = (10,30) last = (22,41)
    REQUIRE_NOTHROW( mi2.set_component
  		     ( 0, contiguous_indexstruct<index_int,1>(12,22) ) ) );
    REQUIRE_NOTHROW( mi2.set_component
  		     ( 1, contiguous_indexstruct<index_int,1>(31,41) ) ) );
    REQUIRE_NOTHROW( mi3 = mi1.struct_union(mi2) );
    INFO( format("{} & {} gives {}",mi1.as_string(),mi2.as_string(),mi3.as_string()) );
    CHECK( mi3.is_multi() );
    auto first = mi3.first_index_r(), last = mi3.last_actual_index_r();
    INFO( format("first = {}, last = {}",first.as_string(),last.as_string()) );
    CHECK( first==domain_coordinate( vector<index_int>{10,30} ) );
    CHECK( last==domain_coordinate( vector<index_int>{22,41} ) );
  }
  SECTION( "very unfit: case where union first/last are not in the originals" ) {
    // m1 = [ (10,30) - (20,40) ]
    // m2 = [ (20,10) - (30,20) ]  so we store pointers to both
    // first = (10,10) last = (30,40)
    REQUIRE_NOTHROW( mi2.set_component
  		     ( 0, contiguous_indexstruct<index_int,1>(20,30) ) ) );
    REQUIRE_NOTHROW( mi2.set_component
  		     ( 1, contiguous_indexstruct<index_int,1>(10,20) ) ) );
    REQUIRE_NOTHROW( mi3 = mi1.struct_union(mi2) );
    CHECK( mi3.is_multi() );
    auto first = mi3.first_index_r(), last = mi3.last_actual_index_r();
    INFO( format("first = {}, last = {}",first.as_string(),last.as_string()) );
    CHECK( first==domain_coordinate( vector<index_int>{10,10} ) );
    CHECK( last==domain_coordinate( vector<index_int>{30,40} ) );
  }
  SECTION( "union of more than two" ) {
    auto
      m1 = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>
      ( vector<indexstruct<index_int,1>>
	{ indexstruct<index_int,1>(new contiguous_indexstruct<index_int,1>(1,1)),
	    indexstruct<index_int,1>(new contiguous_indexstruct<index_int,1>(1,1))
	    } ) ),
      m2 = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>
      ( vector<indexstruct<index_int,1>>
	{ indexstruct<index_int,1>(new contiguous_indexstruct<index_int,1>(2,2)),
	    indexstruct<index_int,1>(new contiguous_indexstruct<index_int,1>(2,2))
	    } ) ),
      m3 = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>
      ( vector<indexstruct<index_int,1>>
	{ indexstruct<index_int,1>(new contiguous_indexstruct<index_int,1>(3,3)),
	    indexstruct<index_int,1>(new contiguous_indexstruct<index_int,1>(3,3))
	    } ) );
    shared_ptr<multi_indexstruct<index_int,1>> u;
    // m2 makes it a multi
    CHECK( !m1.contains(m2) );
    REQUIRE_NOTHROW( u = m1.struct_union(m2) );
    INFO( format("union 1 & 2: {}",u.as_string()) );
    CHECK( u.multi_size()==2 );
    // m3 is added as another multi
    CHECK( !u.contains(m3) );
    REQUIRE_NOTHROW( u = u.struct_union(m3) );
    INFO( format("union 1 & 2 & 3: {}",u.as_string()) );
    CHECK( u.multi_size()==3 );
  }
  SECTION( "multi union with recombination" ) {
    auto 
      m1 = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>
      ( vector<indexstruct<index_int,1>>
	{ indexstruct<index_int,1>(new contiguous_indexstruct<index_int,1>(1,2)),
	    indexstruct<index_int,1>(new contiguous_indexstruct<index_int,1>(1,1))
	    } ) ),
      m2 = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>
      ( vector<indexstruct<index_int,1>>
	{ indexstruct<index_int,1>(new contiguous_indexstruct<index_int,1>(2,2)),
	    indexstruct<index_int,1>(new contiguous_indexstruct<index_int,1>(2,2))
	    } ) );
    shared_ptr<multi_indexstruct<index_int,1>> m3,u;
    REQUIRE_NOTHROW( u = m1.struct_union(m2) );
    INFO( format("union 1 & 2: {}",u.as_string()) );
    CHECK( u.multi_size()==2 );
    SECTION( "absorb" ) {
      REQUIRE_NOTHROW( m3 = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>
	( vector<indexstruct<index_int,1>>
	  { indexstruct<index_int,1>(new contiguous_indexstruct<index_int,1>(2,2)),
	      indexstruct<index_int,1>(new contiguous_indexstruct<index_int,1>(1,1))
	      } ) ) );
      REQUIRE_NOTHROW( u = u.struct_union(m3).force_simplify() );
      INFO( format("absorb {} gives union 1 & 2 & 3: {}",m3.as_string(),u.as_string()) );
      CHECK( u.is_multi() );
      CHECK( u.multi_size()==2 );
    }
    SECTION( "merge" ) {
      REQUIRE_NOTHROW( m3 = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>
	( vector<indexstruct<index_int,1>>
	  { indexstruct<index_int,1>(new contiguous_indexstruct<index_int,1>(1,1)),
	      indexstruct<index_int,1>(new contiguous_indexstruct<index_int,1>(2,2))
	      } ) ) );
      REQUIRE_NOTHROW( u = u.struct_union(m3).force_simplify() );
      INFO( format("merge {} give union 1 & 2 & 3: {}",m3.as_string(),u.as_string()) );
      CHECK( !u.is_multi() );
      CHECK( u.volume()==4 );
    }
  }
}

TEST_CASE( "multidimensional intersection","[multi][indexstruct<index_int,1>][114]" ) {
  shared_ptr<multi_indexstruct<index_int,1>> mi1,mi2,mi3;
  REQUIRE_NOTHROW( mi1 = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>(1) ) );
  REQUIRE_NOTHROW( mi2 = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>(2) ) );
  REQUIRE_THROWS( mi1.struct_union(mi2) );

  REQUIRE_NOTHROW( mi1 = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>(2) ) );
  REQUIRE_NOTHROW( mi1.set_component
		   ( 0, contiguous_indexstruct<index_int,1>(0,10) ) ) );
  REQUIRE_NOTHROW( mi1.set_component
		   ( 1, contiguous_indexstruct<index_int,1>(0,10) ) ) );

  REQUIRE_NOTHROW( mi2.set_component
		   ( 0, contiguous_indexstruct<index_int,1>(10,20) ) ) );
  REQUIRE_NOTHROW( mi2.set_component
		   ( 1, contiguous_indexstruct<index_int,1>(10,20) ) ) );
  REQUIRE_NOTHROW( mi3 = mi1.intersect(mi2) );

  INFO( format("Intersection runs {}--{}",
		    mi3.first_index_r().as_string(),
		    mi3.last_actual_index_r().as_string()) );
  CHECK( mi3.volume()==1 );
}

TEST_CASE( "multi-dimensional minus","[multi][indexstruct<index_int,1>][115]" ) {
  shared_ptr<multi_indexstruct<index_int,1>> one,two,three;
  index_int lo = 1, hi = 4; bool success{true};
  REQUIRE_NOTHROW( one = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>
		   ( vector< indexstruct<index_int,1> >
		     { contiguous_indexstruct<index_int,1>(1,2) ),
			 contiguous_indexstruct<index_int,1>(5,8) )
			 }
		     )
							     ) );
  SECTION( "royally fits" ) {
    REQUIRE_NOTHROW( two = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>
		     ( vector< indexstruct<index_int,1> >
		       { contiguous_indexstruct<index_int,1>(0,3) ),
			   contiguous_indexstruct<index_int,1>(6,8) )
			   }
		       )
							       ) );
  }
  SECTION( "royally fits" ) {
    REQUIRE_NOTHROW( two = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>
  		     ( vector< indexstruct<index_int,1> >
  		       { contiguous_indexstruct<index_int,1>(1,2) ),
  			   contiguous_indexstruct<index_int,1>(6,8) )
  			   }
  		       )
  							       ) );
  }
  SECTION( "no fits" ) {
    REQUIRE_NOTHROW( two = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>
  		     ( vector< indexstruct<index_int,1> >
  		       { contiguous_indexstruct<index_int,1>(2,2) ),
  			   contiguous_indexstruct<index_int,1>(6,8) )
  			   }
  		       )
  							       ) );
    success = false;
  }
  if (!success) { // this used to be unimplemented
    REQUIRE_NOTHROW( three = one.minus(two) );
  } else {
    REQUIRE_NOTHROW( three = one.minus(two) );
    INFO( format("minus result: {}",three.as_string()) );
    CHECK( three.get_component(0).equals
	   ( contiguous_indexstruct<index_int,1>(1,2) ) ) );
    CHECK( three.get_component(1).equals
	   ( contiguous_indexstruct<index_int,1>(5,5) ) ) );
  }
}

TEST_CASE( "find linear location","[multi][locate][indexstruct<index_int,1>][116]" ) {
  shared_ptr<multi_indexstruct<index_int,1>> outer,inner;
  index_int location;

  SECTION( "one-d" ) {
    outer = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>(1) );
    SECTION( "outer contiguous" ) {
      outer.set_component
	(0,contiguous_indexstruct<index_int,1>(11,20)) );
      SECTION( "same dimension" ) {
	inner = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>(2) );
	inner.set_component
	  (0,contiguous_indexstruct<index_int,1>(15,16)) );
	inner.set_component
	  (1,contiguous_indexstruct<index_int,1>(8,9)) );
	REQUIRE_THROWS( location = outer.linear_location_of(inner) );
      }
      SECTION( "proper" ) {
	inner = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>(1) );
	inner.set_component
	  (0,contiguous_indexstruct<index_int,1>(15,16)) );
	REQUIRE_NOTHROW( location = outer.linear_location_of(inner) );
	CHECK( location==(15-11) );
      }
    }
  }
  SECTION( "two-d" ) {
    outer = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>(2) );
    SECTION( "outer contiguous" ) {
      outer.set_component  // 10 deep
	(0,contiguous_indexstruct<index_int,1>(11,20)) );
      outer.set_component // 4 wide
	(1,contiguous_indexstruct<index_int,1>(5,8)) );
      SECTION( "same dimension" ) {
	inner = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>(1) );
	inner.set_component
	  (0,contiguous_indexstruct<index_int,1>(15,16)) );
	REQUIRE_THROWS( location = outer.linear_location_of(inner) );
      }
      SECTION( "proper" ) {
	inner = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>(2) );
	inner.set_component
	  (0,contiguous_indexstruct<index_int,1>(15,16)) );
	inner.set_component
	  (1,contiguous_indexstruct<index_int,1>(6,8)) );
	REQUIRE_NOTHROW( location = outer.linear_location_of(inner) );
	CHECK( location==(15-11)*4+6-5 );
      }
    }
  }
}

TEST_CASE( "relativize and find linear location","[multi][locate][indexstruct<index_int,1>][117]" ) {
  // this uses the same structure as [112]
  shared_ptr<multi_indexstruct<index_int,1>> outer,inner,relative;
  indexstruct<index_int,1> relativec;
  index_int location;

  SECTION( "one-d" ) {
    outer = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>(1) );
    SECTION( "outer contiguous" ) {
      outer.set_component
	(0,contiguous_indexstruct<index_int,1>(11,20)) );
      SECTION( "same dimension" ) {
	inner = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>(2) );
	inner.set_component
	  (0,contiguous_indexstruct<index_int,1>(15,16)) );
	inner.set_component
	  (1,contiguous_indexstruct<index_int,1>(8,9)) );
	REQUIRE_THROWS( relative = inner.relativize_to(outer) );
      }
      SECTION( "proper" ) {
	inner = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>(1) );
	inner.set_component
	  (0,contiguous_indexstruct<index_int,1>(15,16)) );
	REQUIRE_NOTHROW( relative = inner.relativize_to(outer) );
	REQUIRE_NOTHROW( relativec = relative.get_component(0) );
	REQUIRE_NOTHROW( relativec.is_contiguous() );
	REQUIRE_NOTHROW( relative.first_index(0)==4 );
      }
    }
  }
  SECTION( "two-d" ) {
    outer = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>(2) );
    SECTION( "outer contiguous" ) {
      outer.set_component
	(0,contiguous_indexstruct<index_int,1>(11,20)) );
      outer.set_component
	(1,contiguous_indexstruct<index_int,1>(5,8)) );
      SECTION( "same dimension" ) {
	inner = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>(1) );
	inner.set_component
	  (0,contiguous_indexstruct<index_int,1>(15,16)) );
	REQUIRE_THROWS( relative = inner.relativize_to(outer) );
      }
      SECTION( "proper" ) {
	inner = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>(2) );
	inner.set_component
	  (0,contiguous_indexstruct<index_int,1>(15,16)) );
	inner.set_component
	  (1,contiguous_indexstruct<index_int,1>(6,8)) );
	REQUIRE_NOTHROW( relative = inner.relativize_to(outer) );
	REQUIRE_NOTHROW( relativec = relative.get_component(0) );
	REQUIRE_NOTHROW( relativec.is_contiguous() );
	REQUIRE_NOTHROW( relativec = relative.get_component(1) );
	REQUIRE_NOTHROW( relativec.is_contiguous() );
	REQUIRE_NOTHROW( relative.first_index(0)==4 );
	REQUIRE_NOTHROW( relative.first_index(1)==1 );
      }
    }
  }
}

TEST_CASE( "multi-dimensional operations","[multi][120]" ) {
  auto s1 = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>(2) );
  shared_ptr<multi_indexstruct<index_int,1>> s2;
  indexstruct<index_int,1> i0,i1;
  
  s1.set_component
    (0,indexstruct<index_int,1>(  new contiguous_indexstruct<index_int,1>(10,20) ) );
  s1.set_component
    (1,indexstruct<index_int,1>(  new strided_indexstruct<index_int,1>(25,50,5) ) );

  multi_ioperator *op; REQUIRE_NOTHROW( op = new multi_ioperator(2) );
  coordinate<index_int,1> one; one.set(1);
  CHECK( op.dimensionality()==2 );

  SECTION( "first dim" ) {
    REQUIRE_NOTHROW( op.set_operator(0,ioperator(">>2")) );
    REQUIRE_NOTHROW( s2 = s1.operate(op) );
    REQUIRE_NOTHROW( i0 = s2.get_component(0) );
    CHECK( i0.is_contiguous() );
    CHECK( ( i0.first_index()==12 ) );
    CHECK( ( i0.last_actual_index()==22 ) );
    REQUIRE_NOTHROW( i1 = s2.get_component(1) );
    CHECK( !i1.is_contiguous() );
    CHECK( i1.is_strided() );
    CHECK( ( i1.first_index()==25 ) );
    CHECK( ( i1.last_actual_index()==50 ) );
    CHECK( i1.stride()==one*5 );
  }
  SECTION( "second dim" ) {
    REQUIRE_NOTHROW( op.set_operator(1,ioperator(">>2")) );
    REQUIRE_NOTHROW( s2 = s1.operate(op) );
    REQUIRE_NOTHROW( i0 = s2.get_component(0) );
    CHECK( i0.is_contiguous() );
    CHECK( ( i0.first_index()==10 ) );
    CHECK( ( i0.last_actual_index()==20 ) );
    REQUIRE_NOTHROW( i1 = s2.get_component(1) );
    CHECK( !i1.is_contiguous() );
    CHECK( i1.is_strided() );
    CHECK( ( i1.first_index()==27 ) );
    CHECK( ( i1.last_actual_index()==52 ) );
    CHECK( i1.stride()==one*5 );
  }
  SECTION( "both dimensions" ) {
    auto div2 = ioperator(">>2");
    REQUIRE_NOTHROW( op.set_operator(0,div2) );
    REQUIRE_NOTHROW( op.set_operator(1,div2) );
    REQUIRE_NOTHROW( s2 = s1.operate(op,s1) );
    REQUIRE_NOTHROW( i0 = s2.get_component(0) );
    REQUIRE_NOTHROW( i1 = s2.get_component(1) );
    CHECK( ( i0.first_index()==12 ) );
    CHECK( ( i0.last_actual_index()==20 ) );
    CHECK( ( i1.first_index()==27 ) );
    CHECK( ( i1.last_actual_index()==47 ) );
  }
  SECTION( "operate with truncation" ) {
  }
}

TEST_CASE( "multi dimensional iteration, 1d","[multi][range][125]" ) {
  int dim = 1; index_int f=5, l=8;

  multi_indexstruct<index_int,1> segment
    ( contiguous_indexstruct<index_int,1>(f,l) ) );
  multi_indexstruct<index_int,1> begin(1),end(1);
  REQUIRE_NOTHROW( begin = segment.begin() );
  CHECK( (*begin).at(0)==f );
  REQUIRE_NOTHROW( end = segment.end() );
  CHECK( (*end).at(0)==l+1 );
  CHECK( segment.dimensionality()==dim );
  int count = f;
  for ( auto ii=begin; !( ii==end ); ++ii) {
    domain_coordinate i = *ii;
    //print("ranged: {}\n",i.as_string());
    CHECK( i[0]==count );
    count++;
  }
  CHECK( count==f+segment.volume() );
}

TEST_CASE( "multi dimensional iteration, 2d","[multi][range][126]" ) {
  int d;

  multi_indexstruct<index_int,1> segment = contiguous_multi_indexstruct<index_int,1>
    ( domain_coordinate(vector<index_int>{5,5}),
      domain_coordinate(vector<index_int>{6,6}) );
  multi_indexstruct<index_int,1> begin(1),end(1);
  REQUIRE_NOTHROW( begin = segment.begin() );
  CHECK( (*begin).at(0)==5 );
  CHECK( (*begin).at(1)==5 );
  REQUIRE_NOTHROW( end = segment.end() );
  CHECK( (*end).at(0)==7 );
  CHECK( (*end).at(1)==5 );
  int count = 0;
  for ( auto ii=begin; !( ii==end ); ++ii) {
    domain_coordinate i = *ii;
    //print("ranged: {}\n",i.as_string());
    count++;
  }
  CHECK( count==segment.volume() );
}

TEST_CASE( "operate domain_coordinates","[multi][coordinate][operate][130]" ) {
  domain_coordinate coord( vector<index_int>{10,20,30} ),
    shifted(3),divided(3),multiplied(3);
  
  REQUIRE_NOTHROW( shifted = coord+2 );
  CHECK( shifted.dimensionality()==3 );
  CHECK( shifted[0]==12 );
  CHECK( shifted[1]==22 );
  CHECK( shifted[2]==32 );

  REQUIRE_NOTHROW( multiplied = coord*2 );
  CHECK( multiplied.dimensionality()==3 );
  CHECK( multiplied[0]==20 );
  CHECK( multiplied[1]==40 );
  CHECK( multiplied[2]==60 );

  REQUIRE_NOTHROW( divided = coord/2 );
  CHECK( divided.dimensionality()==3 );
  CHECK( divided[0]==05 );
  CHECK( divided[1]==10 );
  CHECK( divided[2]==15 );

  REQUIRE_NOTHROW( shifted=domain_coordinate(multiplied+divided));
  CHECK( shifted.dimensionality()==3 );
  CHECK( shifted[0]==25 );
  CHECK( shifted[1]==50 );
  CHECK( shifted[2]==75 );

  multi_shift_operator mshift1( vector<index_int>{1,2,3} );
  REQUIRE_NOTHROW( shifted = mshift1.operate(&coord) );
  CHECK( shifted[0]==11 );
  CHECK( shifted[1]==22 );
  CHECK( shifted[2]==33 );
}

TEST_CASE( "domain_coordinate operations","[multi][operator][coordinate][131]" ) {
  domain_coordinate
    one( vector<index_int>{10,20,30} ),
    two( vector<index_int>{11,21,31} ),
    three( vector<index_int>{20,40,60} );
  CHECK( two==one+1 );
  CHECK( one==two-1 );
  CHECK( three==one*2 );
}

TEST_CASE( "operate multi indexstruct<index_int,1>","[multi][operate][132]" ) {
  shared_ptr<multi_indexstruct<index_int,1>> brick,sh_brick;

  REQUIRE_NOTHROW( brick = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>(3) ) );
  brick.set_component
    ( 0,indexstruct<index_int,1>(  new contiguous_indexstruct<index_int,1>(10,20) ) );
  brick.set_component
    ( 1,indexstruct<index_int,1>(  new strided_indexstruct<index_int,1>(25,50,5) ) );
  brick.set_component
    ( 2,indexstruct<index_int,1>(  new strided_indexstruct<index_int,1>(5,7,2) ) );

  // constructing a beta; see signature_function
  multi_shift_operator mshift( vector<index_int>{1,2,3} );
  REQUIRE_NOTHROW( sh_brick = mshift.operate(brick) );
  CHECK( sh_brick.first_index_r()==domain_coordinate( vector<index_int>{11,27,8} ) );
}

TEST_CASE( "Bilinear shift test","[operate][multi][133]" ) {
  int dim = 2;
  vector<multi_ioperator*> ops;

  const char *path; int ipath;
  SECTION( "id" ) { path = "id"; ipath = 1;
    ops.push_back( new multi_shift_operator(vector<index_int>{ 0, 0}) );
  }
  SECTION( "extend along x axis" ) { path = "extend x"; ipath = 2;
    ops.push_back( new multi_shift_operator(vector<index_int>{ 0, 0}) );
    ops.push_back( new multi_shift_operator(vector<index_int>{ 0,+1}) );
  }
  SECTION( "extend along y axis" ) { path = "extend y"; ipath = 3;
    ops.push_back( new multi_shift_operator(vector<index_int>{ 0, 0}) );
    ops.push_back( new multi_shift_operator(vector<index_int>{+1, 0}) );
  }
  SECTION( "extend left is truncated out" ) { path = "extend truncate x"; ipath = 4;
    ops.push_back( new multi_shift_operator(vector<index_int>{ 0, 0}) );
    ops.push_back( new multi_shift_operator(vector<index_int>{ 0,-1}) );
  }
  SECTION( "extend up is truncated out" ) { path = "extend truncate y"; ipath = 5;
    ops.push_back( new multi_shift_operator(vector<index_int>{ 0, 0}) );
    ops.push_back( new multi_shift_operator(vector<index_int>{-1, 0}) );
  }
  SECTION( "extend by diagonal shift" ) { path = "extend diag"; ipath = 6;
    ops.push_back( new multi_shift_operator(vector<index_int>{ 0, 0}) );
    ops.push_back( new multi_shift_operator(vector<index_int>{+1,+1}) );
  }
    // ops.push_back( new multi_shift_operator(vector<index_int>{-1,+1}) );
    // ops.push_back( new multi_shift_operator(vector<index_int>{+1,+1}) );
    // ops.push_back( new multi_shift_operator(vector<index_int>{-1,-1}) );
    // ops.push_back( new multi_shift_operator(vector<index_int>{+1,-1}) );
  INFO( "path: " << path );

  shared_ptr<multi_indexstruct<index_int,1>> gamma_struct,
    halo_struct = shared_ptr<multi_indexstruct<index_int,1>>( new empty_multi_indexstruct<index_int,1>(dim) ),
    truncation = shared_ptr<multi_indexstruct<index_int,1>>( new contiguous_multi_indexstruct<index_int,1>
    ( domain_coordinate( vector<index_int>{0,0} ),
      domain_coordinate( vector<index_int>{11,11} ) ) );

  gamma_struct = shared_ptr<multi_indexstruct<index_int,1>>( new contiguous_multi_indexstruct<index_int,1>
    ( domain_coordinate( vector<index_int>{0,0} ),
      domain_coordinate( vector<index_int>{10,10} ) ) );

  { // code from signature_function::make_beta_struct_from_ops
    for ( auto beta_op : ops ) {
      shared_ptr<multi_indexstruct<index_int,1>> beta_struct;
      if (beta_op.is_modulo_op() || truncation==nullptr) {
	beta_struct = gamma_struct.operate(beta_op);
      } else {
	beta_struct = gamma_struct.operate(beta_op,truncation);
      }
      // print("{} Struc: {} op: {} gives {}\n",
      // 		 pcoord.as_string(),
      //	 gamma_struct.as_string(),beta_op.as_string(),beta_struct.as_string());
      if (!beta_struct.is_empty()) {
	halo_struct = halo_struct.struct_union(beta_struct);
	//print("... union={}\n",halo_struct.as_string());
      }
    }
  }

  memory_buffer w;
  format_to(w.end(),"Beta struct from {} by applying:",gamma_struct.as_string());
  for ( auto o : ops ) format_to(w.end()," {},",o.as_string());
  format_to(w.end()," is: {}",halo_struct.as_string());
  INFO(w.data());

  CHECK( !halo_struct.is_empty());
  if (ipath==1 || ipath==4 || ipath==5) {
    CHECK( halo_struct.volume()==gamma_struct.volume() );
  } else if (ipath==2 || ipath==3) {
    CHECK( halo_struct.volume()>gamma_struct.volume() );
  } else if (ipath==6) {
    //    CHECK_THROWS( halo_struct.volume()==gamma_struct.volume() );
    CHECK( halo_struct.is_multi() );
    CHECK( halo_struct.multi_size()==2 );
    CHECK( halo_struct.volume()>gamma_struct.volume() );
  }
}

TEST_CASE( "multi-dimensional derived types","[multi][139]" ) {
  domain_coordinate
    first( vector<index_int>{10,20,30} ),
    last( vector<index_int>{19,29,39} );
  CHECK( first.dimensionality()==3 );
  CHECK( first[0]==10 );
  CHECK( first[1]==20 );
  CHECK( first[2]==30 );
  multi_indexstruct<index_int,1> *brick;
  REQUIRE_NOTHROW( brick = new contiguous_multi_indexstruct<index_int,1>(first,last) );
  CHECK( brick.volume()==1000 );
}

TEST_CASE( "multi sigma operators","[multi][sigma][140]" ) {
  multi_sigma_operator op; int dim = 3; const char *path;

  SECTION( "operator by array of operators" ) { path = "array of operators";
    vector<ioperator> operators(dim);
    for (int id=0; id<dim; id++) {
      operators.at(id) = ioperator(">>2");
    }
    REQUIRE_NOTHROW( op = multi_sigma_operator(operators) );
    CHECK( !op.is_single_operator() );
    CHECK( op.is_point_operator() );
    CHECK( !op.is_coord_struct_operator() );
  }
  SECTION( "operator from pointwise" ) { path = "pointwise";
    REQUIRE_NOTHROW( op = multi_sigma_operator
		     (dim,[] (const domain_coordinate &in) . domain_coordinate {
		       int dim = in.dimensionality();
		       domain_coordinate out(dim);
		       for (int id=0; id<dim; id++)
			 out.set(id,in.coord(id)+2);
		       return out;
		     } ) );
    CHECK( !op.is_single_operator() );
    CHECK( op.is_point_operator() );
    CHECK( !op.is_coord_struct_operator() );
  }
  SECTION( "from coordinate-to-struct operator" ) { path="coord-to-struct";
    REQUIRE_NOTHROW
      ( op = multi_sigma_operator
	(dim,[] (const domain_coordinate &c) . shared_ptr<multi_indexstruct<index_int,1>> {
	  int dim = c.dimensionality();
	  //print("Apply multi sigma in d={}\n",dim);
	  auto m = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>(dim) );
	  for (int id=0; id<dim; id++) {
	    auto shift_struct = indexstruct<index_int,1>
	      (  new contiguous_indexstruct<index_int,1>(c.coord(id)+2) );
	    //print("coord-to-struct dim {} gives: {}\n",id,shift_struct.as_string());
	    m.set_component(id,shift_struct);
	  }
	  return m;
	} ) );
    REQUIRE_NOTHROW( op.set_is_shift_op() );
    CHECK( !op.is_single_operator() );
    CHECK( !op.is_point_operator() );
    CHECK( op.is_coord_struct_operator() );
  }
  
  INFO( "path: " << path );
  domain_coordinate *first,*last,*firster,*laster;
  shared_ptr<multi_indexstruct<index_int,1>> block,blocked;
  REQUIRE_NOTHROW( first = new domain_coordinate( vector<index_int>{1,2,3} ) );
  REQUIRE_NOTHROW( last = new domain_coordinate( vector<index_int>{10,20,30} ) );
  REQUIRE_NOTHROW( block = shared_ptr<multi_indexstruct<index_int,1>>( new contiguous_multi_indexstruct<index_int,1>(first,last) ) );
  // print("block multi has size {}\n",block.multi.size());
  INFO( format("input block: {}\n",block.as_string()) );
  REQUIRE_NOTHROW( blocked = op.operate(block) );
  // REQUIRE_NOTHROW( firster = blocked.first_index() );
  // REQUIRE_NOTHROW( laster = blocked.last_actual_index() );
  // for (int id=0; id<dim; id++) {
  //   CHECK( firster.coord(id)==first.coord(id)+2 );
  //   CHECK( laster.coord(id)==last.coord(id)+2 );
  // }
}

#endif
#endif
