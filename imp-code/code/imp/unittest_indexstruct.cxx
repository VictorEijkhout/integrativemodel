/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2022
 ****
 **** Unit tests for the IMP
 **** based on the CATCH framework (https://github.com/philsquared/Catch)
 ****
 **** unit tests for the indexstruct package
 **** (tests do not actually rely on MPI)
 ****
 ****************************************************************/

#include <stdlib.h>
// #include <string.h>
#include <stdlib.h>
#include <math.h>

#include "catch2/catch_all.hpp"

#include "indexstruct.hpp"
using fmt::memory_buffer;
using fmt::format;
using fmt::print;
using fmt::format_to;
using fmt::to_string;

using std::shared_ptr;
using std::string;
using std::vector;

TEST_CASE( "contiguous indexstruct","[indexstruct][1]" ) {

  shared_ptr<indexstruct<index_int,1>> i1,i2,i3,i4;

  SECTION( "basic construction" ) {
    // type testing
    SECTION( "contiguous" ) {
      i1 = shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(0,5) );
      CHECK( i1->is_contiguous() );
      CHECK( !i1->is_indexed() );
      CHECK( i1->first_index()==0 );
      CHECK( i1->last_index()==5 );
      CHECK( i1->local_size()==6 );
      CHECK( i1->stride()==1 );
      CHECK( i1->find(0)==0 );
      CHECK( i1->find(5)==5 );
      REQUIRE_THROWS( i1->find(6) );

      indexstructure<index_int,1> ii(i1);
      //REQUIRE_NOTHROW( ii = indexstructure(i1) );
      CHECK( ii.is_contiguous() );
      CHECK( !ii.is_indexed() );
      CHECK( ii.first_index()==0 );
      CHECK( ii.last_index()==5 );
      CHECK( ii.local_size()==6 );
      CHECK( ii.stride()==1 );
      CHECK( ii.find(0)==0 );
      CHECK( ii.find(5)==5 );
      REQUIRE_THROWS( ii.find(6) );
    }
    SECTION( "by accretion" ) {
      i1 = shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(2,5) );
      CHECK( i1->local_size()==4 );
      CHECK( i1->first_index()==2 );
      CHECK( i1->last_index()==5 );
      REQUIRE_NOTHROW( i1 = i1->add_element(3) );
      CHECK( i1->local_size()==4 );
      CHECK( i1->first_index()==2 );
      CHECK( i1->last_index()==5 );
      REQUIRE_NOTHROW( i1 = i1->add_element(5) );
      CHECK( i1->local_size()==4 );
      CHECK( i1->first_index()==2 );
      CHECK( i1->last_index()==5 );
      REQUIRE_NOTHROW( i1 = i1->add_element(6) );
      CHECK( i1->is_contiguous() );
      CHECK( i1->local_size()==5 );
      CHECK( i1->first_index()==2 );
      CHECK( i1->last_index()==6 );
      REQUIRE_NOTHROW( i1 = i1->add_element(1) );
      CHECK( i1->is_contiguous() );
      CHECK( i1->local_size()==6 );
      CHECK( i1->first_index()==1 );
      CHECK( i1->last_index()==6 );
      REQUIRE_NOTHROW( i1 = i1->add_element(9) );
      CHECK( !i1->is_strided() );
    }
    SECTION( "more contiguous" ) {
      i1 = shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(1,7) );
      CHECK( i1->is_contiguous() );
      CHECK( !i1->is_indexed() );
      CHECK( i1->first_index()==1 );
      CHECK( i1->last_index()==7 );
      CHECK( i1->stride()==1 );
      REQUIRE_THROWS( i1->find(0) );
    }
    SECTION( "strided" ) {
      i1 = shared_ptr<indexstruct<index_int,1>>( new strided_indexstruct<index_int,1>(2,6,2) );
      CHECK( i1->is_strided() );
      CHECK( !i1->is_contiguous() );
      CHECK( !i1->is_indexed() );
      CHECK( i1->first_index()==2 );
      CHECK( i1->last_index()==6 );
      CHECK( i1->local_size()==3 );
      CHECK( i1->stride()==2 );
      CHECK( i1->find(2)==0 );
      REQUIRE_THROWS( i1->find(3) );
      CHECK( i1->find(4)==1 );
      CHECK( i1->find(6)==2 );
      REQUIRE_THROWS( i1->find(7) );
      REQUIRE_THROWS( i1->find(8) );
    }
  }
  SECTION( "striding and operations" ) {
    i1 = shared_ptr<indexstruct<index_int,1>>{ new strided_indexstruct<index_int,1>(4,7,2) };

    SECTION( "basic stride tests" ) {
      CHECK( !i1->is_contiguous() );
      CHECK( i1->is_strided() );
      CHECK( !i1->is_indexed() );
      CHECK( i1->first_index()==4 );
      CHECK( i1->last_index()==6 );
      CHECK( i1->local_size()==2 );
      CHECK( i1->stride()==2 );

      i2 = shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(4,6) );
      CHECK( !i1->equals(i2) );
      CHECK( !i2->equals(i1) );
      CHECK( i2->get_ith_element(1)==5 );
      CHECK_THROWS( i2->get_ith_element(3) );

      i2 = shared_ptr<indexstruct<index_int,1>>( new strided_indexstruct<index_int,1>(4,6,2) );
      CHECK( i1->equals(i2) );
      CHECK( i2->equals(i1) );
      CHECK( i2->get_ith_element(1)==6 );
      CHECK_THROWS( i2->get_ith_element(3) );

      CHECK( i1->contains_element(4) );
      CHECK( !i1->contains_element(5) );
      CHECK( i1->contains_element(6) );
      CHECK( !i1->contains_element(7) );
    }

    SECTION( "strided containment" ) {
      i2 = shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(4,8) );
      i3 = shared_ptr<indexstruct<index_int,1>>( new strided_indexstruct<index_int,1>(4,8,2) );
      i4 = shared_ptr<indexstruct<index_int,1>>( new strided_indexstruct<index_int,1>(4,8,4) );
      CHECK( i2->contains(i3) );
      CHECK( !i3->contains(i2) );
      CHECK( !i2->equals(i3) );
      CHECK( i2->contains(i4) );
      CHECK( i3->contains(i4) );
      CHECK( !i4->contains(i2) );
      CHECK( !i4->contains(i3) );
    }

    SECTION( "translation forward" ) {
      REQUIRE_NOTHROW( i1 = i1->translate_by(1) );
      CHECK( i1->is_strided() );
      CHECK( !i1->is_contiguous() );
      CHECK( !i1->is_indexed() );
      CHECK( i1->first_index()==5 );
      CHECK( i1->last_index()==7 );
      CHECK( i1->local_size()==2 );
      CHECK( i1->stride()==2 );
    }

    SECTION( "translation backward" ) {
      REQUIRE_NOTHROW( i1 = i1->translate_by(-2) );
      CHECK( !i1->is_contiguous() );
      CHECK( i1->is_strided() );
      CHECK( !i1->is_indexed() );
      CHECK( i1->first_index()==2 );
      CHECK( i1->last_index()==4 );
      CHECK( i1->local_size()==2 );
      CHECK( i1->stride()==2 );
    }

    SECTION( "translation through zero" ) {
      REQUIRE_NOTHROW( i1 = i1->translate_by(-5) );
      CHECK( i1->is_strided() );
      CHECK( !i1->is_contiguous() );
      CHECK( !i1->is_indexed() );
      CHECK( i1->first_index()==-1 );
      CHECK( i1->last_index()==1 );
      CHECK( i1->local_size()==2 );
      CHECK( i1->stride()==2 );
    }
  }
  SECTION( "copy indexstruct" ) {
    
    i1 = shared_ptr<indexstruct<index_int,1>>{ new contiguous_indexstruct<index_int,1>(7,15) };
    REQUIRE_NOTHROW( i2 = i1->make_clone() );
    CHECK( i2->first_index()==7 );
    CHECK( i2->last_index()==15 );
    REQUIRE_NOTHROW( i1 = i1->translate_by(1) );
    CHECK( i1->first_index()==8 );
    CHECK( i1->last_index()==16 );
    CHECK( i2->first_index()==7 );
    CHECK( i2->last_index()==15 );  
  }
  SECTION( "find in contiguous" ) {
    i1 = shared_ptr<indexstruct<index_int,1>>{ new contiguous_indexstruct<index_int,1>(8,12) };
    i2 = shared_ptr<indexstruct<index_int,1>>{ new strided_indexstruct<index_int,1>(10,12,2) };
    index_int loc;
    REQUIRE_NOTHROW( loc = i1->location_of(i2) );
    CHECK( loc==2 );
    REQUIRE_THROWS( loc = i2->location_of(i1) );
  }
  SECTION( "find in strided" ) {
    i1 = shared_ptr<indexstruct<index_int,1>>{ new strided_indexstruct<index_int,1>(8,12,2) };
    i2 = shared_ptr<indexstruct<index_int,1>>{ new strided_indexstruct<index_int,1>(10,12,2) };
    i3 = shared_ptr<indexstruct<index_int,1>>{ new strided_indexstruct<index_int,1>(11,12,2) };
    index_int loc;
    REQUIRE_NOTHROW( loc = i1->location_of(i2) );
    CHECK( loc==1 );
    REQUIRE_THROWS( loc = i1->location_of(i3) );
    REQUIRE_THROWS( loc = i2->location_of(i1) );
  }
}

#if 0
TEST_CASE( "indexed indexstruct","[indexstruct][2]" ) {

  shared_ptr<indexstruct<index_int,1>> i1,i2,i3,i4;

  SECTION( "basic construction" ) {
    SECTION( "correct" ) {
      int len=3; index_int idx[3] = {1,2,4};
      i1 = shared_ptr<indexstruct<index_int,1>>( new indexed_indexstruct<index_int,1>(len,idx) );
      CHECK( !i1->is_contiguous() );
      CHECK( i1->is_indexed() );
      CHECK( i1->first_index()==1 );
      CHECK( i1->last_index()==4 );
      CHECK( i1->local_size()==len );
    }

    SECTION( "unsorted throws an error" ) {
      int len=4; index_int idx[4] = {1,2,6,4};
      CHECK_THROWS( i1 = shared_ptr<indexstruct<index_int,1>>( new indexed_indexstruct<index_int,1>(len,idx) ) );
    }

    SECTION( "negative indices allowed" ) {
      int len=3; index_int idx[3] = {-1,2,4};
      i1 = shared_ptr<indexstruct<index_int,1>>( new indexed_indexstruct<index_int,1>(len,idx) );
      CHECK( !i1->is_contiguous() );
      CHECK( i1->is_indexed() );
      CHECK( i1->first_index()==-1 );
      CHECK( i1->last_index()==4 );
      CHECK( i1->local_size()==len );
      indexstruct<index_int,1>ure ii(i1);
      //REQUIRE_NOTHROW( ii = indexstructure(i1) );
      CHECK( !ii.is_contiguous() );
      CHECK( ii.is_indexed() );
      CHECK( ii.first_index()==-1 );
      CHECK( ii.last_index()==4 );
      CHECK( ii.local_size()==len );
    }

    SECTION( "gradual construction" ) {
      int len=3; index_int idx[3] = {4,9,20};
      auto i1 = shared_ptr<indexstruct<index_int,1>>( new indexed_indexstruct<index_int,1>(len,idx) );
      CHECK( i1->local_size()==3 );
      CHECK( i1->first_index()==4 );
      CHECK( i1->last_index()==20 );
      REQUIRE_NOTHROW( i1 = i1->add_element(9) );
      CHECK( i1->local_size()==3 );
      CHECK( i1->first_index()==4 );
      CHECK( i1->last_index()==20 );
      REQUIRE_NOTHROW( i1 = i1->add_element(30) );
      CHECK( i1->local_size()==4 );
      CHECK( i1->first_index()==4 );
      CHECK( i1->last_index()==30 );
      REQUIRE_NOTHROW( i1 = i1->add_element(1) );
      CHECK( i1->local_size()==5 );
      CHECK( i1->first_index()==1 );
      CHECK( i1->last_index()==30 );
      REQUIRE_NOTHROW( i1 = i1->add_element(10) );
      CHECK( i1->local_size()==6 );
      CHECK( i1->first_index()==1 );
      CHECK( i1->last_index()==30 );
    }
    SECTION( "construction and simplify" ) {
      auto i1 =
	shared_ptr<indexstruct<index_int,1>>
	( new indexed_indexstruct<index_int,1>( vector<index_int>{2,6,8} ) ),
	i2 = 
	shared_ptr<indexstruct<index_int,1>>
	( new indexed_indexstruct<index_int,1>( vector<index_int>{3,5,7} ) );
      REQUIRE_NOTHROW( i4 = i1->force_simplify() );
      INFO( format("indexed {} simplified to {}",i1->as_string(),i4->as_string()) );
      CHECK( i4->is_indexed() );
      REQUIRE_NOTHROW( i1 = i1->add_element(4) );
      REQUIRE_NOTHROW( i4 = i1->force_simplify() );
      CHECK( i4->is_strided() );
      CHECK( i4->stride()==2 );
      REQUIRE_NOTHROW( i4 = i4->struct_union(i2) );
      REQUIRE_NOTHROW( i4 = i4->force_simplify() );
      CHECK( i4->is_strided() );
      CHECK( i4->is_contiguous() );
      // test boundary case of empty
      i1 = shared_ptr<indexstruct<index_int,1>>( new empty_indexstruct<index_int,1>() );
      REQUIRE_NOTHROW( i2 = i1->force_simplify() );
      i1 = shared_ptr<indexstruct<index_int,1>>( new empty_indexstruct<index_int,1>() );
      REQUIRE_NOTHROW( i1 = i1->struct_union
		       ( shared_ptr<indexstruct<index_int,1>>( new empty_indexstruct<index_int,1>() ) ) );
      REQUIRE_NOTHROW( i2 = i1->force_simplify() );
    }
  }

  SECTION( "more simplify" ) {
    auto i1 =
      shared_ptr<indexstruct<index_int,1>>
      ( new indexed_indexstruct<index_int,1>( vector<index_int>{2,4, 10,12, 15} ) );
    INFO( format("starting with indexed {}",i1->as_string()) );
    REQUIRE_NOTHROW( i1->add_element(11) );
    INFO( format("add element 11 gives {}",i1->as_string()) );
    CHECK( i1->is_indexed() );
    REQUIRE_NOTHROW( i2 = i1->force_simplify() );
    INFO( format("simplify to composite {}",i2->as_string()) );
    CHECK( i2->is_composite() );
    REQUIRE_NOTHROW( i3 = i2->convert_to_indexed() );
    INFO( format("back to indexed {}",i3->as_string()) );
    CHECK( i1->equals(i3) );
  }

  SECTION( "striding and operations" ) {
    int len=5; index_int idx[5] = {1,2,4,7,9};
    shared_ptr<indexstruct<index_int,1>> i1;
    REQUIRE_NOTHROW( i1 = shared_ptr<indexstruct<index_int,1>>{ new indexed_indexstruct<index_int,1>(len,idx) } );

    SECTION( "basic stride tests" ) {
      CHECK( !i1->is_contiguous() );
      CHECK( i1->is_indexed() );
      CHECK( i1->first_index()==1 );
      CHECK( i1->last_index()==9 );
      CHECK( i1->local_size()==len );

      CHECK( !i1->contains_element(0) );
      CHECK( i1->contains_element(1) );
      CHECK( i1->contains_element(4) );
      CHECK( !i1->contains_element(5) );
      CHECK( !i1->contains_element(6) );
      CHECK( i1->contains_element(7) );

      CHECK( i1->find(1)==0 );
      CHECK( i1->find(7)==3 );
      REQUIRE_THROWS( i1->find(0) );
      REQUIRE_THROWS( i1->find(8) );
      CHECK_THROWS( i1->get_ith_element(5) );
      CHECK_NOTHROW( i1->get_ith_element(4) );
      CHECK( i1->get_ith_element(3)==7 );
    }

    SECTION( "translation forward" ) {
      REQUIRE_NOTHROW( i1 = i1->translate_by(1) );
      CHECK( !i1->is_contiguous() );
      CHECK( i1->is_indexed() );
      CHECK( i1->first_index()==2 );
      CHECK( i1->last_index()==10 );
      CHECK( i1->local_size()==len );
    }

    SECTION( "translation through zero" ) {
      REQUIRE_NOTHROW( i1 = i1->translate_by(-2) );
      CHECK( !i1->is_contiguous() );
      CHECK( i1->is_indexed() );
      CHECK( i1->first_index()==-1 );
      CHECK( i1->last_index()==7 );
      CHECK( i1->local_size()==len );
    }
  }
  SECTION( "find in indexed" ) {
    int len=5; index_int idx[5] = {1,2,4,7,9};
    REQUIRE_NOTHROW( i1 = shared_ptr<indexstruct<index_int,1>>( new indexed_indexstruct<index_int,1>(len,idx) ) );

    auto
      i2 = shared_ptr<indexstruct<index_int,1>>( new strided_indexstruct<index_int,1>(2,4,2) ),
      i3 = shared_ptr<indexstruct<index_int,1>>( new strided_indexstruct<index_int,1>(7,8,2) );
    index_int loc;
    REQUIRE_NOTHROW( loc = i1->location_of(i2) );
    CHECK( loc==1 );
    REQUIRE_NOTHROW( loc = i1->location_of(i3) );
    CHECK( loc==3 );
    indexstruct<index_int,1>ure ii(i1);
    //REQUIRE_NOTHROW( ii = indexstructure(i1) );
    REQUIRE_NOTHROW( loc = ii.location_of(i2) );
    CHECK( loc==1 );
    indexstructure ii2(i2);
    REQUIRE_NOTHROW( loc = ii.location_of(ii2) );
    CHECK( loc==1 );
    REQUIRE_NOTHROW( loc = ii.location_of(indexstructure(i3)) );
    CHECK( loc==3 );
  }
}

TEST_CASE( "composite indexstruct","[indexstruct][composite][8]" ) {
  shared_ptr<indexstruct<index_int,1>> i1,i2,ifinal;
  shared_ptr<composite_indexstruct<index_int,1>> icomp;

  i1 = shared_ptr<indexstruct<index_int,1>>{ new contiguous_indexstruct<index_int,1>(3,5) };
  i2 = shared_ptr<indexstruct<index_int,1>>{ new contiguous_indexstruct<index_int,1>(10,12) };
  SECTION( "catch old design: obviated" ) {
    REQUIRE_NOTHROW( icomp = shared_ptr<composite_indexstruct<index_int,1>>{ new composite_indexstruct<index_int,1>() } );
    REQUIRE_THROWS( ifinal = icomp->struct_union(i1) );
  }
  SECTION( "improved design" ) {
    SECTION( "right away" ) {
      REQUIRE_NOTHROW( icomp = shared_ptr<composite_indexstruct<index_int,1>>{ new composite_indexstruct<index_int,1>() } );
      REQUIRE_NOTHROW( icomp->push_back(i1) );
      REQUIRE_NOTHROW( icomp->push_back(i2) );
    }
    SECTION( "reverse away" ) {
      REQUIRE_NOTHROW( icomp = shared_ptr<composite_indexstruct<index_int,1>>{ new composite_indexstruct<index_int,1>() } );
      REQUIRE_NOTHROW( icomp->push_back(i2) );
      REQUIRE_NOTHROW( icomp->push_back(i1) );
    }
    REQUIRE_NOTHROW( ifinal = icomp->make_clone() );
    CHECK( !ifinal->is_contiguous() );
    CHECK( ifinal->first_index()==3 );
    CHECK( ifinal->last_index()==12 );
    CHECK( ifinal->local_size()==6 );
  }
  SECTION( "tricky composite simplify with indexed" ) {
    auto i1 = shared_ptr<composite_indexstruct<index_int,1>>( new composite_indexstruct<index_int,1>() );
    auto left_cont = shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(0,9) );
    REQUIRE_NOTHROW( i1->push_back(left_cont) );
    auto right_cont =shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(11,19) );
    REQUIRE_NOTHROW( i1->push_back(right_cont) );
    auto more_cont =shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(23,30) );
    REQUIRE_NOTHROW( i1->push_back(more_cont) );
    auto gaps = shared_ptr<indexstruct<index_int,1>>
      ( new indexed_indexstruct<index_int,1>( vector<index_int>{10,20,22,40} ) );
    REQUIRE_NOTHROW( i1->push_back(gaps) );
    CHECK( i1->get_structs().size()==4 );
    REQUIRE_NOTHROW( i2 = i1->force_simplify() );
    INFO( format("Simplifying {} to {}",i1->as_string(),i2->as_string()) );
    CHECK( i2->is_composite() );
    auto i2comp = dynamic_cast<composite_indexstruct<index_int,1>*>(i2.get());
    if (i2comp==nullptr) CHECK( 0 );
    CHECK( i2comp->get_structs().size()==3 );
  }

  SECTION( "tricky composite simplify with strided" ) {
    auto i1 = shared_ptr<composite_indexstruct<index_int,1>>( new composite_indexstruct<index_int,1>() );
    auto left_cont = shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(0,9) );
    REQUIRE_NOTHROW( i1->push_back(left_cont) );
    auto right_cont =shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(11,19) );
    REQUIRE_NOTHROW( i1->push_back(right_cont) );
    auto more_cont =shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(23,29) );
    REQUIRE_NOTHROW( i1->push_back(more_cont) );
    SECTION( "fully incorporate" ) {
      auto gaps = shared_ptr<indexstruct<index_int,1>>
	( new strided_indexstruct<index_int,1>( 10,30,10 ) );
      REQUIRE_NOTHROW( i1->push_back(gaps) );
      CHECK( i1->get_structs().size()==4 );
      REQUIRE_NOTHROW( i2 = i1->force_simplify() );
      INFO( format("Simplifying {} to {}",i1->as_string(),i2->as_string()) );
      CHECK( i2->is_composite() );
      auto i2comp = dynamic_cast<composite_indexstruct<index_int,1>*>(i2.get());
      if (i2comp==nullptr) CHECK( 0 );
      CHECK( i2comp->get_structs().size()==2 );
    }
    SECTION( "shift left" ) {
      auto gaps = shared_ptr<indexstruct<index_int,1>>
	( new strided_indexstruct<index_int,1>( 10,50,10 ) );
      REQUIRE_NOTHROW( i1->push_back(gaps) );
      CHECK( i1->get_structs().size()==4 );
      REQUIRE_NOTHROW( i2 = i1->force_simplify() );
      INFO( format("Simplifying {} to {}",i1->as_string(),i2->as_string()) );
      CHECK( i2->is_composite() );
      auto i2comp = dynamic_cast<composite_indexstruct<index_int,1>*>(i2.get());
      if (i2comp==nullptr) CHECK( 0 );
      CHECK( i2comp->get_structs().size()==3 );
    }
  }
  SECTION( "composite over simplify" ) {
    auto i1 = shared_ptr<composite_indexstruct<index_int,1>>( new composite_indexstruct<index_int,1>() );
    bool has_index{false};
    auto right_cont =shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(11,19) );
    REQUIRE_NOTHROW( i1->push_back(right_cont) );
    auto left_cont = shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(2,9) );
    REQUIRE_NOTHROW( i1->push_back(left_cont) );
    SECTION( "just two members" ) {
    }
    SECTION( "three can also be incorporated" ) {
      SECTION( "contiguous" ) {
	auto more_cont =shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(21,29) );
	REQUIRE_NOTHROW( i1->push_back(more_cont) );
      }
      SECTION( "indexed" ) {
	auto more_cont =shared_ptr<indexstruct<index_int,1>>( new indexed_indexstruct<index_int,1>( vector<index_int>{21} ) );
	REQUIRE_NOTHROW( i1->push_back(more_cont) );
      }
    }
    SECTION( "index" ) {
      has_index = true;
      auto more =shared_ptr<indexstruct<index_int,1>>( new indexed_indexstruct<index_int,1>( vector<index_int>{0,40,100} ) );
      REQUIRE_NOTHROW( i1->push_back(more) );
    }
    shared_ptr<indexstruct<index_int,1>> i2;
    REQUIRE_NOTHROW( i2 = i1->over_simplify() );
    INFO( format("{} --simplify--> {}",i1->as_string(),i2->as_string()) );
    if (has_index) {
      CHECK( i2->is_composite() );
    } else {
      CHECK( i2->is_contiguous() );
    }
  }
}

TEST_CASE( "enumerating indexstruct<index_int,1>s","[10]" ) {
  int count,cnt=0;
  
  SECTION( "contiguous" ) {
    shared_ptr<indexstruct<index_int,1>> idx;
    REQUIRE_NOTHROW( idx = shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(13,15) ) );
    int value = 13, count = 0;
    SECTION( "traditional" ) {
      REQUIRE_NOTHROW( idx->begin() );
      REQUIRE_NOTHROW( idx->end() );
      try {
	for (auto i=idx->begin(); i!=idx->end(); ++i) {
	  CHECK( *i==value );
	  CHECK( idx->get_ith_element(count)==(*i) );
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
	  CHECK( idx->get_ith_element(count)==i );
	  value++; count++;
	}
      } catch( string c ) {
	print("Contiguous ranging loop failed: {}\n",c);
      }
    }
    CHECK( count==3 );
  }
  SECTION( "strided" ) {
    shared_ptr<indexstruct<index_int,1>> idx;
    REQUIRE_NOTHROW( idx = shared_ptr<indexstruct<index_int,1>>( new strided_indexstruct<index_int,1>(3,10,2) ) );
    CHECK( idx->first_index()==3 );
    CHECK( idx->last_index()==9 );
    int value = 3, count = 0;
    SECTION( "traditional" ) {
      for (auto i=idx->begin(); i!=idx->end(); ++i) {
	CHECK( *i==value );
	CHECK( idx->get_ith_element(count)==(*i) );
	value += 2; count++;
      }
    }
    SECTION( "ranged" ) {
      for (auto i : *idx) {
	CHECK( i==value );
	CHECK( idx->get_ith_element(count)==i );
	value += 2; count++;
      }
    }
  }
  SECTION( "indexed" ) {
    index_int *ar = new index_int[4]{2,3,5,8};
    shared_ptr<indexstruct<index_int,1>> idx;
    REQUIRE_NOTHROW( idx = shared_ptr<indexstruct<index_int,1>>( new indexed_indexstruct<index_int,1>(4,ar) ) );
    count = 0;
    SECTION( "traditional" ) {
      for (auto i=idx->begin(); i!=idx->end(); ++i) {
	CHECK( *i==ar[count] );
	CHECK( idx->get_ith_element(cnt++)==(*i) );
	count++;
      }
    }
    SECTION( "ranged" ) {
      for (auto i : *idx) {
	CHECK( i==ar[count] );
	CHECK( idx->get_ith_element(cnt++)==i );
	count++;
      }
    }
    CHECK( count==4 );
  }
  SECTION( "composite" ) {
    auto
      i1 = shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(1,10) ),
      i2 = shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(21,30) );
    shared_ptr<indexstruct<index_int,1>> icomp;
    REQUIRE_NOTHROW( icomp = shared_ptr<indexstruct<index_int,1>>{ i1->make_clone() } );
    REQUIRE_NOTHROW( icomp = icomp->struct_union(i2) );
    CHECK( icomp->is_composite() );
    CHECK( icomp->local_size()==20 );
    cnt = 0; const char *path;
    SECTION( "traditional" ) { path = "traditional";
      for ( indexstruct<index_int,1> i=icomp->begin(); i!=icomp->end(); ++i ) {
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

TEST_CASE( "indexstruct<index_int,1> intersections","[indexstruct<index_int,1>][intersect][20]" ) {
  
  shared_ptr<indexstruct<index_int,1>> i1,i2,i3,i4;
  SECTION( "first cont" ) {
    i1 = shared_ptr<indexstruct<index_int,1>>{ new contiguous_indexstruct<index_int,1>(1,10) };
    indexstruct<index_int,1>ure I1( contiguous_indexstruct<index_int,1>(1,10) );
    SECTION( "cont-cont" ) {
      i2 = shared_ptr<indexstruct<index_int,1>>{ new contiguous_indexstruct<index_int,1>(5,12) };
      REQUIRE_NOTHROW( i3 = i1->intersect(i2) );
      REQUIRE( i3!=nullptr );
      CHECK( i3->is_contiguous() );
      CHECK( i3->first_index()==5 );
      CHECK( i3->last_index()==10 );
      CHECK( !i1->contains(i2) );
      CHECK( i1->contains(i3) );
      CHECK( i2->contains(i3) );

      indexstruct<index_int,1>ure I2( contiguous_indexstruct<index_int,1>(5,12) ), I3;
      REQUIRE_NOTHROW( I3 = I1.intersect(I2) );
      CHECK( I3.is_contiguous() );
      CHECK( I3.first_index()==5 );
      CHECK( I3.last_index()==10 );

      i2 = shared_ptr<indexstruct<index_int,1>>{ new contiguous_indexstruct<index_int,1>(10,12) };
      REQUIRE_NOTHROW( i3 = i1->intersect(i2) );
      REQUIRE( i3!=nullptr );
      CHECK( i3->is_contiguous() );
      CHECK( i3->first_index()==10 );
      CHECK( i3->last_index()==10 );
      CHECK( !i1->contains(i2) );
      CHECK( i1->contains(i3) );
      CHECK( i2->contains(i3) );
      REQUIRE_THROWS( i4 = i2->relativize_to(i1) );
      REQUIRE_NOTHROW( i4 = i3->relativize_to(i1) ); // [10,10] in [1,10] is [9,9]
      CHECK( i4->is_contiguous() );
      CHECK( i4->first_index()==9 );
      CHECK( i4->last_index()==9 );
      REQUIRE_NOTHROW( i4 = i3->relativize_to(i2) );
      CHECK( i4->is_contiguous() );
      CHECK( i4->first_index()==0 );
      CHECK( i4->last_index()==0 );

      i2 = shared_ptr<indexstruct<index_int,1>>{ new contiguous_indexstruct<index_int,1>(11,12) };
      REQUIRE_NOTHROW( i3 = i1->intersect(i2) );
      REQUIRE( i3->is_empty() );

      i2 = shared_ptr<indexstruct<index_int,1>>{ new strided_indexstruct<index_int,1>(10,12,2) };
      i3 = shared_ptr<indexstruct<index_int,1>>( new strided_indexstruct<index_int,1>(8,14,2) );
      REQUIRE_NOTHROW( i4 = i2->relativize_to(i3) );
      CHECK( i4->stride()==1 );
      CHECK( i4->local_size()==2 );
      CHECK( i4->first_index()==1 );
      CHECK( i4->last_index()==2 );
    }
    SECTION( "cont-idx" ) {
      int len=3; index_int idx[3] = {4,8,11};
      i2 = shared_ptr<indexstruct<index_int,1>>{ new indexed_indexstruct<index_int,1>(len,idx) };
      REQUIRE_NOTHROW( i3 = i1->intersect(i2) );
      REQUIRE( !i3->is_empty() );
      CHECK( i3->is_indexed() );
      CHECK( i3->first_index()==4 );
      CHECK( i3->last_index()==8 );
      CHECK( !i1->contains(i2) );
      CHECK( i1->contains(i3) );
      CHECK( i2->contains(i3) );
      REQUIRE_THROWS( i2->relativize_to(i1) );

      len=3; index_int idxs[3] = {4,8,10}; 
      i2 = shared_ptr<indexstruct<index_int,1>>{ new indexed_indexstruct<index_int,1>(len,idxs) };
      REQUIRE_NOTHROW( i3 = i1->intersect(i2) ); // [1,10] & [4,8,10] => i3 = [4,8,10]
      REQUIRE( !i3->is_empty() );
      CHECK( i3->is_indexed() );
      CHECK( i3->first_index()==4 );
      CHECK( i3->last_index()==10 );
      CHECK( i1->contains(i2) );
      CHECK( i1->contains(i3) );
      CHECK( i2->contains(i3) );

      CHECK( i3->is_indexed() );

      REQUIRE_NOTHROW( i4 = i3->relativize_to(i1) ); // [4,8,10] in [1:10] i3 is indexed

      CHECK( i4->is_indexed() );
      CHECK( i4->first_index()==3 );
      CHECK( i4->last_index()==9 );
    }
  }
  SECTION( "stride-stride" ) {
    i1 = shared_ptr<indexstruct<index_int,1>>{ new strided_indexstruct<index_int,1>(10,20,2) };
    CHECK( i1->local_size()==6 );
    i2 = shared_ptr<indexstruct<index_int,1>>{ new strided_indexstruct<index_int,1>(12,14,2) };
    REQUIRE_NOTHROW( i3 = i1->intersect(i2) );
    CHECK( i3->local_size()==2 );
    CHECK( i3->is_strided() );
    CHECK( i3->stride()==2 );

    i2 = shared_ptr<indexstruct<index_int,1>>{ new strided_indexstruct<index_int,1>(12,22,10) };
    REQUIRE_NOTHROW( i3 = i1->intersect(i2) );
    CHECK( i3->local_size()==1 );

    i2 = shared_ptr<indexstruct<index_int,1>>{ new strided_indexstruct<index_int,1>(12,20,4) };
    REQUIRE_NOTHROW( i3 = i1->intersect(i2) );
    CHECK( i3->local_size()==3 );
    CHECK( i3->is_strided() );

    i2 = shared_ptr<indexstruct<index_int,1>>{ new strided_indexstruct<index_int,1>(12,20,5) };
    REQUIRE_NOTHROW( i3 = i1->intersect(i2) );
    CHECK( i3->local_size()==1 );

    i2 = shared_ptr<indexstruct<index_int,1>>{ new strided_indexstruct<index_int,1>(13,20,4) };
    REQUIRE_NOTHROW( i3 = i1->intersect(i2) );
    CHECK( i3!=nullptr );
    CHECK( i3->is_empty() );
  }
  SECTION( "strided-indexed" ) {
    i1 = shared_ptr<indexstruct<index_int,1>>{ new strided_indexstruct<index_int,1>(10,20,10) };
    CHECK( i1->local_size()==2 );
    i2 = shared_ptr<indexstruct<index_int,1>>{
      new indexed_indexstruct<index_int,1>( vector<index_int>{10,16,20} ) };
    CHECK( i2->local_size()==3 );
    REQUIRE_NOTHROW( i3 = i1->relativize_to(i2) );
    CHECK( i3->local_size()==2 );
    CHECK( i3->first_index()==0 );
    CHECK( i3->last_index()==2 );
  }
  SECTION( "idx-idx" ) {
    shared_ptr<indexstruct<index_int,1>> i5{nullptr};
    i2 = shared_ptr<indexstruct<index_int,1>>{
      new indexed_indexstruct<index_int,1>( vector<index_int>{4,8,11} ) };
    i3 = shared_ptr<indexstruct<index_int,1>>{
      new indexed_indexstruct<index_int,1>( vector<index_int>{3,8,10,11,12} ) };
    REQUIRE_NOTHROW( i4 = i2->intersect(i3) );
    REQUIRE( i4!=nullptr );
    CHECK( i4->local_size()==2 );
    CHECK( i4->is_indexed() );
    CHECK( i4->first_index()==8 );
    CHECK( i4->last_index()==11 );
    CHECK( !i2->contains(i3) );
    CHECK( i2->contains(i4) );
    CHECK( i3->contains(i4) );

    indexstruct<index_int,1>ure I2(i2), I3(i3), I4;
    REQUIRE_NOTHROW( I4 = I2.intersect(I3) );
    CHECK( I4.local_size()==2 );
    CHECK( I4.is_indexed() );
    CHECK( I4.first_index()==8 );
    CHECK( I4.last_index()==11 );

    REQUIRE_THROWS( i5 = i3->relativize_to(i2) );
    REQUIRE_THROWS( i5 = i2->relativize_to(i3) );
    CHECK( i2->contains(i4) );
    CHECK( i2->is_indexed() );
    CHECK( i4->is_indexed() );
    REQUIRE_NOTHROW( i5 = i4->relativize_to(i2) ); // [8,11] in indexed:[4,8,11]
    CHECK( i5->is_indexed() );
    CHECK( i5->first_index()==1 );
    CHECK( i5->last_index()==2 );
    REQUIRE_NOTHROW( i5 = i4->relativize_to(i3) ); // [8,11] in [3,8,10,11,12]
    CHECK( i5->is_indexed() );
    CHECK( i5->first_index()==1 );
    CHECK( i5->last_index()==3 );
  }
}

TEST_CASE( "indexstruct<index_int,1> differences","[indexstruct<index_int,1>][minus][21]" ) {
  indexstruct<index_int,1> *it;
  shared_ptr<indexstruct<index_int,1>> i1,i2,i3;
  SECTION( "cont-cont non-overlapping" ) {
    i1 = shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(5,15) );
    i2 = shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(20,30) );
    SECTION( "one way" ) {
      REQUIRE_NOTHROW( i3 = i1->minus(i2) );
      INFO( i3->as_string() );
      CHECK( i3->is_contiguous() );
      CHECK( i3->equals(i1) );
    }
    SECTION( "other way" ) {
      REQUIRE_NOTHROW( i3 = i2->minus(i1) ); 
      CHECK( i3->is_contiguous() );
      CHECK( i3->equals(i2) );
    }
  }
  SECTION( "cont-cont containment" ) {
    i1 = shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(5,30) );
    i2 = shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(15,20) );
    SECTION( "one way" ) {
      REQUIRE_NOTHROW( i3 = i1->minus(i2) );
      CHECK( i3->local_size()==20 );
      CHECK( i3->first_index()==5 );
      CHECK( i3->last_index()==30 );
    }
    SECTION( "other way" ) {
      REQUIRE_NOTHROW( i3 = i2->minus(i1) );
      CHECK( i3->is_empty() );
    }
  }
  SECTION( "cont-cont for real" ) {
    i1 = shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(5,20) );
    i2 = shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(15,30) );
    SECTION( "one way" ) {
      REQUIRE_NOTHROW( i3 = i1->minus(i2) );
      INFO( i3->as_string() );
      CHECK( i3->is_contiguous() );
      CHECK( i3->first_index()==5 );
      CHECK( i3->last_index()==14 );
    }
    SECTION( "other way" ) {
      REQUIRE_NOTHROW( i3 = i2->minus(i1) );
      CHECK( i3->is_contiguous() );
      CHECK( i3->first_index()==21 );
      CHECK( i3->last_index()==30 );
    }
  }
  SECTION( "contiguous minus contiguous, creating gap" ) {
    i1 = shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(1,40) );
    i2 = shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(11,20) );
    REQUIRE_NOTHROW( i3 = i1->minus(i2) );
    INFO( "resulting i3: " << i3->as_string() );
    CHECK( i3->local_size()==30 );
    CHECK( !i3->contains_element(11) );
    CHECK( i3->is_composite() );
    composite_indexstruct<index_int,1> *i4;
    REQUIRE_NOTHROW( i4 = dynamic_cast<composite_indexstruct<index_int,1>*>(i3.get()) );
    CHECK( i4!=nullptr );
    CHECK( i4->get_structs().size()==2 );
  }
  SECTION( "strided minus contiguous, creating gap" ) {
    i1 = shared_ptr<indexstruct<index_int,1>>( new strided_indexstruct<index_int,1>(1,41,4) );
    CHECK( i1->local_size()==11 );
    i2 = shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(12,16) ); // this only cuts 1
    REQUIRE_NOTHROW( i3 = i1->minus(i2) );
    INFO( "resulting i3: " << i3->as_string() );
    CHECK( i3->is_composite() );
    CHECK( !i3->contains_element(13) );
    CHECK( i3->local_size()==10 );
    composite_indexstruct<index_int,1> *i4;
    REQUIRE_NOTHROW( i4 = dynamic_cast<composite_indexstruct<index_int,1>*>(i3.get()) );
    CHECK( i4!=nullptr );
    CHECK( i4->get_structs().size()==2 );
  }
  SECTION( "strided minus contiguous, hitting nothing" ) {
    i1 = shared_ptr<indexstruct<index_int,1>>( new strided_indexstruct<index_int,1>(1,41,4) );
    CHECK( i1->local_size()==11 );
    i2 = shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(14,16)); // this hits nothing: falls between 13-17
    REQUIRE_NOTHROW( i3 = i1->minus(i2) );
    INFO( "resulting i3: " << i3->as_string() );
    CHECK( i3->local_size()==11 );
    CHECK( i3->is_strided() );
  }
  SECTION( "indexed cont" ) {
    i1 = shared_ptr<indexstruct<index_int,1>>( new strided_indexstruct<index_int,1>( 4,16,4 ) );
    i1 = i1->convert_to_indexed() ;
    CHECK( i1->is_indexed() );
    CHECK( i1->first_index()==4 );
    CHECK( i1->last_index()==16 );
    i2  = shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>( 13,17 ) );
    REQUIRE_NOTHROW( i3 = i1->minus(i2) );
    CHECK( i3->is_indexed() );
    CHECK( i3->first_index()==4 );
    CHECK( i3->last_index()==12 );

    indexstruct<index_int,1>ure I1(strided_indexstruct<index_int,1>( 4,16,4 )), I2(i2), I3;
    REQUIRE_NOTHROW( I3 = I1.minus(I2) );
    CHECK( I3.local_size()==3 );
    CHECK( I3.first_index()==4 );
    CHECK( I3.last_index()==12 );

  }
  SECTION( "indexed-indexed" ) {
    i1 = shared_ptr<indexstruct<index_int,1>>( new strided_indexstruct<index_int,1>( 4,16,4 ) );
    i1 = shared_ptr<indexstruct<index_int,1>>( i1->convert_to_indexed() );
    i2 = shared_ptr<indexstruct<index_int,1>>( new strided_indexstruct<index_int,1>( 5,17,4 ) );
    i2 = shared_ptr<indexstruct<index_int,1>>( i2->convert_to_indexed() );
    REQUIRE_NOTHROW( i3 = i1->minus(i2) );
    CHECK( i3->equals(i1) );
    i2 = shared_ptr<indexstruct<index_int,1>>( new strided_indexstruct<index_int,1>( 16,20,2 ) );
    i2 = shared_ptr<indexstruct<index_int,1>>( i2->convert_to_indexed() );
    REQUIRE_NOTHROW( i3 = i1->minus(i2) );
    CHECK( i3->local_size()==3 );
    CHECK( i3->first_index()==4 );
    CHECK( i3->last_index()==12 );
  }
}

TEST_CASE( "indexstruct<index_int,1> unions","[indexstruct<index_int,1>][union][22]" ) {
  shared_ptr<indexstruct<index_int,1>> i1,i2,i3;

  SECTION( "convert from stride 1" ) {
    i1 = shared_ptr<indexstruct<index_int,1>>{ new contiguous_indexstruct<index_int,1>(2,2+SMALLBLOCKSIZE-1) };
    CHECK( i1->is_contiguous() );
    CHECK_NOTHROW( i2 = i1->convert_to_indexed() );
    CHECK( i2->is_indexed() );
    CHECK( i2->first_index()==2 );
    CHECK( i2->last_index()==2+SMALLBLOCKSIZE-1 );
    CHECK( i2->local_size()==i1->local_size() );
  }
  SECTION( "convert from stride 2" ) {
    i1 = shared_ptr<indexstruct<index_int,1>>{ new strided_indexstruct<index_int,1>(2,2+2*SMALLBLOCKSIZE-2,2) };
    CHECK_NOTHROW( i2 = i1->convert_to_indexed() );
    CHECK( i2->is_indexed() );
    CHECK( i2->first_index()==2 );
    CHECK( i2->last_index()==2+2*SMALLBLOCKSIZE-2 );
    CHECK( i2->local_size()==i1->local_size() );
  }
  SECTION( "cont-cont" ) {
    i1 = shared_ptr<indexstruct<index_int,1>>{ new contiguous_indexstruct<index_int,1>(1,10) };
    indexstruct<index_int,1>ure I1(contiguous_indexstruct<index_int,1>(1,10));
    SECTION( "1" ) {
      i2 = shared_ptr<indexstruct<index_int,1>>{ new contiguous_indexstruct<index_int,1>(5,12) };
      indexstruct<index_int,1>ure I2(contiguous_indexstruct<index_int,1>(5,12)), I3;
      REQUIRE_NOTHROW( i3 = i1->struct_union(i2) );
      REQUIRE_NOTHROW( I3 = I1.struct_union(I2) );
      REQUIRE( i3!=nullptr );
      CHECK( i3->is_contiguous() );
      CHECK( i3->first_index()==1 );
      CHECK( i3->last_index()==12 );
    }
    SECTION( "2" ) {
      i2 = shared_ptr<indexstruct<index_int,1>>{ new contiguous_indexstruct<index_int,1>(11,13) };
      REQUIRE_NOTHROW( i3 = i1->struct_union(i2) );
      REQUIRE( i3!=nullptr );
      CHECK( i3->is_contiguous() );
      CHECK( i3->first_index()==1 );
      CHECK( i3->last_index()==13 );
    }
    SECTION( "3" ) {
      SECTION( "extend right" ) {
	i1 = shared_ptr<indexstruct<index_int,1>>{ new contiguous_indexstruct<index_int,1>(1,10) };
	i2 = shared_ptr<indexstruct<index_int,1>>{ new contiguous_indexstruct<index_int,1>(13) };
	REQUIRE_NOTHROW( i3 = i1->struct_union(i2) ); // [1--10]+[13]

	REQUIRE( i3!=nullptr );
	CHECK( !i3->is_indexed() );
	CHECK( i3->is_composite() );
	index_int i1l,i2l;
	REQUIRE_NOTHROW( i1l = i1->local_size() );
	REQUIRE_NOTHROW( i2l = i2->local_size() );
	CHECK( i3->local_size()==(i1l+i2l) );
	CHECK( i3->first_index()==1 );
	CHECK( i3->last_index()==13 );
      }
      SECTION( "extend left" ) {
	i1 = shared_ptr<indexstruct<index_int,1>>{ new contiguous_indexstruct<index_int,1>(1,10) };
	i2 = shared_ptr<indexstruct<index_int,1>>{ new contiguous_indexstruct<index_int,1>(14) };
	REQUIRE_NOTHROW( i3 = i2->struct_union(i1) ); // [1--10]+[14]
	//print("i3 {}\n",i3->as_string());
	REQUIRE( i3!=nullptr );
	CHECK( !i3->is_indexed() );
	CHECK( i3->is_composite() );
	CHECK( i3->local_size()==(i1->local_size()+i2->local_size()) );
	CHECK( i3->first_index()==1 );
	CHECK( i3->last_index()==14 );
      }
    }
  }
  SECTION( "cont-idx giving indexed" ) {
    i1 = shared_ptr<indexstruct<index_int,1>>{ new contiguous_indexstruct<index_int,1>(5,8) };
    i2 = shared_ptr<indexstruct<index_int,1>>( new strided_indexstruct<index_int,1>(8,12,2) );
    CHECK( i2->local_size()==3 );
    REQUIRE_NOTHROW( i2 = i2->convert_to_indexed() );
    CHECK( i2->local_size()==3 );
    REQUIRE_NOTHROW( i3 = i1->struct_union(i2.get()) ); // [5-8] & [8,10,12] overlap 1
    INFO( "i3 should be [5-8] & [8,10,12]: " << i3->as_string() );
    CHECK( !i3->is_contiguous() );
    CHECK( i1->local_size()==4 );
    CHECK( i2->local_size()==3 );
    CHECK( i3->local_size()==6 );
    CHECK( i3->first_index()==5 );
    CHECK( i3->last_index()==12);
  }
  SECTION( "cont-idx extending" ) {
    i1 = shared_ptr<indexstruct<index_int,1>>{ new contiguous_indexstruct<index_int,1>(5,11) };
    i2 = shared_ptr<indexstruct<index_int,1>>( new strided_indexstruct<index_int,1>(8,12,2) );
    indexstruct<index_int,1>ure I3,
      I1(contiguous_indexstruct<index_int,1>(5,11)),
      I2(strided_indexstruct<index_int,1>(8,12,2));
    CHECK( i2->local_size()==3 );
    REQUIRE_NOTHROW( i2 = i2->convert_to_indexed() );
    CHECK( i2->is_indexed() );
    CHECK( i2->local_size()==3 );
    REQUIRE_NOTHROW( i3 = i1->struct_union(i2.get()) );
    REQUIRE_NOTHROW( I3 = I1.struct_union(I2) );
    CHECK( i1->local_size()==7 );
    CHECK( i3->is_contiguous() );
    CHECK( i3->local_size()==8 );
    CHECK( i3->first_index()==5 );
    CHECK( i3->last_index()==12);
    // VLE do we need this? CHECK( I3.is_contiguous() );
    CHECK( I3.local_size()==8 );
    CHECK( I3.first_index()==5 );
    CHECK( I3.last_index()==12);
  }
  SECTION( "cont-idx extending2" ) {
    i1 = shared_ptr<indexstruct<index_int,1>>{ new contiguous_indexstruct<index_int,1>(5,11) };
    i2 = shared_ptr<indexstruct<index_int,1>>( new strided_indexstruct<index_int,1>(6,12,2) );
    CHECK( i2->local_size()==4 );
    REQUIRE_NOTHROW( i2 = i2->convert_to_indexed() );
    CHECK( i2->is_indexed() );
    CHECK( i2->local_size()==4 );
    REQUIRE_NOTHROW( i3 = i1->struct_union(i2.get()) );
    CHECK( i1->local_size()==7 );
    CHECK( i3->is_contiguous() );
    CHECK( i3->local_size()==8 );
    CHECK( i3->first_index()==5 );
    CHECK( i3->last_index()==12);
  }
  SECTION( "idx-cont" ) {
    i1 = shared_ptr<indexstruct<index_int,1>>{ new strided_indexstruct<index_int,1>(8,12,2) };
    //    i1->convert_to_indexed();
    i2 = shared_ptr<indexstruct<index_int,1>>{ new contiguous_indexstruct<index_int,1>(5,8) };
    REQUIRE_NOTHROW( i3 = i1->struct_union(i2) );
    CHECK( i3->local_size()==6 );
    CHECK( i3->first_index()==5 );
    CHECK( i3->last_index()==12);
  }
  SECTION( "tricky composite stuff" ) {
    shared_ptr<composite_indexstruct<index_int,1>> icomp;
    REQUIRE_NOTHROW
      ( icomp = shared_ptr<composite_indexstruct<index_int,1>>{ new composite_indexstruct<index_int,1>() } );
    CHECK( icomp->is_composite() );
    REQUIRE_NOTHROW
      ( icomp->push_back( shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(1,10) ) ) );
    REQUIRE_NOTHROW
      ( icomp->push_back( shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(31,40) ) ) );
    i1 = icomp->make_clone();
    CHECK( i1->is_composite() );
    vector< shared_ptr<indexstruct<index_int,1>> > members;
    indexstruct<index_int,1>ure I1;
    REQUIRE_NOTHROW( I1 = indexstruct<index_int,1>ure(composite_indexstruct<index_int,1>()) );
    CHECK( I1.is_composite() );
    REQUIRE_NOTHROW( I1.push_back( contiguous_indexstruct<index_int,1>(1,10) ) );
    return;
    REQUIRE_NOTHROW( I1.push_back( contiguous_indexstruct<index_int,1>(31,40) ) );
    SECTION( "can not merge" ) {
      i2 = shared_ptr<indexstruct<index_int,1>>{ new strided_indexstruct<index_int,1>(11,15,2) };
      REQUIRE_NOTHROW( i1 = i1->struct_union(i2) );
      REQUIRE_NOTHROW( members = dynamic_cast<composite_indexstruct<index_int,1>*>(i1.get())->get_structs() );
      CHECK( members.size()==3 );
    }
    SECTION( "can merge" ) {
      i2 = shared_ptr<indexstruct<index_int,1>>{ new contiguous_indexstruct<index_int,1>(11,15) };
      REQUIRE_NOTHROW( i1 = i1->struct_union(i2) );
      REQUIRE_NOTHROW( members = dynamic_cast<composite_indexstruct<index_int,1>*>(i1.get())->get_structs() );
      CHECK( members.size()==2 );
    }
    CHECK( i1->contains_element(10) );
    CHECK( i1->contains_element(11) );
  }
}

TEST_CASE( "struct disjoint","[23]" ) {
  shared_ptr<indexstruct<index_int,1>> i1,i2;
  indexstruct<index_int,1>ure I1,I2;
  SECTION( "disjoint strided" ) {
    REQUIRE_NOTHROW( i1 = shared_ptr<indexstruct<index_int,1>>( new strided_indexstruct<index_int,1>(1,10,2) ) );
    REQUIRE_NOTHROW( i2 = shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(10,20) ) );
    CHECK( i1->disjoint(i2) );
    REQUIRE_NOTHROW( I1 = indexstruct<index_int,1>ure(strided_indexstruct<index_int,1>(1,10,2)) );
    REQUIRE_NOTHROW( I2 = indexstruct<index_int,1>ure(contiguous_indexstruct<index_int,1>(10,20)) );
    CHECK( I1.disjoint(I2) );
  }
  SECTION( "disjoint strided, interleaved" ) {
    REQUIRE_NOTHROW( i1 = shared_ptr<indexstruct<index_int,1>>( new strided_indexstruct<index_int,1>(1,10,2) ) );
    REQUIRE_NOTHROW( i2 = shared_ptr<indexstruct<index_int,1>>( new strided_indexstruct<index_int,1>(8,20,2) ) );
    CHECK( i1->disjoint(i2) );
  }
  SECTION( "disjoint strided, hard to tell" ) {
    REQUIRE_NOTHROW( i1 = shared_ptr<indexstruct<index_int,1>>( new strided_indexstruct<index_int,1>(0,10,5) ) );
    REQUIRE_NOTHROW( i2 = shared_ptr<indexstruct<index_int,1>>( new strided_indexstruct<index_int,1>(7,9,2) ) );
    CHECK_THROWS( i1->disjoint(i2) );
  }
  SECTION( "disjoint indexed & strided, range" ) {
    REQUIRE_NOTHROW( i1 = shared_ptr<indexstruct<index_int,1>>
		     ( new indexed_indexstruct<index_int,1>(vector<index_int>{1,3,6}) ) );
    REQUIRE_NOTHROW( i2 = shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(7,10) ) );
    SECTION( "one way" ) { CHECK( i1->disjoint(i2) ); }
    SECTION( "oth way" ) { CHECK( i2->disjoint(i1) ); }
  }
  SECTION( "disjoint indexed, range" ) {
    REQUIRE_NOTHROW( i1 = shared_ptr<indexstruct<index_int,1>>
		     ( new indexed_indexstruct<index_int,1>(vector<index_int>{1,3,6}) ) );
    REQUIRE_NOTHROW( i2 = shared_ptr<indexstruct<index_int,1>>
		     ( new indexed_indexstruct<index_int,1>(vector<index_int>{7,8,10}) ) );
    CHECK( i1->disjoint(i2) );
  }
  SECTION( "disjoint indexed, hard to tell" ) {
    REQUIRE_NOTHROW( i1 = shared_ptr<indexstruct<index_int,1>>
		     ( new indexed_indexstruct<index_int,1>(vector<index_int>{1,3,16}) ) );
    REQUIRE_NOTHROW( i2 = shared_ptr<indexstruct<index_int,1>>
		     ( new indexed_indexstruct<index_int,1>(vector<index_int>{7,8,10}) ) );
    CHECK_THROWS( i1->disjoint(i2) );
    REQUIRE_NOTHROW( I1 = indexstruct<index_int,1>ure(indexed_indexstruct<index_int,1>(vector<index_int>{1,3,16})) );
    REQUIRE_NOTHROW( I2 = indexstruct<index_int,1>ure(indexed_indexstruct<index_int,1>(vector<index_int>{7,8,10})) );
    CHECK_THROWS( I1.disjoint(I2) );
  }
}

TEST_CASE( "struct containment","[24]" ) {
  shared_ptr<indexstruct<index_int,1>> i1,i2,i3;
  indexstruct<index_int,1>ure I1,I2,I3;
  SECTION( "cont" ) {
    i1 = shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(1,10) );

    i2 = shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(2,10) );
    REQUIRE( i1->contains(i2) );
    REQUIRE( !i2->contains(i1) );

    i2 = shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(2,11) );
    REQUIRE( !i1->contains(i2) );
    REQUIRE( !i2->contains(i1) );

    i2 = shared_ptr<indexstruct<index_int,1>>( new strided_indexstruct<index_int,1>(3,11,3) ); // 3,6,9
    REQUIRE( i1->contains(i2) );
    REQUIRE( !i2->contains(i1) );

    i2 = shared_ptr<indexstruct<index_int,1>>( new strided_indexstruct<index_int,1>(4,11,3) ); // 4,7,10
    REQUIRE( i1->contains(i2) );
    REQUIRE( !i2->contains(i1) );
  }
  SECTION( "stride" ) {
    i1 = shared_ptr<indexstruct<index_int,1>>( new strided_indexstruct<index_int,1>(10,20,3) );

    i2 = shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(13) );
    REQUIRE( i1->contains(i2) );
  }
  SECTION( "idx" ) {
    int len=5; index_int idx[5] = {1,2,4,6,9};
    i1 = shared_ptr<indexstruct<index_int,1>>( new indexed_indexstruct<index_int,1>(len,idx) );

    i2 = shared_ptr<indexstruct<index_int,1>>( new strided_indexstruct<index_int,1>(4,4,3) );
    REQUIRE( i1->contains(i2) );

    i2 = shared_ptr<indexstruct<index_int,1>>( new strided_indexstruct<index_int,1>(4,6,2) );
    REQUIRE( i1->contains(i2) );

    i2 = shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(4,4) );
    REQUIRE( i1->contains(i2) );
  }
  SECTION( "composite" ) {
    auto i1 = shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(0,10) );
    REQUIRE_NOTHROW( i1 = i1->struct_union
		     ( shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(20,29) ) ) );
    CHECK( i1->is_composite() );

    i2 = shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(2,5) );
    CHECK( i1->contains(i2) );

    I1 = indexstruct<index_int,1>ure(contiguous_indexstruct<index_int,1>(0,10));
    I1 = I1.struct_union(indexstruct<index_int,1>ure(contiguous_indexstruct<index_int,1>(20,29)));
    I2 = indexstruct<index_int,1>ure(contiguous_indexstruct<index_int,1>(2,5));
    CHECK( I1.contains(I2) );

    i2 = shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(22,25) );
    CHECK( i1->contains(i2) );
    I2 = indexstruct<index_int,1>ure(contiguous_indexstruct<index_int,1>(22,25));
    CHECK( I1.contains(I2) );

    i2 = shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(8,25) );
    CHECK( !i1->contains(i2) );
    I2 = indexstruct<index_int,1>ure(contiguous_indexstruct<index_int,1>(8,25));
    CHECK( !I1.contains(I2) );

  }
}

TEST_CASE( "struct split","[split][25]" ) {
  SECTION( "contiguous" ) {
    auto i1 = shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(10,20) );
    indexstruct<index_int,1>ure I1(contiguous_indexstruct<index_int,1>(10,20));
    shared_ptr<indexstruct<index_int,1>> c;
    SECTION( "non intersect" ) {
      auto i2 = shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(30,40) );
      REQUIRE_THROWS( c = i1->split(i2) );
    }
    SECTION( "contains" ) {
      auto i2 = shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(0,40) );
      REQUIRE_THROWS( c = i1->split(i2) );
    }
    SECTION( "right" ) {
      auto i2 = shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(15,30) );
      REQUIRE_NOTHROW( c = i1->split(i2) );
      composite_indexstruct<index_int,1> *cc = dynamic_cast<composite_indexstruct<index_int,1>*>(c.get());
      CHECK( cc!=nullptr );
      auto ss = cc->get_structs();
      CHECK( ss.size()==2 );
      auto l = shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(10,14) );
      CHECK( ss.at(0)->equals(l) );
      auto r = shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(15,20) );
      CHECK( ss.at(1)->equals(r) );
    }
    SECTION( "left" ) {
      auto i2 = shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(0,15) );
      REQUIRE_NOTHROW( c = i1->split(i2) );
      composite_indexstruct<index_int,1> *cc = dynamic_cast<composite_indexstruct<index_int,1>*>(c.get());
      CHECK( cc!=nullptr );
      auto ss = cc->get_structs();
      CHECK( ss.size()==2 );
      auto l = shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(10,15) );
      CHECK( ss.at(0)->equals(l) );
      auto r = shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(16,20) );
      CHECK( ss.at(1)->equals(r) );
    }
  }
  SECTION( "strided" ) {
    auto i1 = shared_ptr<indexstruct<index_int,1>>( new strided_indexstruct<index_int,1>(10,20,2) );
    indexstruct<index_int,1>ure I1(strided_indexstruct<index_int,1>(10,20,2));
    shared_ptr<indexstruct<index_int,1>> c;
    SECTION( "non intersect" ) {
      auto i2 = shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(30,40) );
      indexstruct<index_int,1>ure I2(contiguous_indexstruct<index_int,1>(30,40)), c;
      REQUIRE_THROWS( c = i1->split(i2) );
      REQUIRE_THROWS( c = I1.split(I2) );
    }
    SECTION( "contains" ) {
      auto i2 = shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(0,40) );
      REQUIRE_THROWS( c = i1->split(i2) );
    }
    SECTION( "right" ) {
      auto i2 = shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(15,30) );
      REQUIRE_NOTHROW( c = i1->split(i2) );
      INFO( format("split to: {}",c->as_string()) );
      composite_indexstruct<index_int,1> *cc = dynamic_cast<composite_indexstruct<index_int,1>*>(c.get());
      CHECK( cc!=nullptr );
      auto ss = cc->get_structs();
      CHECK( ss.size()==2 );
      {
	auto l = shared_ptr<indexstruct<index_int,1>>( new strided_indexstruct<index_int,1>(10,14,2) );
	CHECK( ss.at(0)->equals(l) );
      }
      {
	auto r = shared_ptr<indexstruct<index_int,1>>( new strided_indexstruct<index_int,1>(16,20,2) );
	INFO( format("Is {}, should be {}",ss.at(1)->as_string(),r->as_string()) );
	CHECK( ss.at(1)->equals(r) );
      }
    }
    SECTION( "left" ) {
      auto i2 = shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(0,15) );
      REQUIRE_NOTHROW( c = i1->split(i2) );
      composite_indexstruct<index_int,1> *cc = dynamic_cast<composite_indexstruct<index_int,1>*>(c.get());
      CHECK( cc!=nullptr );
      auto ss = cc->get_structs();
      CHECK( ss.size()==2 );
      auto l = shared_ptr<indexstruct<index_int,1>>( new strided_indexstruct<index_int,1>(10,14,2) );
      CHECK( ss.at(0)->equals(l) );
      auto r = shared_ptr<indexstruct<index_int,1>>( new strided_indexstruct<index_int,1>(16,20,2) );
      CHECK( ss.at(1)->equals(r) );
    }
  }
}

index_int itimes2(index_int i) { return 2*i; }
indexstruct<index_int,1> *itimes2i(index_int i) { return new contiguous_indexstruct<index_int,1>(2*i,2*i); }

TEST_CASE( "structs and operations","[indexstruct<index_int,1>][operate][30]" ) {
  shared_ptr<indexstruct<index_int,1>> i1;
  shared_ptr<indexstruct<index_int,1>> i2;
  indexstruct<index_int,1>ure I1,I2;
  ioperator op; 

  SECTION( "multiply by constant" ) {
    i1 = shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(5,10) );
    I1 = indexstruct<index_int,1>ure(contiguous_indexstruct<index_int,1>(5,10));
    REQUIRE_NOTHROW( op = ioperator("*2") );
    REQUIRE_NOTHROW( i2 = i1->operate(op) );
    REQUIRE_NOTHROW( I2 = I1.operate(op) );
    // printf("i2 s/b strided: <<%s>>\n",i2->as_string().data());
    CHECK( i2->is_strided() );
    CHECK( i2->first_index()==10 );
    CHECK( i2->last_index()==20 );
    CHECK( i2->local_size()==i1->local_size() );
    CHECK( i2->stride()==2 );
  }
  SECTION( "multiply range by constant" ) {
    i1 = shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(5,10) );
    REQUIRE_NOTHROW( op = ioperator("x2") );
    REQUIRE_NOTHROW( i2 = i1->operate(op) );
    CHECK( i2->is_contiguous() );
    CHECK( i2->first_index()==10 );
    CHECK( i2->last_index()==21 );
    CHECK( i2->local_size()==i1->local_size()*2 );
    // CHECK( i2->is_strided() );
    CHECK( i2->stride()==1 );
    REQUIRE_NOTHROW( I1 = indexstruct<index_int,1>ure(i1) );
    REQUIRE_NOTHROW( I2 = I1.operate(op) );
    CHECK( I2.is_contiguous() );
    CHECK( I2.last_index()==21 );
  }
  SECTION( "multiply by function" ) {
    i1 = shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(5,10) );
    REQUIRE_NOTHROW( op = ioperator(&itimes2) );
    index_int is2;
    REQUIRE_NOTHROW( is2 = op.operate(1) );
    CHECK( is2==2 );
    
    REQUIRE_NOTHROW( i2 = i1->operate(op) );
    // CHECK( i2->is_strided() ); // VLE can we get this to work?
    CHECK( i2->first_index()==10 );
    CHECK( i2->last_index()==20 );
  }
  SECTION( "shift strided" ) {
    i1 = shared_ptr<indexstruct<index_int,1>>( new strided_indexstruct<index_int,1>(1,10,2) );
    CHECK( i1->first_index()==1 );
    CHECK( i1->last_index()==9 );
    CHECK( i1->local_size()==5 );
    SECTION( "bump" ) {
      REQUIRE_NOTHROW( op = ioperator("<=1") );
    }
    SECTION( "mod" ) {
      REQUIRE_NOTHROW( op = ioperator("<<1") );
    }
    REQUIRE_NOTHROW( i2 = i1->operate(op) );
    CHECK( i2->first_index()==0 );
    CHECK( i2->last_index()==8 );
    CHECK( i2->local_size()==5 );
  }
  SECTION( "test truncating by itself" ) {
    i1 = shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>( 1,10 ) );
    REQUIRE_NOTHROW( i2 = i1->truncate_left(4) );
    CHECK( i2->is_contiguous() );
    CHECK( i2->first_index()==4 );
  }
  SECTION( "shift strided with truncate" ) {
    i1 = shared_ptr<indexstruct<index_int,1>>( new strided_indexstruct<index_int,1>(0,10,2) );
    REQUIRE_NOTHROW( op = ioperator("<=1") );
    REQUIRE_NOTHROW( i2 = i1->operate(op,0,100) );
    CHECK( i2->first_index()==1 );
    CHECK( i2->last_index()==9 );
    CHECK( i2->local_size()==5 );
    REQUIRE_NOTHROW( I1 = indexstruct<index_int,1>ure(i1) );
    REQUIRE_NOTHROW( I2 = I1.operate(op,0,100) );
    CHECK( I2.local_size()==5 );
  }
}

TEST_CASE( "division operation","[indexstruct<index_int,1>][operate][31]" ) {
  shared_ptr<indexstruct<index_int,1>> i1,i2; ioperator op;

  SECTION( "simple division" ) {
    REQUIRE_NOTHROW( op = ioperator("/2") );
  
    SECTION( "contiguous1" ) {
      i1 = shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(0,10) );
      REQUIRE_NOTHROW( i2 = i1->operate(op) );
      CHECK( i2->first_index()==0 );
      CHECK( i2->last_index()==5 );
    }
    SECTION( "contiguous2" ) {
      i1 = shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(0,9) );
      REQUIRE_NOTHROW( i2 = i1->operate(op) );
      CHECK( i2->first_index()==0 );
      CHECK( i2->last_index()==4 );
    }
  }
  SECTION( "contiguous division" ) {
    REQUIRE_NOTHROW( op = ioperator(":2") );
  
    SECTION( "contiguous1" ) {
      i1 = shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(0,10) );
      REQUIRE_NOTHROW( i2 = i1->operate(op) );
      CHECK( i2->first_index()==0 );
      CHECK( i2->last_index()==4 );
    }
    SECTION( "contiguous2" ) {
      i1 = shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(0,9) );
      REQUIRE_NOTHROW( i2 = i1->operate(op) );
      CHECK( i2->first_index()==0 );
      CHECK( i2->last_index()==4 );
    }
  }
}

TEST_CASE( "copy indexstruct<index_int,1>","[indexstruct<index_int,1>][copy][42]" ) {
  shared_ptr<indexstruct<index_int,1>> i1,i2;

  i1 = shared_ptr<indexstruct<index_int,1>>{ new contiguous_indexstruct<index_int,1>(0,10) };
  REQUIRE_NOTHROW( i2 = i1->make_clone() );
  // make a copy
  CHECK( i1->local_size()==i2->local_size() );
  CHECK( i1->first_index()==i2->first_index() );
  CHECK( i1->last_index()==i2->last_index() );

  // shift the original
  REQUIRE_NOTHROW( i1 = i1->translate_by(1) );
  CHECK( i1->local_size()==i2->local_size() );
  CHECK( i1->first_index()==i2->first_index()+1 );
  CHECK( i1->last_index()==i2->last_index()+1 );
}

TEST_CASE( "big shift operators","[operator][shift][43]" ) {
  ioperator i;
  i = ioperator("shift",5);
  CHECK( i.operate(6)==11 );
  CHECK_NOTHROW( i.operate(-11)==-6 );
  i = ioperator("shift",-3);
  CHECK( i.operate(6)==3 );
  CHECK( i.operate(1)==-2 );
  CHECK_NOTHROW( i.operate(-6)==-2 );  
}

TEST_CASE( "arbitrary shift of indexstruct<index_int,1>","[indexstruct<index_int,1>][shift][44]" ) {
  shared_ptr<indexstruct<index_int,1>> i1,i2;
  indexstruct<index_int,1>ure I1,I2;
  i1 = shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(5,7) );
  REQUIRE_NOTHROW( I1 = indexstruct<index_int,1>ure(i1) );
  SECTION( "shift by" ) {
    REQUIRE_NOTHROW( i2 = i1->operate( ioperator("shift",7) ) );
    CHECK( i2->first_index()==12 );
    CHECK( i2->last_index()==14 );
    REQUIRE_NOTHROW( I2 = I1.operate( ioperator("shift",7) ) );
    CHECK( I2.first_index()==12 );
    CHECK( I2.last_index()==14 );
  }
  SECTION( "shift to" ) {
    REQUIRE_NOTHROW( i2 = i1->operate( ioperator("shiftto",7) ) );
    CHECK( i2->first_index()==7 );
    CHECK( i2->last_index()==9 );
  }

}

TEST_CASE( "sigma operator stuff","[50]" ) {
  sigma_operator sop;
  //auto times2 = shared_ptr<ioperator>( new ioperator("*2") );
  ioperator times2("*2");
  auto cont = shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(10,20) );
  shared_ptr<indexstruct<index_int,1>> sstruct; const char *path;
  SECTION( "point" ) { path = "operate by point";
    REQUIRE_NOTHROW( sop = sigma_operator(times2) );
  }
  SECTION( "struct" ) { path = "operate by struct";
    REQUIRE_NOTHROW
      ( sop = sigma_operator
	( [times2] ( const indexstruct<index_int,1> &i) -> shared_ptr<indexstruct<index_int,1>>
	  { return i.operate(times2); } ) );
  }
  INFO( "path: " << path );
  REQUIRE_NOTHROW( sstruct = cont->operate(sop) );
  CHECK( sstruct->first_index()==20 );
  CHECK( sstruct->last_index()==40 );
  indexstruct<index_int,1>ure Cont,Sstruct;
  REQUIRE_NOTHROW( Cont = indexstruct<index_int,1>ure(cont) );
  REQUIRE_NOTHROW( Sstruct = Cont.operate(sop) );
  CHECK( Sstruct.first_index()==20 );
  CHECK( Sstruct.last_index()==40 );
}

TEST_CASE( "create multidimensional by component","[multi][indexstruct<index_int,1>][100]" ) {
  shared_ptr<multi_indexstruct<index_int,1>> mi,mt;
  multi_indexstruct<index_int,1>ure Mi,Mt;

  // we can not create zero-dimensional
  REQUIRE_THROWS( mi = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>(0) ) );
  REQUIRE_THROWS( Mi = multi_indexstruct<index_int,1>ure(0) );
  // create two-dimensional
  REQUIRE_NOTHROW( mi = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>(2) ) );
  REQUIRE_NOTHROW( Mi = multi_indexstruct<index_int,1>ure(2) );
  // set two component
  REQUIRE_NOTHROW( mi->set_component
		   ( 0, shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(2,10) ) ) );
  REQUIRE_NOTHROW( mi->set_component
		   ( 1, shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(40,41) ) ) );
  CHECK( mi->volume()==9*2 );
  // can set twice
  REQUIRE_NOTHROW( mi->set_component
		   ( 1, shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(30,41) ) ) );
  CHECK( mi->volume()==9*12 );
  // can not set outside dimension bounds
  REQUIRE_THROWS( mi->set_component
		  ( 2, shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(40,41) ) ) );
  CHECK( mi->volume()==9*12 );
  domain_coordinate val(1);
  REQUIRE_NOTHROW( val = mi->local_size_r() );
  CHECK( val.at(0)==9 );
  CHECK( val.at(1)==12 );
  REQUIRE_NOTHROW( val = mi->first_index_r() );
  CHECK( val.at(0)==2 );
  CHECK( val.at(1)==30 );
  REQUIRE_NOTHROW( val = mi->last_index_r() );
  CHECK( val.at(0)==10 );
  CHECK( val.at(1)==41 );

  mt = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>(2) );
  mt->set_component(0,shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(4,9)) );
  mt->set_component(1,shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(40,40)) );
  REQUIRE_NOTHROW( val = mi->location_of(mt) );
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
    vector<shared_ptr<indexstruct<index_int,1>>> blocks(dim);
    for (int id=0; id<dim; id++)
      blocks[id] = shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(0,nlocal+id-1) );
    m = multi_indexstruct<index_int,1>(blocks);
  }
  REQUIRE( m.volume()==v );
}

TEST_CASE( "multidimensional containment","[multi][111]" ) {
  shared_ptr<multi_indexstruct<index_int,1>> m1,m1a,m2,m2a;
  SECTION( "simple:" ) {
    REQUIRE_NOTHROW
      ( m1 = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>
	( vector<shared_ptr<indexstruct<index_int,1>>>{
	  shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(0,9) ) } ) ) );
    INFO( "m1:" << m1->as_string() );
    SECTION( "contains simple" ) {
      REQUIRE_NOTHROW
	( m2 = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>
	  ( vector<shared_ptr<indexstruct<index_int,1>>>{
	    shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(1,5) ) } ) ) );
      INFO( "m2:" << m2->as_string() );
      CHECK( m1->contains(m2) );
    }
    SECTION( "not contains simple" ) {
      REQUIRE_NOTHROW
	( m2 = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>
	  ( vector<shared_ptr<indexstruct<index_int,1>>>{
	    shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(1,15) ) } ) ) );
      INFO( "m2:" << m2->as_string() );
      CHECK( !m1->contains(m2) );
    }
  }
  SECTION( "multi:" ) {
    REQUIRE_NOTHROW
      ( m1 = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>
	( vector<shared_ptr<indexstruct<index_int,1>>>{
	  shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(0,9) ) } ) ) );
    REQUIRE_NOTHROW
      ( m1a = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>
	( vector<shared_ptr<indexstruct<index_int,1>>>{
	  shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(20,29) ) } )
						  ) );
    REQUIRE_NOTHROW( m1 = m1->struct_union(m1a) );
    INFO( "m1:" << m1->as_string() );
    SECTION( "contains simple" ) {
      REQUIRE_NOTHROW
	( m2 = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>
	  ( vector<shared_ptr<indexstruct<index_int,1>>>{
	    shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(1,5) ) } ) ) );
      INFO( "m2:" << m2->as_string() );
      CHECK( m1->contains(m2) );
    }
    SECTION( "not contains simple" ) {
      REQUIRE_NOTHROW
	( m2 = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>
	  ( vector<shared_ptr<indexstruct<index_int,1>>>{
	    shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(5,25) ) } ) ) );
      INFO( "m2:" << m2->as_string() );
      CHECK( !m1->contains(m2) );
    }
  }
}

TEST_CASE( "True multidimensional containment","[multi][112]" ) {
  shared_ptr<multi_indexstruct<index_int,1>> m1,m1a,m2,m2a;
  SECTION( "simple:" ) {
    REQUIRE_NOTHROW
      ( m1 = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>
	( vector<shared_ptr<indexstruct<index_int,1>>>{
	  shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(0,9) ),
	  shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(0,9) )
	    } ) ) );
    INFO( "m1:" << m1->as_string() );
    SECTION( "contains simple" ) {
      REQUIRE_NOTHROW
	( m2 = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>
	  ( vector<shared_ptr<indexstruct<index_int,1>>>{
	    shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(1,5) ),
	    shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(1,5) )
	      } ) ) );
      INFO( "m2:" << m2->as_string() );
      CHECK( m1->contains(m2) );
    }
    SECTION( "not contains simple" ) {
      REQUIRE_NOTHROW
	( m2 = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>
	  ( vector<shared_ptr<indexstruct<index_int,1>>>{
	    shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(1,15) ),
	    shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(1,15) )
	      } ) ) );
      INFO( "m2:" << m2->as_string() );
      CHECK( !m1->contains(m2) );
    }
  }
  SECTION( "multi:" ) {
    REQUIRE_NOTHROW
      ( m1 = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>
	( vector<shared_ptr<indexstruct<index_int,1>>>{
	  shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(0,9) ),
	  shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(0,9) )
	    } ) ) );
    REQUIRE_NOTHROW
      ( m1a = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>
	( vector<shared_ptr<indexstruct<index_int,1>>>{
	  shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(20,29) ),
	  shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(20,29) )
	    } ) )
	);
    REQUIRE_NOTHROW( m1 = m1->struct_union(m1a) );
    INFO( "m1:" << m1->as_string() );
    SECTION( "contains simple" ) {
      REQUIRE_NOTHROW
	( m2 = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>
	  ( vector<shared_ptr<indexstruct<index_int,1>>>{
	    shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(1,5) ),
	    shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(1,5) )
	      } ) ) );
      INFO( "m2:" << m2->as_string() );
      CHECK( m1->contains(m2) );
    }
    SECTION( "not contains simple" ) {
      REQUIRE_NOTHROW
	( m2 = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>
	  ( vector<shared_ptr<indexstruct<index_int,1>>>{
	    shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(5,25) ),
	    shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(5,25) )
	      } ) ) );
      INFO( "m2:" << m2->as_string() );
      CHECK( !m1->contains(m2) );
    }
  }
}

TEST_CASE( "multidimensional union","[multi][union][indexstruct<index_int,1>][113]" ) {
  shared_ptr<multi_indexstruct<index_int,1>> mi1,mi2,mi3; int diff;
  REQUIRE_NOTHROW( mi1 = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>(1) ) );
  REQUIRE_NOTHROW( mi2 = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>(2) ) );
  REQUIRE_THROWS( mi1->struct_union(mi2) );

  REQUIRE_NOTHROW( mi1 = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>(2) ) );
  REQUIRE_NOTHROW( mi1->set_component
		   ( 0, shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(10,20) ) ) );
  REQUIRE_NOTHROW( mi1->set_component
		   ( 1, shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(30,40) ) ) );

  SECTION( "fit in dimension 1: multis are merged" ) {
    REQUIRE_NOTHROW( mi2->set_component
		     ( 0, shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(10,20) ) ) );
    REQUIRE_NOTHROW( mi2->set_component
		     ( 1, shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(31,41) ) ) );
    CHECK( mi1->can_union_in_place(mi2,diff) );
    CHECK( mi2->can_union_in_place(mi1,diff) );
    REQUIRE_NOTHROW( mi3 = mi1->struct_union(mi2)->force_simplify() );
    INFO( format("Union {} and {} gives {}",
		      mi1->as_string(),mi2->as_string(),mi3->as_string()) );
    CHECK( !mi3->is_multi() );
    CHECK( mi3->first_index_r()==domain_coordinate( vector<index_int>{10,30} ) );
    CHECK( mi3->last_index_r()==domain_coordinate( vector<index_int>{20,41} ) );
  }

  SECTION( "fit in dimension 2: multis are merged" ) {
    REQUIRE_NOTHROW( mi2->set_component
		     ( 0, shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(12,22) ) ) );
    REQUIRE_NOTHROW( mi2->set_component
		     ( 1, shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(30,40) ) ) );
    CHECK( mi1->can_union_in_place(mi2,diff) );
    CHECK( mi2->can_union_in_place(mi1,diff) );
    REQUIRE_NOTHROW( mi3 = mi1->struct_union(mi2)->force_simplify() );
    CHECK( !mi3->is_multi() );
    CHECK( mi3->first_index_r()==domain_coordinate( vector<index_int>{10,30} ) );
    CHECK( mi3->last_index_r()==domain_coordinate( vector<index_int>{22,40} ) );
  }

  SECTION( "unfit: first comes from one and last second original" ) {
    // m1 = [ (10,30) - (20,40) ]
    // m2 = [ (12,31) - (22,41) ]  so we store pointers to both
    // first = (10,30) last = (22,41)
    REQUIRE_NOTHROW( mi2->set_component
  		     ( 0, shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(12,22) ) ) );
    REQUIRE_NOTHROW( mi2->set_component
  		     ( 1, shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(31,41) ) ) );
    REQUIRE_NOTHROW( mi3 = mi1->struct_union(mi2) );
    INFO( format("{} & {} gives {}",mi1->as_string(),mi2->as_string(),mi3->as_string()) );
    CHECK( mi3->is_multi() );
    auto first = mi3->first_index_r(), last = mi3->last_index_r();
    INFO( format("first = {}, last = {}",first.as_string(),last.as_string()) );
    CHECK( first==domain_coordinate( vector<index_int>{10,30} ) );
    CHECK( last==domain_coordinate( vector<index_int>{22,41} ) );
  }
  SECTION( "very unfit: case where union first/last are not in the originals" ) {
    // m1 = [ (10,30) - (20,40) ]
    // m2 = [ (20,10) - (30,20) ]  so we store pointers to both
    // first = (10,10) last = (30,40)
    REQUIRE_NOTHROW( mi2->set_component
  		     ( 0, shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(20,30) ) ) );
    REQUIRE_NOTHROW( mi2->set_component
  		     ( 1, shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(10,20) ) ) );
    REQUIRE_NOTHROW( mi3 = mi1->struct_union(mi2) );
    CHECK( mi3->is_multi() );
    auto first = mi3->first_index_r(), last = mi3->last_index_r();
    INFO( format("first = {}, last = {}",first.as_string(),last.as_string()) );
    CHECK( first==domain_coordinate( vector<index_int>{10,10} ) );
    CHECK( last==domain_coordinate( vector<index_int>{30,40} ) );
  }
  SECTION( "union of more than two" ) {
    auto
      m1 = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>
      ( vector<shared_ptr<indexstruct<index_int,1>>>
	{ shared_ptr<indexstruct<index_int,1>>(new contiguous_indexstruct<index_int,1>(1,1)),
	    shared_ptr<indexstruct<index_int,1>>(new contiguous_indexstruct<index_int,1>(1,1))
	    } ) ),
      m2 = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>
      ( vector<shared_ptr<indexstruct<index_int,1>>>
	{ shared_ptr<indexstruct<index_int,1>>(new contiguous_indexstruct<index_int,1>(2,2)),
	    shared_ptr<indexstruct<index_int,1>>(new contiguous_indexstruct<index_int,1>(2,2))
	    } ) ),
      m3 = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>
      ( vector<shared_ptr<indexstruct<index_int,1>>>
	{ shared_ptr<indexstruct<index_int,1>>(new contiguous_indexstruct<index_int,1>(3,3)),
	    shared_ptr<indexstruct<index_int,1>>(new contiguous_indexstruct<index_int,1>(3,3))
	    } ) );
    shared_ptr<multi_indexstruct<index_int,1>> u;
    // m2 makes it a multi
    CHECK( !m1->contains(m2) );
    REQUIRE_NOTHROW( u = m1->struct_union(m2) );
    INFO( format("union 1 & 2: {}",u->as_string()) );
    CHECK( u->multi_size()==2 );
    // m3 is added as another multi
    CHECK( !u->contains(m3) );
    REQUIRE_NOTHROW( u = u->struct_union(m3) );
    INFO( format("union 1 & 2 & 3: {}",u->as_string()) );
    CHECK( u->multi_size()==3 );
  }
  SECTION( "multi union with recombination" ) {
    auto 
      m1 = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>
      ( vector<shared_ptr<indexstruct<index_int,1>>>
	{ shared_ptr<indexstruct<index_int,1>>(new contiguous_indexstruct<index_int,1>(1,2)),
	    shared_ptr<indexstruct<index_int,1>>(new contiguous_indexstruct<index_int,1>(1,1))
	    } ) ),
      m2 = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>
      ( vector<shared_ptr<indexstruct<index_int,1>>>
	{ shared_ptr<indexstruct<index_int,1>>(new contiguous_indexstruct<index_int,1>(2,2)),
	    shared_ptr<indexstruct<index_int,1>>(new contiguous_indexstruct<index_int,1>(2,2))
	    } ) );
    shared_ptr<multi_indexstruct<index_int,1>> m3,u;
    REQUIRE_NOTHROW( u = m1->struct_union(m2) );
    INFO( format("union 1 & 2: {}",u->as_string()) );
    CHECK( u->multi_size()==2 );
    SECTION( "absorb" ) {
      REQUIRE_NOTHROW( m3 = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>
	( vector<shared_ptr<indexstruct<index_int,1>>>
	  { shared_ptr<indexstruct<index_int,1>>(new contiguous_indexstruct<index_int,1>(2,2)),
	      shared_ptr<indexstruct<index_int,1>>(new contiguous_indexstruct<index_int,1>(1,1))
	      } ) ) );
      REQUIRE_NOTHROW( u = u->struct_union(m3)->force_simplify() );
      INFO( format("absorb {} gives union 1 & 2 & 3: {}",m3->as_string(),u->as_string()) );
      CHECK( u->is_multi() );
      CHECK( u->multi_size()==2 );
    }
    SECTION( "merge" ) {
      REQUIRE_NOTHROW( m3 = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>
	( vector<shared_ptr<indexstruct<index_int,1>>>
	  { shared_ptr<indexstruct<index_int,1>>(new contiguous_indexstruct<index_int,1>(1,1)),
	      shared_ptr<indexstruct<index_int,1>>(new contiguous_indexstruct<index_int,1>(2,2))
	      } ) ) );
      REQUIRE_NOTHROW( u = u->struct_union(m3)->force_simplify() );
      INFO( format("merge {} give union 1 & 2 & 3: {}",m3->as_string(),u->as_string()) );
      CHECK( !u->is_multi() );
      CHECK( u->volume()==4 );
    }
  }
}

TEST_CASE( "multidimensional intersection","[multi][indexstruct<index_int,1>][114]" ) {
  shared_ptr<multi_indexstruct<index_int,1>> mi1,mi2,mi3;
  REQUIRE_NOTHROW( mi1 = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>(1) ) );
  REQUIRE_NOTHROW( mi2 = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>(2) ) );
  REQUIRE_THROWS( mi1->struct_union(mi2) );

  REQUIRE_NOTHROW( mi1 = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>(2) ) );
  REQUIRE_NOTHROW( mi1->set_component
		   ( 0, shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(0,10) ) ) );
  REQUIRE_NOTHROW( mi1->set_component
		   ( 1, shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(0,10) ) ) );

  REQUIRE_NOTHROW( mi2->set_component
		   ( 0, shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(10,20) ) ) );
  REQUIRE_NOTHROW( mi2->set_component
		   ( 1, shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(10,20) ) ) );
  REQUIRE_NOTHROW( mi3 = mi1->intersect(mi2) );

  INFO( format("Intersection runs {}--{}",
		    mi3->first_index_r().as_string(),
		    mi3->last_index_r().as_string()) );
  CHECK( mi3->volume()==1 );
}

TEST_CASE( "multi-dimensional minus","[multi][indexstruct<index_int,1>][115]" ) {
  shared_ptr<multi_indexstruct<index_int,1>> one,two,three;
  index_int lo = 1, hi = 4; bool success{true};
  REQUIRE_NOTHROW( one = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>
		   ( vector< shared_ptr<indexstruct<index_int,1>> >
		     { shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(1,2) ),
			 shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(5,8) )
			 }
		     )
							     ) );
  SECTION( "royally fits" ) {
    REQUIRE_NOTHROW( two = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>
		     ( vector< shared_ptr<indexstruct<index_int,1>> >
		       { shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(0,3) ),
			   shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(6,8) )
			   }
		       )
							       ) );
  }
  SECTION( "royally fits" ) {
    REQUIRE_NOTHROW( two = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>
  		     ( vector< shared_ptr<indexstruct<index_int,1>> >
  		       { shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(1,2) ),
  			   shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(6,8) )
  			   }
  		       )
  							       ) );
  }
  SECTION( "no fits" ) {
    REQUIRE_NOTHROW( two = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>
  		     ( vector< shared_ptr<indexstruct<index_int,1>> >
  		       { shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(2,2) ),
  			   shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(6,8) )
  			   }
  		       )
  							       ) );
    success = false;
  }
  if (!success) { // this used to be unimplemented
    REQUIRE_NOTHROW( three = one->minus(two) );
  } else {
    REQUIRE_NOTHROW( three = one->minus(two) );
    INFO( format("minus result: {}",three->as_string()) );
    CHECK( three->get_component(0)->equals
	   ( shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(1,2) ) ) );
    CHECK( three->get_component(1)->equals
	   ( shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(5,5) ) ) );
  }
}

TEST_CASE( "find linear location","[multi][locate][indexstruct<index_int,1>][116]" ) {
  shared_ptr<multi_indexstruct<index_int,1>> outer,inner;
  index_int location;

  SECTION( "one-d" ) {
    outer = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>(1) );
    SECTION( "outer contiguous" ) {
      outer->set_component
	(0,shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(11,20)) );
      SECTION( "same dimension" ) {
	inner = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>(2) );
	inner->set_component
	  (0,shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(15,16)) );
	inner->set_component
	  (1,shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(8,9)) );
	REQUIRE_THROWS( location = outer->linear_location_of(inner) );
      }
      SECTION( "proper" ) {
	inner = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>(1) );
	inner->set_component
	  (0,shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(15,16)) );
	REQUIRE_NOTHROW( location = outer->linear_location_of(inner) );
	CHECK( location==(15-11) );
      }
    }
  }
  SECTION( "two-d" ) {
    outer = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>(2) );
    SECTION( "outer contiguous" ) {
      outer->set_component  // 10 deep
	(0,shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(11,20)) );
      outer->set_component // 4 wide
	(1,shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(5,8)) );
      SECTION( "same dimension" ) {
	inner = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>(1) );
	inner->set_component
	  (0,shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(15,16)) );
	REQUIRE_THROWS( location = outer->linear_location_of(inner) );
      }
      SECTION( "proper" ) {
	inner = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>(2) );
	inner->set_component
	  (0,shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(15,16)) );
	inner->set_component
	  (1,shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(6,8)) );
	REQUIRE_NOTHROW( location = outer->linear_location_of(inner) );
	CHECK( location==(15-11)*4+6-5 );
      }
    }
  }
}

TEST_CASE( "relativize and find linear location","[multi][locate][indexstruct<index_int,1>][117]" ) {
  // this uses the same structure as [112]
  shared_ptr<multi_indexstruct<index_int,1>> outer,inner,relative;
  shared_ptr<indexstruct<index_int,1>> relativec;
  index_int location;

  SECTION( "one-d" ) {
    outer = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>(1) );
    SECTION( "outer contiguous" ) {
      outer->set_component
	(0,shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(11,20)) );
      SECTION( "same dimension" ) {
	inner = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>(2) );
	inner->set_component
	  (0,shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(15,16)) );
	inner->set_component
	  (1,shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(8,9)) );
	REQUIRE_THROWS( relative = inner->relativize_to(outer) );
      }
      SECTION( "proper" ) {
	inner = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>(1) );
	inner->set_component
	  (0,shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(15,16)) );
	REQUIRE_NOTHROW( relative = inner->relativize_to(outer) );
	REQUIRE_NOTHROW( relativec = relative->get_component(0) );
	REQUIRE_NOTHROW( relativec->is_contiguous() );
	REQUIRE_NOTHROW( relative->first_index(0)==4 );
      }
    }
  }
  SECTION( "two-d" ) {
    outer = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>(2) );
    SECTION( "outer contiguous" ) {
      outer->set_component
	(0,shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(11,20)) );
      outer->set_component
	(1,shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(5,8)) );
      SECTION( "same dimension" ) {
	inner = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>(1) );
	inner->set_component
	  (0,shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(15,16)) );
	REQUIRE_THROWS( relative = inner->relativize_to(outer) );
      }
      SECTION( "proper" ) {
	inner = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>(2) );
	inner->set_component
	  (0,shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(15,16)) );
	inner->set_component
	  (1,shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(6,8)) );
	REQUIRE_NOTHROW( relative = inner->relativize_to(outer) );
	REQUIRE_NOTHROW( relativec = relative->get_component(0) );
	REQUIRE_NOTHROW( relativec->is_contiguous() );
	REQUIRE_NOTHROW( relativec = relative->get_component(1) );
	REQUIRE_NOTHROW( relativec->is_contiguous() );
	REQUIRE_NOTHROW( relative->first_index(0)==4 );
	REQUIRE_NOTHROW( relative->first_index(1)==1 );
      }
    }
  }
}

TEST_CASE( "multi-dimensional operations","[multi][120]" ) {
  auto s1 = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>(2) );
  shared_ptr<multi_indexstruct<index_int,1>> s2;
  shared_ptr<indexstruct<index_int,1>> i0,i1;
  
  s1->set_component
    (0,shared_ptr<indexstruct<index_int,1>>(  new contiguous_indexstruct<index_int,1>(10,20) ) );
  s1->set_component
    (1,shared_ptr<indexstruct<index_int,1>>(  new strided_indexstruct<index_int,1>(25,50,5) ) );

  multi_ioperator *op; REQUIRE_NOTHROW( op = new multi_ioperator(2) );
  CHECK( op->get_dimensionality()==2 );

  SECTION( "first dim" ) {
    REQUIRE_NOTHROW( op->set_operator(0,ioperator(">>2")) );
    REQUIRE_NOTHROW( s2 = s1->operate(op) );
    REQUIRE_NOTHROW( i0 = s2->get_component(0) );
    CHECK( i0->is_contiguous() );
    CHECK( i0->first_index()==12 );
    CHECK( i0->last_index()==22 );
    REQUIRE_NOTHROW( i1 = s2->get_component(1) );
    CHECK( !i1->is_contiguous() );
    CHECK( i1->is_strided() );
    CHECK( i1->first_index()==25 );
    CHECK( i1->last_index()==50 );
    CHECK( i1->stride()==5 );
  }
  SECTION( "second dim" ) {
    REQUIRE_NOTHROW( op->set_operator(1,ioperator(">>2")) );
    REQUIRE_NOTHROW( s2 = s1->operate(op) );
    REQUIRE_NOTHROW( i0 = s2->get_component(0) );
    CHECK( i0->is_contiguous() );
    CHECK( i0->first_index()==10 );
    CHECK( i0->last_index()==20 );
    REQUIRE_NOTHROW( i1 = s2->get_component(1) );
    CHECK( !i1->is_contiguous() );
    CHECK( i1->is_strided() );
    CHECK( i1->first_index()==27 );
    CHECK( i1->last_index()==52 );
    CHECK( i1->stride()==5 );
  }
  SECTION( "both dimensions" ) {
    auto div2 = ioperator(">>2");
    REQUIRE_NOTHROW( op->set_operator(0,div2) );
    REQUIRE_NOTHROW( op->set_operator(1,div2) );
    REQUIRE_NOTHROW( s2 = s1->operate(op,s1) );
    REQUIRE_NOTHROW( i0 = s2->get_component(0) );
    REQUIRE_NOTHROW( i1 = s2->get_component(1) );
    CHECK( i0->first_index()==12 );
    CHECK( i0->last_index()==20 );
    CHECK( i1->first_index()==27 );
    CHECK( i1->last_index()==47 );
  }
  SECTION( "operate with truncation" ) {
  }
}

TEST_CASE( "multi dimensional iteration, 1d","[multi][range][125]" ) {
  int dim = 1; index_int f=5, l=8;

  multi_indexstruct<index_int,1> segment
    ( shared_ptr<indexstruct<index_int,1>>( new contiguous_indexstruct<index_int,1>(f,l) ) );
  multi_indexstruct<index_int,1> begin(1),end(1);
  REQUIRE_NOTHROW( begin = segment.begin() );
  CHECK( (*begin).at(0)==f );
  REQUIRE_NOTHROW( end = segment.end() );
  CHECK( (*end).at(0)==l+1 );
  CHECK( segment.get_dimensionality()==dim );
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
  CHECK( shifted.get_dimensionality()==3 );
  CHECK( shifted[0]==12 );
  CHECK( shifted[1]==22 );
  CHECK( shifted[2]==32 );

  REQUIRE_NOTHROW( multiplied = coord*2 );
  CHECK( multiplied.get_dimensionality()==3 );
  CHECK( multiplied[0]==20 );
  CHECK( multiplied[1]==40 );
  CHECK( multiplied[2]==60 );

  REQUIRE_NOTHROW( divided = coord/2 );
  CHECK( divided.get_dimensionality()==3 );
  CHECK( divided[0]==05 );
  CHECK( divided[1]==10 );
  CHECK( divided[2]==15 );

  REQUIRE_NOTHROW( shifted=domain_coordinate(multiplied+divided));
  CHECK( shifted.get_dimensionality()==3 );
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
  brick->set_component
    ( 0,shared_ptr<indexstruct<index_int,1>>(  new contiguous_indexstruct<index_int,1>(10,20) ) );
  brick->set_component
    ( 1,shared_ptr<indexstruct<index_int,1>>(  new strided_indexstruct<index_int,1>(25,50,5) ) );
  brick->set_component
    ( 2,shared_ptr<indexstruct<index_int,1>>(  new strided_indexstruct<index_int,1>(5,7,2) ) );

  // constructing a beta; see signature_function
  multi_shift_operator mshift( vector<index_int>{1,2,3} );
  REQUIRE_NOTHROW( sh_brick = mshift.operate(brick) );
  CHECK( sh_brick->first_index_r()==domain_coordinate( vector<index_int>{11,27,8} ) );
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
      if (beta_op->is_modulo_op() || truncation==nullptr) {
	beta_struct = gamma_struct->operate(beta_op);
      } else {
	beta_struct = gamma_struct->operate(beta_op,truncation);
      }
      // print("{} Struc: {} op: {} gives {}\n",
      // 		 pcoord->as_string(),
      //	 gamma_struct->as_string(),beta_op->as_string(),beta_struct->as_string());
      if (!beta_struct->is_empty()) {
	halo_struct = halo_struct->struct_union(beta_struct);
	//print("... union={}\n",halo_struct->as_string());
      }
    }
  }

  memory_buffer w;
  format_to(w.end(),"Beta struct from {} by applying:",gamma_struct->as_string());
  for ( auto o : ops ) format_to(w.end()," {},",o->as_string());
  format_to(w.end()," is: {}",halo_struct->as_string());
  INFO(w.data());

  CHECK( !halo_struct->is_empty());
  if (ipath==1 || ipath==4 || ipath==5) {
    CHECK( halo_struct->volume()==gamma_struct->volume() );
  } else if (ipath==2 || ipath==3) {
    CHECK( halo_struct->volume()>gamma_struct->volume() );
  } else if (ipath==6) {
    //    CHECK_THROWS( halo_struct->volume()==gamma_struct->volume() );
    CHECK( halo_struct->is_multi() );
    CHECK( halo_struct->multi_size()==2 );
    CHECK( halo_struct->volume()>gamma_struct->volume() );
  }
}

TEST_CASE( "multi-dimensional derived types","[multi][139]" ) {
  domain_coordinate
    first( vector<index_int>{10,20,30} ),
    last( vector<index_int>{19,29,39} );
  CHECK( first.get_dimensionality()==3 );
  CHECK( first[0]==10 );
  CHECK( first[1]==20 );
  CHECK( first[2]==30 );
  multi_indexstruct<index_int,1> *brick;
  REQUIRE_NOTHROW( brick = new contiguous_multi_indexstruct<index_int,1>(first,last) );
  CHECK( brick->volume()==1000 );
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
		     (dim,[] (const domain_coordinate &in) -> domain_coordinate {
		       int dim = in.get_dimensionality();
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
	(dim,[] (const domain_coordinate &c) -> shared_ptr<multi_indexstruct<index_int,1>> {
	  int dim = c.get_dimensionality();
	  //print("Apply multi sigma in d={}\n",dim);
	  auto m = shared_ptr<multi_indexstruct<index_int,1>>( new multi_indexstruct<index_int,1>(dim) );
	  for (int id=0; id<dim; id++) {
	    auto shift_struct = shared_ptr<indexstruct<index_int,1>>
	      (  new contiguous_indexstruct<index_int,1>(c.coord(id)+2) );
	    //print("coord-to-struct dim {} gives: {}\n",id,shift_struct->as_string());
	    m->set_component(id,shift_struct);
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
  // print("block multi has size {}\n",block->multi.size());
  INFO( format("input block: {}\n",block->as_string()) );
  REQUIRE_NOTHROW( blocked = op.operate(block) );
  // REQUIRE_NOTHROW( firster = blocked->first_index() );
  // REQUIRE_NOTHROW( laster = blocked->last_index() );
  // for (int id=0; id<dim; id++) {
  //   CHECK( firster->coord(id)==first->coord(id)+2 );
  //   CHECK( laster->coord(id)==last->coord(id)+2 );
  // }
}

#endif
