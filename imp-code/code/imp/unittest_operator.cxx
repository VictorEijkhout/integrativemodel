TEST_CASE( "Elementary ioperator stuff","[operate][01]" ) {
  ioperator iop;
  print("VLE test [01] should throw. Why doesn't it?\n");
  //CHECK_THROWS( iop = ioperator("no_such_thing") );
}

TEST_CASE( "Test ioperator right workings modulo","[operate][02][modulo]" ) {
  auto i1 = ioperator(">>1");
  CHECK( i1.is_shift_op() );
  CHECK( i1.is_modulo_op() );
  CHECK( i1.amount()==1 );
  CHECK( i1.operate(0)==1 );
  CHECK_NOTHROW( i1.operate(-1) );
  CHECK( i1.operate(5)==6 );
  CHECK( i1.is_shift_op() );
  CHECK( i1.amount()==1 );
  CHECK( i1.is_modulo_op() );
  CHECK( i1.operate(5,6)==6 );
  CHECK( i1.operate(6,6)==0 );
}

TEST_CASE( "Test ioperator right workings","[operate][03]" ) {
  auto i2 = ioperator(">=1");
  INFO( format("testing iop: {}",i2.as_string()) );
  CHECK( !i2.is_modulo_op() );
  CHECK( i2.operate(0)==1 );
  CHECK_NOTHROW( i2.operate(-1) );
  CHECK( i2.operate(15)==16 );
  CHECK( i2.operate(15,15)==15 );
}

TEST_CASE( "Test ioperator left workings modulo","[operate][04][modulo]" ) {
  auto i1 = ioperator("<<1");
  CHECK( i1.is_modulo_op() );
  CHECK( i1.operate(1)==0 );
  CHECK_NOTHROW( i1.operate(-1) );
  //  CHECK_THROWS( i1.operate(0) );
  CHECK( i1.operate(6)==5 );
  CHECK( i1.operate(0,6)==6 );
}

TEST_CASE( "Test ioperator left workings bump","[operate][05]" ) {
  auto i2 = ioperator("<=1");
  CHECK( !i2.is_modulo_op() );
  CHECK( i2.operate(0)==-1 );
  CHECK_NOTHROW( i2.operate(-1) );
  CHECK( i2.operate(16)==15 );
  CHECK( i2.operate(15,16)==14 );
}

TEST_CASE( "Test ioperator shift workings modulo","[operate][06][modulo]" ) {
  auto i1 = ioperator(">>1");
  CHECK( i1.is_modulo_op() );
  CHECK( i1.inverse_operate(1)==0 );
  CHECK_THROWS( i1.inverse_operate(-1) );
  CHECK( i1.inverse_operate(6)==5 );
  CHECK_THROWS( i1.inverse_operate(0) );
  CHECK( i1.inverse_operate(0,6)==5 );

  auto i3 = ioperator("<<1");
  CHECK( i3.is_modulo_op() );
  CHECK( i3.operate(1)==0 );
  CHECK_NOTHROW( i3.operate(-1) );
  CHECK_NOTHROW( i3.operate(0) );
  CHECK( i3.operate(6)==5 );
  CHECK( i3.operate(0,6)==6 );

}

TEST_CASE( "Test ioperator shift workings","[operate][07]" ) {
  auto i2 = ioperator(">=1");
  CHECK( !i2.is_modulo_op() );
  CHECK( i2.inverse_operate(1)==0 );
  CHECK_THROWS( i2.inverse_operate(0) );
  CHECK_THROWS( i2.inverse_operate(-1) );
  CHECK( i2.inverse_operate(16)==15 );
  CHECK( i2.inverse_operate(15,16)==14 );

  auto i4 = ioperator("<=1");
  CHECK( !i4.is_modulo_op() );
  CHECK( i4.operate(0)==-1 );
  CHECK( i4.operate(0,5)==0 );
  CHECK_NOTHROW( i4.operate(-1) );
  CHECK( i4.operate(16)==15 );
  CHECK( i4.operate(15,16)==14 );
}

TEST_CASE( "Index structure operate","[index][operate][modulo][10]" ) {

  shared_ptr<indexstruct> i1 = shared_ptr<indexstruct>{ new contiguous_indexstruct(3,7) },
    i2 = shared_ptr<indexstruct>{ new contiguous_indexstruct(0,3) };
  shared_ptr<indexstruct> ii;
  auto
    r1 = ioperator(">>1"),
    m1 = ioperator(">=1"),
    l2 = ioperator("<<2"),
    m3 = ioperator("<=3");

  SECTION( "simple shifts" ) {
    ii = i1->operate(r1);
    CHECK( ii->first_index()==4 ); 
    CHECK( ii->last_index()==8 );
    ii = i1->operate(m1);
    CHECK( ii->first_index()==4 ); 
    CHECK( ii->last_index()==8 );
    ii = i1->operate(l2);
    CHECK( ii->first_index()==1 ); 
    CHECK( ii->last_index()==5 );
    ii = i2->operate(l2); // we don't complain about negative indices
    CHECK( ii->first_index()==-2 ); 
    CHECK( ii->last_index()==1 );
  }

  SECTION( "multiply" ) {
    auto t4 = ioperator("*4");
    ii = i1->operate(t4);
    CHECK( ii->first_index()==12 );
    CHECK( ii->last_index()==28 );
    CHECK( ii->stride()==4 );
    CHECK( ii->stride()==4*i1->stride() );
 }

  SECTION( "divide" ) {
    auto d2 = ioperator("/2");
    ii = i1->operate(d2);
    CHECK( ii->first_index()==1 );
    CHECK( ii->last_index()==3 );
  }

  SECTION( "big shift" ) {
    auto ip = shift_operator( 500 );
    CHECK( ip.is_shift_op() );
    CHECK( ip.amount()==500 );
    CHECK( ip.operate(12)==512 );
    ii = i1->operate( ip );
    CHECK( ii->first_index()==503 );
    CHECK( ii->last_index()==507 );
    auto im = shift_operator( -500 );
    CHECK( im.is_shift_op() );
    CHECK( im.amount()==-500 );
    CHECK( im.operate(10)==-490 );
    ii = i1->operate( im );
    CHECK( ii->first_index()==-497 );
    CHECK( ii->last_index()==-493 );
  }

  // SECTION( "base multiply" ) {
  //   ioperator *t4 = new ioperator("x4");
  //   ii = i1->operate(t4);
  //   CHECK( ii->first_index()==12 );
  //   CHECK( ii->last_index()==16 );
  //   CHECK( ii->local_size()==i1->local_size() );
  //   CHECK( ii->stride()==i1->stride() );
  // }
}

TEST_CASE( "Index structure intersect and union","[index][intersect][11]" ) {
  auto 
    i1 = shared_ptr<indexstruct>( new contiguous_indexstruct(0,5) ),
    i2 = shared_ptr<indexstruct>( new contiguous_indexstruct(3,6) ),
    i3 = shared_ptr<indexstruct>( new contiguous_indexstruct(3,4) );
  auto i4 = i1->intersect(i2);
  CHECK( i4!=nullptr );
  CHECK( i4->first_index()==3 );
  CHECK( i4->last_index()==5 );
  auto i5 = i1->intersect(i3);
  CHECK( i5!=nullptr );
  CHECK( i5->first_index()==3 );
  CHECK( i5->last_index()==4 );

  shared_ptr<indexstruct> isect;
  shared_ptr<indexstruct> ii;
  SECTION( "empty result" ) {
    ii = shared_ptr<indexstruct>{ new contiguous_indexstruct(6,7) };
    REQUIRE_NOTHROW( isect = i1->intersect(ii) );
    CHECK( isect->is_empty() );
  }
  SECTION( "intersect with contained indexed" ) {
    int len = 3;
    index_int *idx = new index_int[len];
    for (int i=0; i<len; i++) 
      idx[i] = 1+i;
    ii = shared_ptr<indexstruct>{ new indexed_indexstruct(len,idx) };
    REQUIRE_NOTHROW( isect = i1->intersect(ii) );
    CHECK( isect->local_size()==len );
    CHECK( isect->is_indexed() );
  }

  SECTION( "intersect with non contained indexed" ) {
    int len = 3;
    index_int *idx = new index_int[len];
    for (int i=0; i<len; i++) 
      idx[i] = 4+i; // 4,5 in; 6 not
    ii = shared_ptr<indexstruct>{ new indexed_indexstruct(len,idx) }; // remember i1 = 0--5
    SECTION( "one way intersect" ) {
      REQUIRE_NOTHROW( isect=i1->intersect(ii) );
      CHECK( isect->local_size()==2 );
      CHECK( isect->is_indexed() );
    }
    SECTION( "other way around intersect" ) {
      REQUIRE_NOTHROW( isect=ii->intersect(i1) );
      CHECK( isect->local_size()==2 );
      CHECK( isect->is_indexed() );
    }
  }
  SECTION( "intersection of two indexeds" ) {
    int len = 5;
    index_int *idx = new index_int[len]; // 0,2,4,6,8
    index_int *isx = new index_int[len]; // 1,4,7,10,13
    for (int i=0; i<len; i++) {
      idx[i] = 2*i; isx[i] = 1+3*i; }
    ii = shared_ptr<indexstruct>{ new indexed_indexstruct(len,idx) };
    REQUIRE_NOTHROW( isect=ii->intersect
		     ( shared_ptr<indexstruct>{ new indexed_indexstruct(len,isx) } ) );
    CHECK( isect->local_size()==1 );
    CHECK( isect->is_indexed() );
    CHECK( isect->first_index()==4 );
  }

  auto i8 = shared_ptr<indexstruct>{ new contiguous_indexstruct(5,6) };
  CHECK_NOTHROW( i8 = i8->struct_union( new contiguous_indexstruct(6,8) ) );
  CHECK( i8->first_index()==5 );
  CHECK( i8->last_index()==8 );
  CHECK_NOTHROW( i8 = i8->struct_union( new contiguous_indexstruct(7,8) ) );
  CHECK( i8->first_index()==5 );
  CHECK( i8->last_index()==8 );

  auto i9 = shared_ptr<indexstruct>{ new strided_indexstruct(5,10,2) }; // converted to 5,9,2
  // SECTION( "step compatibility" ) {
  //   CHECK_THROWS( i9 = i9->union_with( new contiguous_indexstruct(7,8) ) );
  // }
  SECTION( "this works" ) {
    CHECK_NOTHROW( i9 = i9->struct_union( new strided_indexstruct(7,9,2) ) );
    CHECK( i9->first_index()==5 );
    CHECK( i9->last_index()==9 );
    CHECK( i9->stride()==2 );
  }

  SECTION( "make relative to other struct" ) {
    shared_ptr<indexstruct> i10,i11,i12;
    CHECK_NOTHROW( i10 = shared_ptr<indexstruct>{ new contiguous_indexstruct(2,12) } );
    CHECK_NOTHROW( i11 = shared_ptr<indexstruct>{ new contiguous_indexstruct(2,5) } );
    CHECK_NOTHROW( i12 = shared_ptr<indexstruct>{ new contiguous_indexstruct(3,8) } );
    print("anohther non-throwing throw\n");
    //    CHECK_THROWS( ii = i10->relativize_to(i11,true) );
    CHECK_NOTHROW( ii = i11->relativize_to(i10) );
    CHECK( ii->first_index()==0 );
    CHECK( ii->last_index()==3 );
    CHECK_NOTHROW( ii = i12->relativize_to(i10) );
    CHECK( ii->first_index()==1 );
    CHECK( ii->last_index()==6 );
  }

  SECTION( "union with incompatible strides or non-contiguous" ) {
    auto i21 = shared_ptr<indexstruct>( new strided_indexstruct(0,10,2) );
    SECTION( "overlap" ) {
      auto i22 = shared_ptr<indexstruct>( new strided_indexstruct(10,20,2) );
      CHECK_NOTHROW( ii = i21->struct_union( i22 ) );
      CHECK( ii->first_index()==0 );
      CHECK( ii->last_index()==20 );
    }
    SECTION( "wrong base offset" ) {
      auto i23 = shared_ptr<indexstruct>( new strided_indexstruct(11,20,2) );
      // VLE doesn't throw      CHECK_THROWS( ii = i21->struct_union( i23 ) );
    }
    SECTION( "wrong stride" ) {
      auto i24 = shared_ptr<indexstruct>( new strided_indexstruct(10,20,3) );
      // VLE doesn't throw      CHECK_THROWS( ii = i21->struct_union( i24 ) );
    }
    SECTION( "the right gap" ) {
      auto i25 = shared_ptr<indexstruct>( new strided_indexstruct(12,20,2) );
      // VLE doesn't throw CHECK_THROWS( ii = i21->struct_union( i25 ) );
    }
  }
}

TEST_CASE( "copy indexstruct","[indexstruct][copy][12]" ) {
  shared_ptr<indexstruct> i1,i2;

  i1 = shared_ptr<indexstruct>{ new contiguous_indexstruct(0,10) };
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

TEST_CASE( "big shift operators","[operator][shift][13]" ) {
  ioperator i;
  i = ioperator("shift",5);
  CHECK( i.operate(6)==11 );
  CHECK_NOTHROW( i.operate(-11)==-6 );
  i = ioperator("shift",-3);
  CHECK( i.operate(6)==3 );
  CHECK( i.operate(1)==-2 );
  CHECK_NOTHROW( i.operate(-6)==-2 );  
}

TEST_CASE( "arbitrary shift of indexstruct","[indexstruct][shift][14]" ) {
  shared_ptr<indexstruct> i1;
  shared_ptr<indexstruct> i2;
  i1 = shared_ptr<indexstruct>{ new contiguous_indexstruct(5,7) };

  SECTION( "shift by" ) {
    REQUIRE_NOTHROW( i2 = i1->operate( ioperator("shift",7)) );
    CHECK( i2->first_index()==12 );
    CHECK( i2->last_index()==14 );
  }
  SECTION( "shift to" ) {
    REQUIRE_NOTHROW( i2 = i1->operate( ioperator("shiftto",7) ) );
    CHECK( i2->first_index()==7 );
    CHECK( i2->last_index()==9 );
  }

}

