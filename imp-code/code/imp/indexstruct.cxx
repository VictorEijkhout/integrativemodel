/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2022
 ****
 **** indexstruct and ioperator package implementation
 ****
 ****************************************************************/

#include "imp_coord.h"
#include "indexstruct.hpp"

#include <sstream>
using std::cout, std::stringstream;

#include "fmt/format.h"
using fmt::format, fmt::print,fmt::memory_buffer, fmt::format_to, fmt::to_string;

using std::function;
using std::move;
using std::make_shared, std::shared_ptr;

using std::string;
using std::vector;

#include <tuple>
using std::pair;

/****************
 ****************
 **************** structs
 ****************
 ****************/

/*! Volume of bounding box,
 * this equals volume for contiguous,
 * but is more for strided and such
 */
template<typename I,int d>
I indexstruct<I,d>::outer_volume() const {
  auto outer_vector = last_index()-first_index();
  return outer_vector.span();
};

template<typename I,int d>
int indexstruct<I,d>::type_as_int() const {
  if (is_empty()) return 1;
  else if (is_contiguous()) return 2;
  else if (is_strided()) return 3;
  else if (is_indexed()) return 4;
  else if (is_composite()) return 5;
  else
    throw(format("Type can not be converted to int"));
};

template<typename I,int d>
void indexstruct<I,d>::report_unimplemented( string s ) const {
  string e = fmt::format("Trying to use query <<{}>> on undefined indexstruct",s);
  cout << e+"\n";
  throw(e);
};

template<typename I,int d>
std::shared_ptr<indexstruct<I,d>> indexstruct<I,d>::make_strided(bool trace) const {
  report_unimplemented("make_strided");
  return nullptr;
};

template<typename I,int d>
shared_ptr<indexstruct<I,d>> indexstruct<I,d>::add_element( coordinate<I,d> idx ) const {
  report_unimplemented("add_element");
  return shared_ptr<indexstruct<I,d>> ( make_shared<empty_indexstruct<I,d>>() );
};

template<typename I,int d>
void indexstruct<I,d>::add_in_element( coordinate<I,d> idx ) {
  report_unimplemented("add_in_element");
};

template<typename I,int d>
bool indexstruct<I,d>::equals( const std::shared_ptr<indexstruct<I,d>>& idx ) const {
  throw(fmt::format("Equals not implemented for <<{}>> and <<{}>>",
		    type_as_string(),idx->type_as_string()));
};

//! Operate, then cut to let lo/hi be the lowest/highest index, inclusive.
template<typename I,int d>
shared_ptr<indexstruct<I,d>> indexstruct<I,d>::operate
    ( const ioperator<I,d> &op, coordinate<I,d> lo,coordinate<I,d> hi ) const {
  shared_ptr<indexstruct<I,d>>
    noleft = this->operate(op)->truncate_left(lo),
    noright = noleft->truncate_right(hi);
  return noright;
};

template<typename I,int d>
bool indexstruct<I,d>::contains_element_in_range( const coordinate<I,d>& idx) const {
  return first_index()<=idx && idx<=last_actual_index();
};

//! Base test for disjointness; derived classes will build on this.
template<typename I,int d>
bool indexstruct<I,d>::disjoint( shared_ptr<indexstruct<I,d>> idx ) {
  return first_index()>idx->last_actual_index() || last_index()<idx->first_index();
};

//! Operate and truncate to boundary coordinates
template<typename I,int d>
shared_ptr<indexstruct<I,d>> indexstruct<I,d>::operate
        ( const sigma_operator<I,d> &op, coordinate<I,d> lo,coordinate<I,d> hi ) const {
  shared_ptr<indexstruct<I,d>>
    noleft = this->operate(op)->truncate_left(lo),
    noright = noleft->truncate_right(hi);
  return noright;
};

//! Operate with `ioperator', then cut boundaries to fit within `outer'.
template<typename I,int d>
shared_ptr<indexstruct<I,d>> indexstruct<I,d>::operate
    ( const ioperator<I,d> &op,shared_ptr<indexstruct<I,d>> outer ) const {
  auto lo = outer->first_index(), hi = outer->last_index();
  return this->operate(op,lo,hi);
};

//! Operate with `sigma_operator', then cut boundaries to fit within `outer'.
template<typename I,int d>
shared_ptr<indexstruct<I,d>> indexstruct<I,d>::operate
    ( const sigma_operator<I,d> &op,shared_ptr<indexstruct<I,d>> outer ) const {
  auto lo = outer->first_index(), hi = outer->last_index();
  return this->operate(op,lo,hi);
};

//! Let `trunc' be the first index in the truncated struct.
template<typename I,int d>
shared_ptr<indexstruct<I,d>> indexstruct<I,d>::truncate_left( coordinate<I,d> trunc ) {
  if (trunc<=first_index()) {
    return this->make_clone();
    //return shared_ptr<indexstruct<I,d>>( this->make_clone() ); //shared_from_this();
  } else {
    auto truncated = this->minus
      ( shared_ptr<indexstruct<I,d>>
	( new contiguous_indexstruct<I,d>(this->first_index(),trunc-1) ) );
    return truncated;
  }
};

//! Let `trunc' be the last index in the truncated struct.
template<typename I,int d>
shared_ptr<indexstruct<I,d>> indexstruct<I,d>::truncate_right( coordinate<I,d> trunc ) {
  if (last_index()<=trunc) {
    return this->make_clone(); //shared_from_this();
    //return shared_ptr<indexstruct<I,d>>( this->make_clone() ); //shared_from_this();
  } else {
    auto truncated = this->minus
      ( shared_ptr<indexstruct<I,d>>
	( new contiguous_indexstruct<I,d>(trunc+1,this->last_index()) ) );
    return truncated;
  }
};

/****
 **** Empty indexstruct
 ****/
template<typename I,int d>
shared_ptr<indexstruct<I,d>> empty_indexstruct<I,d>::add_element( coordinate<I,d> idx ) const {
  return shared_ptr<indexstruct<I,d>>
    ( make_shared<indexed_indexstruct<I,d>>(vector<coordinate<I,d>>{idx}) );
};

template<typename I,int d>
void empty_indexstruct<I,d>::add_in_element( coordinate<I,d> idx ) {
  throw("can not add_in to empty");
};

template<typename I,int d>
shared_ptr<indexstruct<I,d>> empty_indexstruct<I,d>::struct_union
        ( shared_ptr<indexstruct<I,d>> idx,bool trace ) {
  if (trace) print("union empty returns other\n");
  return idx;
};

template<typename I,int d>
std::string contiguous_indexstruct<I,d>::as_string() const {
  if (d==1)
    return fmt::format("contiguous: [{}--{}]",this->first[0],this->last[0]);
  else if (d==2)
    return fmt::format("contiguous: [{}--{}] x [{}--{}]",
		       this->first[0],this->last[0],
		       this->first[1],this->last[1]
		       );
  else
    return fmt::format("contiguous {}D: [{}--{}]",
		       d,this->first[0],this->last[0]);
};

/****
 **** Strided indexstruct
 ****/

/*
 * Constructors
 */
//! Constructor from arrays, delegates by making coordinates from them.
template<typename I,int d>
strided_indexstruct<I,d>::strided_indexstruct
    (const std::array<I,d> f,const std::array<I,d>  l,I s)
      : strided_indexstruct<I,d>
      ( coordinate<I,d>(f), coordinate<I,d>(l), constant_coordinate<I,d>(s) ) {
};

/*!
  Base constructor, first copies first and last,
  then shrinks last to make sure it is precisely included in the stride
*/
template<typename I,int d>
strided_indexstruct<I,d>::strided_indexstruct
    (const coordinate<I,d>& f,const coordinate<I,d>& l,const coordinate<I,d>& s)
      : first(f),last(l),stride_amount(s) {
  for (int id=0; id<d; id++)  // make sure last is actually included
    if (first[id]!=last[id])
      last[id] -= (last[id]-first[id])%stride_amount[id];
};

/*
 * stuff
 */
/*! Volume of the bounding box
 * For stride>1 this is one short of the next strided point
 */
template<typename I,int d>
I strided_indexstruct<I,d>::outer_volume()  const {
  auto outer_vector = (last_index()-first_index()+stride_amount-1)+1;
  return outer_vector.span();
};

//! Number of points contained in this strided structure.
template<typename I,int d>
I strided_indexstruct<I,d>::volume()  const {
  auto outer_vector =
    ( /* one before next strided: */ last_actual_index()+stride_amount-1
      - first_index() + 1 )
    /stride_amount;
  return outer_vector.span();
};

template<typename I,int d>
bool strided_indexstruct<I,d>::equals( const shared_ptr<indexstruct<I,d>>& idx ) const {
  if (idx->is_empty())
    return false;
  strided_indexstruct *strided = dynamic_cast<strided_indexstruct*>(idx.get());
  if (strided!=nullptr)
    return first==strided->first_index() && last==strided->last_actual_index()
      && stride_amount==strided->stride();
  else {
    auto simple = idx->force_simplify();
    if (simple->is_strided())
      return equals(simple);
    else return false;
  }
};

// template<typename I,int d>
// void strided_indexstruct<I,d>::add_in_element( coordinate<I,d> idx ) {
//   auto duplicate = this->make_clone();
//   return duplicate->add_element(idx);
// };

template<typename I,int d>
shared_ptr<indexstruct<I,d>> strided_indexstruct<I,d>::add_element( coordinate<I,d> idx ) const {
  if (contains_element(idx))
    return this->make_clone();
  else if (idx==first-stride_amount) {
    if (stride_amount==1)
      return shared_ptr<indexstruct<I,d>>
	( make_shared<contiguous_indexstruct<I,d>>(idx,last) );
    else
      return shared_ptr<indexstruct<I,d>>
	( make_shared<strided_indexstruct<I,d>>(idx,last,stride_amount) );
  } else if (idx==last+stride_amount) {
    if (stride_amount==1)
      return shared_ptr<indexstruct<I,d>>
	( make_shared<contiguous_indexstruct<I,d>>(first,idx) );
    else
      return shared_ptr<indexstruct<I,d>>
	( make_shared<strided_indexstruct<I,d>>(first,idx,stride_amount) );
  } else {
    auto indexed = make_shared<indexed_indexstruct<I,d>>();
    if (d==1) {
      for ( auto v=first_index(); v!=last_index(); v=v+stride() )
	indexed->add_in_element(v);
    } else throw("Can not convert stride to index in >1D");
    return indexed;
  }
};

template<typename I,int d>
bool strided_indexstruct<I,d>::contains_element( const coordinate<I,d>& idx ) const {
  return first<=idx && idx<=last && (idx-first)%stride_amount==0;
};

template<typename I,int d>
I strided_indexstruct<I,d>::find( coordinate<I,d> idx ) const {
  if (!contains_element(idx))
    throw(format("Index {} to find is out of range <<{}>>",
		 idx[0],this->as_string()));
  auto locvec = (idx-first)/stride_amount;
  return locvec[0];
};

template<typename I,int d>
coordinate<I,d> strided_indexstruct<I,d>::get_ith_element( const I i ) const {
  if (i<0 || i>=volume())
    throw(fmt::format("Index {} out of range for {}",i,as_string()));
  return first+stride_amount*i;
};

template<typename I,int d>
bool strided_indexstruct<I,d>::can_merge_with_type
        ( std::shared_ptr<indexstruct<I,d>> idx) {
  strided_indexstruct *strided = dynamic_cast<strided_indexstruct*>(idx.get());
  if (strided!=nullptr) { // strided & strided
    return stride_amount==strided->stride_amount &&
      first%stride_amount==strided->first%strided->stride_amount &&
      strided->first<=last+stride_amount && first-stride_amount<=strided->last;
  } else return 0;
};

template<typename I,int d>
bool strided_indexstruct<I,d>::contains( const shared_ptr<indexstruct<I,d>>& idx ) const {
  if (idx->volume()==0) return true;
  if (this->volume()==0) return false;
  if (idx->volume()==1)
    return contains_element( idx->first_index() );

  /*
   * Case : contains other strided
   */
  strided_indexstruct<I,d>* strided = dynamic_cast<strided_indexstruct<I,d>*>(idx.get());
  if (strided!=nullptr)
    return first<=strided->first && strided->last<=last
      && (strided->first-first)%stride_amount==0 && strided->stride_amount%stride_amount==0;

  /*
   * Case: contains indexed
   */
  indexed_indexstruct<I,d>* indexed = dynamic_cast<indexed_indexstruct<I,d>*>(idx.get());
  if (indexed!=nullptr) { // strided & indexed
    //auto nonconst = make_clone();
    return contains_element(indexed->first_index())
      && contains_element(indexed->last_index());
  }

  /*
   * Case: contains composite
   */
  composite_indexstruct<I,d>* composite = dynamic_cast<composite_indexstruct<I,d>*>(idx.get());
  if (composite!=nullptr) { // strided & composite
    for ( auto s : composite->get_structs() )
      if (!contains(s))
	return false;
    return true;
  }

  throw(std::string("unimplemented strided contains case"));
};

template<typename I,int d>
std::string strided_indexstruct<I,d>::as_string() const {
  return fmt::format("strided: [{}-by{}--{}]",
		     first.as_string(),stride_amount.as_string(),last.as_string());
};


/*! Disjointness test for strided.
  \todo bunch of unimplemented cases \todo unittest for indexed csae
*/
//bool strided_indexstruct<I,d>::disjoint( indexstruct<I,d>* idx ) {
template<typename I,int d>
bool strided_indexstruct<I,d>::disjoint( shared_ptr<indexstruct<I,d>> idx ) {
  bool range_disjoint = indexstruct<I,d>::disjoint(idx);
  if (range_disjoint==true)
    return true;
  else {
    // disjoint with other strided
    strided_indexstruct<I,d>* strided = dynamic_cast<strided_indexstruct<I,d>*>(idx.get());
    if (strided!=nullptr) {
      if (stride()==idx->stride())
	return first_index()%stride()!=idx->first_index()%stride();
      else
	throw(fmt::format("unimplemented disjoint for different strides"));
    }
    // disjoint from indexed
    indexed_indexstruct<I,d>* indexed = dynamic_cast<indexed_indexstruct<I,d>*>(idx.get());
    if (indexed!=nullptr) {
      for ( auto i : *indexed ) {
	if (contains_element(i))
	  return false;
      }
    }
    // other cases not covered yet
    throw(fmt::format("unimplemented disjoint: strided & {}",idx->type_as_string()));
  }
};

template<typename I,int d>
bool strided_indexstruct<I,d>::has_intersect( shared_ptr<indexstruct<I,d>> idx ) {
  if (idx->is_empty())
    return false;
  if (contains(idx))
    return true;
  if (idx->contains(this->shared_from_this()))
    return true; 

  /*
   * Case : intersect with other strided
   */
  strided_indexstruct<I,d>* strided = dynamic_cast<strided_indexstruct*>(idx.get());
  if (strided!=nullptr) { // strided & strided
    if (stride_amount==strided->stride_amount) {
      if (first%stride_amount!=strided->first%strided->stride_amount) {
	return false;
      } else {
	auto
	  mn = coordmax<I,d>(first,strided->first),
	  mx = coordmin<I,d>(last,strided->last);
	return not (mx<mn);
      }
    } else if (idx->volume()<volume()) { // case of different strides, `this' s/b small
      return idx->has_intersect(this->shared_from_this());
    } else { // case of different strides
      return true;
    }
  }
  /*
   * Case: intersect with indexed => reverse
   */
  indexed_indexstruct<I,d>* indexed = dynamic_cast<indexed_indexstruct<I,d>*>(idx.get());
  if (indexed!=nullptr) { // strided & indexed
    return indexed->has_intersect(this->shared_from_this());
  }
  throw(fmt::format("Unimplemented has_intersect <<{}>> and <<{}>>",
		    this->as_string(),idx->as_string()));
};

template<typename I,int d>
shared_ptr<indexstruct<I,d>> strided_indexstruct<I,d>::translate_by( coordinate<I,d> shift ) const {
  return shared_ptr<indexstruct<I,d>>{
    new strided_indexstruct<I,d>(first+shift,last+shift,stride_amount) };
};

template<typename I,int d>
shared_ptr<indexstruct<I,d>> strided_indexstruct<I,d>::intersect
    ( shared_ptr<indexstruct<I,d>> idx ) {
  if (idx->is_empty())
    return shared_ptr<indexstruct<I,d>>( make_shared<empty_indexstruct<I,d>>() );
  if (contains(idx))
    return idx->make_clone();
  if (idx->contains(this->shared_from_this()))
    return this->make_clone();
  if (idx->last_index()<first_index() or last_index()<idx->first_index() )
    return shared_ptr<indexstruct<I,d>>( make_shared<empty_indexstruct<I,d>>() );

  /*
   * Case : intersect with other strided
   */
  strided_indexstruct<I,d>* strided = dynamic_cast<strided_indexstruct<I,d>*>(idx.get());
  if (strided!=nullptr) { // strided & strided
    if (stride_amount==strided->stride_amount) {
      if (first%stride_amount!=strided->first%strided->stride_amount) {
	return shared_ptr<indexstruct<I,d>>( make_shared<empty_indexstruct<I,d>>() );
      } else {
	auto
	  mn = coordmax<I,d>(first,strided->first),
	  mx = coordmin<I,d>(last,strided->last);
	if (mx<mn)
	  return shared_ptr<indexstruct<I,d>>( make_shared<empty_indexstruct<I,d>>() );
	else if (stride_amount==1)
	  return shared_ptr<indexstruct<I,d>>( make_shared<contiguous_indexstruct<I,d>>(mn,mx) );
	else
	  return shared_ptr<indexstruct<I,d>>
	    ( make_shared<strided_indexstruct<I,d>>(mn,mx,stride_amount) );
      }
    } else if (idx->volume()<volume()) { // case of different strides, `this' s/b small
      return idx->intersect(this->shared_from_this());
    } else { // case of different strides
      auto rstruct = shared_ptr<indexstruct<I,d>>( make_shared<empty_indexstruct<I,d>>() );
      for ( auto i : *this ) {
	if (idx->contains_element(i))
	  rstruct = rstruct->add_element(i);
      }
      return rstruct->force_simplify();
      //throw(std::string("Unimplemented stride-stride intersection"));
    }
  }

  /*
   * Case: intersect with indexed => reverse
   */
  indexed_indexstruct<I,d>* indexed = dynamic_cast<indexed_indexstruct<I,d>*>(idx.get());
  if (indexed!=nullptr) { // strided & indexed
    return indexed->make_clone()->intersect(this->shared_from_this());
  }

  /*
   * Case: intersect with composite => reverse
   */
  composite_indexstruct<I,d>* composite = dynamic_cast<composite_indexstruct<I,d>*>(idx.get());
  if (composite!=nullptr) { // strided & composite
    return composite->intersect(this->shared_from_this());
  }

  throw(fmt::format("Unimplemented intersect <<{}>> and <<{}>>",
		    this->as_string(),idx->as_string()));
};

//! \todo make unittesting of minus composite
//! \todo weird clone because of a const problem.
template<typename I,int d>
shared_ptr<indexstruct<I,d>> strided_indexstruct<I,d>::minus
    ( shared_ptr<indexstruct<I,d>> idx ) const {
  // easy cases
  if (idx->is_empty())
    return this->make_clone();
  //return shared_ptr<indexstruct<I,d>>( this->make_clone());
  if (idx->contains(this->make_clone())) // this->shared_from_this()))
    return shared_ptr<indexstruct<I,d>>( make_shared<empty_indexstruct<I,d>>() );

  /*
   * Case: the idx set is in the interior
   */
  if (this->first_index()<idx->first_index() and idx->last_index()<this->last_index()) {
    shared_ptr<indexstruct<I,d>> left,right,mid;
    auto midfirst = idx->first_index(), midlast = idx->last_index();
    if (stride_amount==1) {
      left = shared_ptr<indexstruct<I,d>>
	( make_shared<contiguous_indexstruct<I,d>>(this->first_index(),midfirst-1) );
      right = shared_ptr<indexstruct<I,d>>
	( make_shared<contiguous_indexstruct<I,d>>(midlast+1,this->last_index()) );
    } else {
      // left
      left = shared_ptr<indexstruct<I,d>>
	( make_shared<strided_indexstruct<I,d>>(this->first_index(),midfirst-1,stride_amount) );
      // right
      auto rfirst = midlast+1;
      while (rfirst%stride_amount!=last%stride_amount)
	rfirst = rfirst+1;
      right = shared_ptr<indexstruct<I,d>>
	( make_shared<strided_indexstruct<I,d>>(rfirst,this->last_index(),stride_amount) );
    }
    auto comp = left->struct_union(right);
    return comp;
  }

  /*
   * Case : minus contiguous. Easy.
   */
  contiguous_indexstruct<I,d>* contiguous = dynamic_cast<contiguous_indexstruct<I,d>*>(idx.get());
  if (contiguous!=nullptr) {
    auto ifirst = contiguous->first_index(),ilast = contiguous->last_index();
    shared_ptr<indexstruct<I,d>> contmin;
    if (ilast<first || ifirst>last) // disjoint 
      return this->make_clone(); //shared_from_this();
    else if (ifirst<=first) { // cut left part
      if (stride_amount==1)
	contmin = shared_ptr<indexstruct<I,d>>{
	  make_shared<contiguous_indexstruct<I,d>>( ilast+stride_amount,last ) };
      else
	contmin = shared_ptr<indexstruct<I,d>>{
	  make_shared<strided_indexstruct<I,d>>( ilast+stride_amount,last,stride_amount ) };
    } else { // cut right part
      if (stride_amount==1)
	contmin = shared_ptr<indexstruct<I,d>>{
	  make_shared<contiguous_indexstruct<I,d>>( first,ifirst-1/*stride_amount*/ ) };
      else
	contmin = shared_ptr<indexstruct<I,d>>{
	  make_shared<strided_indexstruct<I,d>>( first,ifirst-1/*stride_amount*/,stride_amount ) };
    }
    return contmin;
  }
  
  /*
   * Case : minus other strided
   */
  strided_indexstruct<I,d>* strided = dynamic_cast<strided_indexstruct<I,d>*>(idx.get());
  if (strided!=nullptr) { // strided & strided
    if (stride_amount==strided->stride_amount) {
      shared_ptr<indexstruct<I,d>> stridmin{nullptr};
      if (first%stride_amount!=strided->first%strided->stride_amount) {
	return this->make_clone(); // interleaving case
      } else {
	if (contains_element(idx->first_index()) && contains_element(idx->last_index()))
	  throw(std::string("should yield composite; unimplemented"));
	auto f = idx->first_index(), l = idx->last_index();
	if (f<=first) { // cut on the left
	  f = coordmax<I,d>(l+1,first); l = last;
	} else { // cut on the right
	  l = coordmin<I,d>(f-1,last); f = first;
	}
	if (stride_amount==1)
	  stridmin = shared_ptr<indexstruct<I,d>>{ make_shared<contiguous_indexstruct<I,d>>(f,l) };
	else
	  stridmin = shared_ptr<indexstruct<I,d>>{ make_shared<strided_indexstruct<I,d>>(f,l,stride_amount) };
      }
      return stridmin;
    } else {
      auto
	idxthis = this->convert_to_indexed(), idxother = idx->convert_to_indexed();
      auto stridmin =
	idxthis->minus(idxother);
      return stridmin;
    }
  }

  /*
   * Minus indexed: first convert itself to indexed
   */
  indexed_indexstruct<I,d>* indexed = dynamic_cast<indexed_indexstruct<I,d>*>(idx.get());
  if (indexed!=nullptr) { // strided & indexed
    auto
      this_index = this->convert_to_indexed();
    indexed_indexstruct<I,d>* test_index = dynamic_cast<indexed_indexstruct<I,d>*>(this_index.get());
    if (test_index==nullptr) // remove this test after a while
      throw(fmt::format("failed conversion of <<{}>> to indexed",this->as_string()));
    return this_index->minus(idx); //(shared_ptr<indexstruct<I,d>>(indexed));
  }

  /*
   * Minus composite
   */
  composite_indexstruct<I,d>* composite = dynamic_cast<composite_indexstruct<I,d>*>(idx.get());
  if (composite!=nullptr) { // strided & composite
    auto rstruct = make_clone();
    for ( auto s : composite->get_structs() ) {
      rstruct = rstruct->minus(s);
    }
    return rstruct;
  }

  throw(fmt::format("Unimplemented stride minus {}",idx->type_as_string()));
};

//snippet stridedrelativize
template<typename I,int d>
shared_ptr<indexstruct<I,d>> strided_indexstruct<I,d>::relativize_to
    ( shared_ptr<indexstruct<I,d>> idx,bool trace ) {
  if (trace)
    print("attempt to relativize {} to {}\n",this->as_string(),idx->as_string());
  if (!idx->contains(this->shared_from_this())) {
    if (trace) print("Not contained; throwing\n");
    throw(fmt::format("Need containment for relativize {} to {}",
                      this->as_string(),idx->as_string()));
  }
  if (volume()==1)
    return shared_ptr<indexstruct<I,d>>{
      make_shared<contiguous_indexstruct<I,d>>( idx->first_index() ) };

  /*
   * Case : relativize against other strided
   */
  strided_indexstruct<I,d>* strided = dynamic_cast<strided_indexstruct<I,d>*>(idx.get());
  if (strided!=nullptr) { // strided & strided
    if (stride_amount==strided->stride_amount) { // same stride
      if (first%stride_amount!=strided->first%strided->stride_amount) { // interleaved
        return shared_ptr<indexstruct<I,d>>{ make_shared<empty_indexstruct<I,d>>() };
      } else {
	auto
	  mn = coordmax<I,d>(first,strided->first),
	  mx = coordmin<I,d>(last,strided->last);
        if (mn>mx)
	  return shared_ptr<indexstruct<I,d>>{ make_shared<empty_indexstruct<I,d>>() };
        else {
          return shared_ptr<indexstruct<I,d>>{ make_shared<contiguous_indexstruct<I,d>
						      >((mn-strided->first)/stride_amount,(mx-strided->first)/stride_amount) };
        }
      }
    }
  }

  /*
   * Case : relativize against indexed
   */
  indexed_indexstruct<I,d>* indexed = dynamic_cast<indexed_indexstruct<I,d>*>(idx.get());
  if (indexed!=nullptr) {
    auto relext = shared_ptr<indexstruct<I,d>>( make_shared<indexed_indexstruct<I,d>>() );
    throw("disabled case");
    // for (auto i : *this) {
    //   relext->add_in_element( idx->find(i) );
    // }
    return relext;
  }

  /*
   * Case : relativize against composite
   */
  composite_indexstruct<I,d>* composite = dynamic_cast<composite_indexstruct<I,d>*>(idx.get());
  if (composite!=nullptr) {
    // easy case: contained in one member of the composite
    auto structs = composite->get_structs(); int found_s{-1}; I shift{0};
    for (int is=0; is<structs.size(); is++) {
      auto istruct = structs.at(is);
      if (istruct->contains(idx)) {
	auto rstruct = relativize_to(istruct);
	return rstruct->operate( shift_operator<I,d>(shift) );
      }
      shift += istruct->volume();
    }
    // general case
    auto rel = shared_ptr<indexstruct<I,d>>( make_shared<empty_indexstruct<I,d>>() );
    for ( auto s : structs ) {
      auto intersection = intersect(s);
      rel = rel->struct_union( intersection->relativize_to(s) );
    }
    return rel;
  }

  print("Unimplemented or invalid strided relativize to {}",idx->type_as_string());
  throw(format("Unimplemented or invalid strided relativize to {}",idx->type_as_string()));
};
//snippet

//! \todo that copy only serves const-correctness
template<typename I,int d>
shared_ptr<indexstruct<I,d>> strided_indexstruct<I,d>::convert_to_indexed() const {
  auto indexed = shared_ptr<indexstruct<I,d>>( make_shared<indexed_indexstruct<I,d>>() );
  indexed->reserve( volume() );
  auto this_copy = *this;
  for (auto v : this_copy)
    indexed = indexed->add_element(v);
  return indexed;
};

template<typename I,int d>
shared_ptr<indexstruct<I,d>> strided_indexstruct<I,d>::struct_union
        ( shared_ptr<indexstruct<I,d>> idx,bool trace ) {
  if (contains(idx)) {
    if (trace) print("union with already contained struct\n");
    return this->shared_from_this();
  } else if (idx->contains(this->shared_from_this())) {
    if (trace) print("union with containing struct\n");
    return idx;
  }
  
  /*
   * Case : union with other strided
   */
  strided_indexstruct<I,d>* strided = dynamic_cast<strided_indexstruct<I,d>*>(idx.get());
  if (strided!=nullptr) {
    if (trace) print("union strided+strided\n");
    const auto& amt = idx->stride();
    const auto& frst = idx->first_index();
    const auto& lst = idx->last_index();
    if (stride_amount==amt && first%stride_amount==frst%amt &&
	frst<=last+stride_amount && first-stride_amount<=lst) {
      auto
	mn = coordmin<I,d>(first,frst),
	mx = coordmax<I,d>(last,lst);
      if (stride_amount==1)
	return shared_ptr<indexstruct<I,d>>{make_shared<contiguous_indexstruct<I,d>>(mn,mx)};
      else
	return shared_ptr<indexstruct<I,d>>{make_shared<strided_indexstruct<I,d>>(mn,mx,stride_amount)};
    } else if (volume()<5 && idx->volume()<5) {
      auto indexed = convert_to_indexed();
      return indexed->struct_union( idx->convert_to_indexed() );
    } else {
      auto composite = make_shared<composite_indexstruct<I,d>>();
      // composite = composite->struct_union(this->make_clone());
      // composite = composite->struct_union(idx->make_clone());
      composite->push_back(this->make_clone());
      composite->push_back(idx->make_clone());
      return composite->force_simplify();
    }
  }

  /*
   * Case : union with indexed
   */
  indexed_indexstruct<I,d>* indexed = dynamic_cast<indexed_indexstruct<I,d>*>(idx.get());
  if (indexed!=nullptr) {
    if (trace) print("union strided+indexed\n");
    auto extstrided = this->make_clone();
    for (auto v : *indexed) {
      if (extstrided->contains_element(v)) {
	if (trace) print("already contains element: {}\n",v.as_string()); // formatter broken
	continue;
      } else if (extstrided->can_incorporate(v)) {
	if (trace) print("can extend with element: {}\n",v.as_string()); // formatter broken
	extstrided = extstrided->add_element(v);
      } else { // an indexed struct can incorporate arbitrary crap
	if (trace) print("can not incorporate: return as indexed\n");
	auto indexplus = indexed->struct_union(this->shared_from_this());
	if (trace) print("reverse incorporation as indexed: {}\n",indexplus->as_string());
	indexplus = indexplus->force_simplify(trace);
	if (trace) print("force simplify: {}\n",indexplus->as_string());
	return indexplus;
      }
    }
    return shared_ptr<indexstruct<I,d>>( extstrided->force_simplify() );
  }

  /*
   * Case : union with composite
   */
  composite_indexstruct<I,d>* composite = dynamic_cast<composite_indexstruct<I,d>*>(idx.get());
  if (composite!=nullptr) {
    if (trace) print("union strided+composite\n");
    return idx->struct_union(this->shared_from_this());
  }

  throw(fmt::format("Unimplemented union: {} & {}",
		    type_as_string(),idx->type_as_string()));
};

//! Split along the start/end points of a second indexstruct
template<typename I,int d>
shared_ptr<indexstruct<I,d>> strided_indexstruct<I,d>::split
    ( shared_ptr<indexstruct<I,d>> idx ) {
  const auto& s = stride();
  auto res = make_shared<composite_indexstruct<I,d>>();
  auto f = first_index(), ff = idx->first_index(), l = last_index(), ll = idx->last_index();
  shared_ptr<indexstruct<I,d>> s1,s2;
  if (this->contains_element_in_range(ff)) {
    s1 = shared_ptr<indexstruct<I,d>>(make_shared<contiguous_indexstruct<I,d>>(f,ff-1));
    s2 = shared_ptr<indexstruct<I,d>>(make_shared<contiguous_indexstruct<I,d>>(ff,l));
  } else if (this->contains_element_in_range(ll)) {
    s1 = shared_ptr<indexstruct<I,d>>(make_shared<contiguous_indexstruct<I,d>>(f,ll));
    s2 = shared_ptr<indexstruct<I,d>>(make_shared<contiguous_indexstruct<I,d>>(ll+1,l));
    // res->push_back( intersect(s1) ); res->push_back( intersect(s2) );
  } else
    throw(fmt::format("Needs to contain split points"));
  res->push_back( intersect(s1) );
  res->push_back( intersect(s2) );
  return res;
};

/****
 **** Contiguous indexstruct
 ****/

/****
 **** Indexed indexstruct
 ****/
template<typename I,int d>
indexed_indexstruct<I,d>::indexed_indexstruct( const vector<coordinate<I,d>> idxs )
  : indices(idxs) {
  require_sorted(indices);
};

template<typename I,int d>
indexed_indexstruct<I,d>::indexed_indexstruct( const vector<I> idxs )
  : indexed_indexstruct<I,d>( vector<coordinate<I,d>>( idxs.size() ) ) {
  for ( int i=0; i<indices.size(); i++ ) {
    indices.at(i) = constant_coordinate<I,d>( idxs.at(i) );
  }
  require_sorted(indices);
};

// template<typename I,int d>
// indexed_indexstruct<I,d>::indexed_indexstruct( const I len,const I *idxs ) {
//   indices.reserve(len);
//   I iold;
//   for (int i=0; i<len; i++) {
//     I inew = idxs[i];
//     if (i>0 && inew<=iold) throw(std::string("Only construct from sorted array"));
//     indices.push_back(inew);
//     iold = inew;
//   }
// };

/*!
  Add an element to an indexed indexstruct. We don't simplify: that
  can be done through \ref force_simplify.
*/
template<typename I,int d>
shared_ptr<indexstruct<I,d>> indexed_indexstruct<I,d>::add_element( coordinate<I,d> idx ) const {
  auto added(*this);
  added.add_in_element(idx);
  return added.make_clone();
};

//! Add an element into this structure.
template<typename I,int d>
void indexed_indexstruct<I,d>::add_in_element( coordinate<I,d> idx ) {
  if (indices.size()==0 || idx>indices.at(indices.size()-1)) {
    indices.push_back(idx);
  } else {
    // this loop can not be ranged....
    for (auto it=indices.begin(); it!=indices.end(); ++it) {
      if (*it==idx) break;
      else if (*it>idx) { indices.insert(it,idx); break; }
    }
  }
};

template<typename I,int d>
shared_ptr<indexstruct<I,d>> indexed_indexstruct<I,d>::translate_by( coordinate<I,d> shift ) const {
  auto translated = shared_ptr<indexstruct<I,d>>{
    new indexed_indexstruct<I,d>() };
  for (int loc=0; loc<indices.size(); loc++)
    translated->add_in_element( indices.at(loc)+shift );
  return translated;
};

//! See if we can turn in indexed into contiguous, but cautiously.
template<typename I,int d>
shared_ptr<indexstruct<I,d>> indexed_indexstruct<I,d>::simplify() {
  if (this->volume()>SMALLBLOCKSIZE)
    return force_simplify();
  else
    return this->make_clone();
};

/*!
  See if we can turn in indexed into contiguous or strided.
*/
template<typename I,int d>
shared_ptr<indexstruct<I,d>> indexed_indexstruct<I,d>::make_strided(bool trace) const {
  int ileft = 0, iright = this->volume()-1;
  // try detect strided
  I stride;
  if (is_strided_between_indices(ileft,iright,stride,trace)) {
    auto first = get_ith_element(ileft), last = get_ith_element(iright);
    if (stride==1)
      return shared_ptr<indexstruct<I,d>>
	( make_shared<contiguous_indexstruct<I,d>>(first,last) );
    else
      return shared_ptr<indexstruct<I,d>>
	( make_shared<strided_indexstruct<I,d>>(first,last,stride) );
  } else return nullptr;
};

template<typename I,int d>
shared_ptr<indexstruct<I,d>> indexed_indexstruct<I,d>::force_simplify(bool trace) const {
  print( "force simplify {}\n",this->as_string());
  try {
    if (false) {
    } else if (this->volume()==0) {
      return shared_ptr<indexstruct<I,d>>( make_shared<empty_indexstruct<I,d>>() );
    } else if (this->volume()==this->outer_volume()) {
      // easy simplification to contiguous
      return
	shared_ptr<indexstruct<I,d>>
	( make_shared<contiguous_indexstruct<I,d>>(first_index(),last_actual_index()) );
    } else {
      auto ret = make_strided(trace);
      if (ret!=nullptr) {
	if (trace) print("simplify to strided\n");
	return ret;
      } else {
	if (d>1) throw( "force case only for d==1" );
	I ileft = 0, iright = this->volume()-1;
	// find strided subsections
	auto first = get_ith_element(ileft), last = get_ith_element(iright);
	for (I find_left=ileft; find_left<iright; find_left++) {
	  I top_right = iright; if (find_left==ileft) top_right--;
	  I stride;
	  for (I find_right=top_right; find_right>find_left+1; find_right--) {
	    // if we find one section, we replace it by a 3-composite. ultimately recursive?
	    if (is_strided_between_indices(find_left,find_right,stride,trace)) {
	      auto comp = shared_ptr<composite_indexstruct<I,d>>
		( make_shared<composite_indexstruct<I,d>>() );
	      auto [found_left,found_right] = [=,*this,&comp] () {
		auto found_left = get_ith_element(find_left),
		  found_right = get_ith_element(find_right);
		if (find_left>ileft) { // there is stuff to the left
		  auto left = get_ith_element(find_left);
		  auto left_part = this->minus
		    ( shared_ptr<indexstruct<I,d>>
		      (make_shared<contiguous_indexstruct<I,d>>(left,last)) );
		  comp->push_back(left_part);
		} 
		return pair(found_left,found_right);
	      }();
	      shared_ptr<indexstruct<I,d>> strided;
	      if (stride==1)
		strided = shared_ptr<indexstruct<I,d>>
		  ( make_shared<contiguous_indexstruct<I,d>>(found_left,found_right) );
	      else
		strided = shared_ptr<indexstruct<I,d>>
		  ( make_shared<strided_indexstruct<I,d>>(found_left,found_right,stride) );
	      comp->push_back(strided);
	      if (find_right<iright) { // there is stuff to the right
		auto right = get_ith_element(find_right);
		auto right_part = this->minus
		  ( shared_ptr<indexstruct<I,d>>(make_shared<contiguous_indexstruct<I,d>>(first,right)) );
		right_part = right_part->force_simplify();
		//print("composing with right: {}\n",right_part->as_string());
		if (right_part->is_composite()) {
		  composite_indexstruct<I,d>* new_right =
		    dynamic_cast<composite_indexstruct<I,d>*>(right_part.get());
		  if (new_right==nullptr)
		    throw(fmt::format("could not upcast supposed composite (indexed simplify)"));
		  for ( auto ir : new_right->get_structs() )
		    comp->push_back(ir);
		} else
		  comp->push_back(right_part);
		//print(" .. gives {}\n",comp->as_string());
	      }
	      //print("composed as {}\n",comp->as_string());
	      return comp;
	    }
	  }
	}
	// if the above loop did not return, just return ourself
	return this->make_clone();
      }
    }
  } catch (string e) { print("Error: {}",e);
    throw("Could not simplify indexed indexstruct");
  } catch( ... ) {
    throw("Could not simplify indexed indexstruct");
  }
};

//! Detect a strided subsection in the indexed structure
template<typename I,int d>
bool indexed_indexstruct<I,d>::is_strided_between_indices( I ileft,I iright,I &stride,bool trace) const {
  auto first = get_ith_element(ileft), last = get_ith_element(iright);
  I n_index = iright-ileft+1;
  if (n_index==1)
    throw(fmt::format("Single point should have been caught"));

  // if this is strided, what would the stride be?
  auto stride_vector = (last-first)/(n_index-1);
  stride = stride_vector[0];
  // accept only identical in all dimensions
  if (stride!=stride_vector[d-1])
    // we should test the in between ones
    return false;

  // let's see if the bounds are proper for the stride
  if (trace) print("first={}, last={}, does that fit {} x stride {}\n",
		   first[0],last[0],n_index-1,stride);
  if ( not ( first+(n_index-1)*stride==last) ) {
    if (trace) print("stride does not fit\n");
    return false;
  }

  // test if everything in between is also strided
  for (I inext=ileft+1; inext<iright; inext++) {
    auto elt = get_ith_element(inext);
    auto inplace = (elt-first)%stride;
    if (trace)
      print("element {}: {} has modulo: {}\n",
	    inext,elt.as_string(),inplace.as_string());  // formatter broken
    if ( not ( inplace==0 ) ) // coordinate<I,d>(0) ) )
      // strange. why doesn't the != operator work here?
      return false;
  }
  return true; // iright>ileft+1; // should be at least 3 elements
};

template<typename I,int d>
coordinate<I,d> indexed_indexstruct<I,d>::get_ith_element( const I i ) const {
  if (i<0 || i>=this->volume())
    throw(fmt::format("Index {} out of range for {}",i,as_string()));
  return indices[i];
};

template<typename I,int d>
shared_ptr<indexstruct<I,d>> indexed_indexstruct<I,d>::intersect
    ( shared_ptr<indexstruct<I,d>> idx ) {
  /*
   * Case : intersect with strided
   */
  strided_indexstruct<I,d>* strided = dynamic_cast<strided_indexstruct<I,d>*>(idx.get());
  if (strided!=nullptr) { // indexed & strided
    auto limited = shared_ptr<indexstruct<I,d>>( make_shared<indexed_indexstruct<I,d>>() );
    auto first = strided->first_index(), last = strided->last_index();
    for (auto v : indices)
      if (first<=v && v<=last)
    	limited = limited->add_element(v);
    return limited;
  }
  /*
   * Case: intersect with indexed => reverse
   */
  indexed_indexstruct<I,d>* indexed = dynamic_cast<indexed_indexstruct<I,d>*>(idx.get());
  if (indexed!=nullptr) { // strided & indexed
    auto limited = shared_ptr<indexstruct<I,d>>( make_shared<indexed_indexstruct<I,d>>() );
    for (auto v : indices)
      if (indexed->contains_element(v))
	limited = limited->add_element(v);
    return limited;
  }
  throw(std::string("Unimplemented intersect with indexed"));
};

//! \todo lose those clones
template<typename I,int d>
bool indexed_indexstruct<I,d>::contains( const shared_ptr<indexstruct<I,d>>& idx ) const {
  if (idx->volume()==0) return true;
  if (this->volume()==0) return false;

  /*
   * Case : contains strided
   */
  strided_indexstruct<I,d>* strided = dynamic_cast<strided_indexstruct<I,d>*>(idx.get());
  auto nonconst = make_clone();
  if (strided!=nullptr) {
    for (auto v : *strided) {
      if (!nonconst->contains_element(v)) {
	return false; }
    }
    return true;
  }
  /*
   * Case: contains indexed
   */
  indexed_indexstruct<I,d>* indexed = dynamic_cast<indexed_indexstruct<I,d>*>(idx.get());
  if (indexed!=nullptr) {
    auto nonconst = make_clone();
    for (auto v : indexed->indices) {
      if (!nonconst->contains_element(v)) {
	return false;
      }
    }
    return true;
  }

  /*
   * Case: contains composite
   */
  composite_indexstruct<I,d>* composite = dynamic_cast<composite_indexstruct<I,d>*>(idx.get());
  if (composite!=nullptr) {
    //    auto nonconst = make_clone();
    for (auto s : composite->get_structs())
      if (!/*nonconst->*/contains(s)) return false;
    return true;
  }
  
  throw(fmt::format("indexed contains case not implemented: {}",idx->type_as_string()));
};

//! \todo weird clone because of a const problem.
template<typename I,int d>
shared_ptr<indexstruct<I,d>> indexed_indexstruct<I,d>::minus
    ( shared_ptr<indexstruct<I,d>> idx ) const{
  if (idx->is_empty())
    return this->make_clone();
  if (idx->contains(this->make_clone())) //(this->shared_from_this()))
    return shared_ptr<indexstruct<I,d>>(make_shared<empty_indexstruct<I,d>>());
    
  /*
   * Case : minus strided
   */
  strided_indexstruct<I,d>* strided = dynamic_cast<strided_indexstruct<I,d>*>(idx.get());
  if (strided!=nullptr) {
    auto indexminus = shared_ptr<indexstruct<I,d>>( make_shared<indexed_indexstruct<I,d>>() );
    for (auto v : indices)
      if (!idx->contains_element(v))
	indexminus = indexminus->add_element(v);
    if (indexminus->volume()==0)
      return shared_ptr<indexstruct<I,d>>(make_shared<empty_indexstruct<I,d>>());
    else
      return indexminus;
  }
  /*
   * Case: minus other indexed
   */
  indexed_indexstruct<I,d>* indexed = dynamic_cast<indexed_indexstruct<I,d>*>(idx.get());
  if (indexed!=nullptr) {
    auto indexminus = shared_ptr<indexstruct<I,d>>( make_shared<indexed_indexstruct<I,d>>() );
    for (auto v : indices)
      if (!idx->contains_element(v))
	indexminus = indexminus->add_element(v);
    if (indexminus->volume()==0)
      return shared_ptr<indexstruct<I,d>>(make_shared<empty_indexstruct<I,d>>());
    else
      return indexminus;
  }
  /*
   * Case: minus other composite
   */
  composite_indexstruct<I,d>* composite = dynamic_cast<composite_indexstruct<I,d>*>(idx.get());
  if (composite!=nullptr) {
    auto indexminus = this->make_clone();
    auto structs = composite->get_structs();
    for (auto s : structs) {
      indexminus = indexminus->minus(s);
    }
    return indexminus;
  }
  throw(std::string("Unimplemented indexed minus"));
};

//! Disjointness test for indexed. \todo bunch of unimplemented cases
template<typename I,int d>
bool indexed_indexstruct<I,d>::disjoint( shared_ptr<indexstruct<I,d>> idx ) {
  bool range_disjoint = indexstruct<I,d>::disjoint(idx);
  if (range_disjoint==true)
    return true;
  else {
    strided_indexstruct<I,d>* strided = dynamic_cast<strided_indexstruct<I,d>*>(idx.get());
    if (strided!=nullptr) {
      return strided->disjoint(this->shared_from_this());
    } else 
      throw(fmt::format("unimplemented disjoint: indexed & {}",idx->type_as_string()));
  }
};

//! \todo what do we need that nonconst for? equals and contains are const methods.
template<typename I,int d>
bool indexed_indexstruct<I,d>::equals( const shared_ptr<indexstruct<I,d>>& idx ) const {
  // UGLY: just using shared_from_this gives you a const shared_ptr
  auto nonconst = make_clone(); // this->shared_from_this() ????
  return this->contains(idx) && idx->contains(nonconst);
};

template<typename I,int d>
shared_ptr<indexstruct<I,d>> indexed_indexstruct<I,d>::relativize_to
    ( shared_ptr<indexstruct<I,d>> idx,bool trace ) {
  if (!idx->contains(this->shared_from_this()))
    throw(fmt::format("Need containment for relativize {} to {}",
		      this->as_string(),idx->as_string()));

  /*
   * Case : relativize against other strided
   */
  strided_indexstruct<I,d>* strided = dynamic_cast<strided_indexstruct<I,d>*>(idx.get());
  if (strided!=nullptr) {
    auto shift = strided->first_index();
    return shared_ptr<indexstruct<I,d>>{ this->translate_by( -shift ) };
  }

  /*
   * Case : relativize against indexed
   */
  indexed_indexstruct<I,d>* indexed = dynamic_cast<indexed_indexstruct<I,d>*>(idx.get());
  if (indexed!=nullptr) {
    auto relative = shared_ptr<indexstruct<I,d>>{ make_shared<indexed_indexstruct<I,d>>() };
    int count = 0;
    for (auto v : indices)
      throw("this completely doesn't make sense");
      // relative = relative->add_element( indexed->find(v) );
    return relative;
  }

  /*
   * Case : relativize against composite
   */
  composite_indexstruct<I,d>* composite = dynamic_cast<composite_indexstruct<I,d>*>(idx.get());
  if (composite!=nullptr) {
    auto relative = shared_ptr<indexstruct<I,d>>{ make_shared<indexed_indexstruct<I,d>>() };
    int count = 0;
    for (auto v : indices)
      throw("I doubt that this makes sense");
      //relative = relative->add_element( composite->find(v) );
    return relative;
  }
  
  throw(fmt::format("Unimplemented indexed relativize to {}",idx->type_as_string()));
};

template<typename I,int d>
string indexed_indexstruct<I,d>::as_string() const {
  memory_buffer w;
  format_to(std::back_inserter(w),"[");
  for ( const auto& i : indices )
    format_to(std::back_inserter(w),"{}",i.as_string());  // formatter broken
  format_to(std::back_inserter(w),"]");
  return to_string(w);
};

//! \todo can we lose that clone?
template<typename I,int d>
shared_ptr<indexstruct<I,d>> indexed_indexstruct<I,d>::struct_union
        ( shared_ptr<indexstruct<I,d>> idx,bool trace ) {
  if (contains(idx)) {
    if (trace) print("union with already contained struct\n");
    return this->make_clone();
  } else if (idx->contains(this->shared_from_this())) {
    if (trace) print("union with containing struct\n");
    return idx->make_clone();
  }

  /*
   * Case : union with strided
   */
  strided_indexstruct<I,d>* strided = dynamic_cast<strided_indexstruct<I,d>*>(idx.get());
  if (strided!=nullptr) {
    if (trace) print("union indexed+strided\n");
    if (strided->volume()<SMALLBLOCKSIZE) {
      auto indexed = this->make_clone();
      const auto& s = strided->stride();
      for (auto i : *strided)
	indexed = indexed->add_element(i);
      return indexed->make_clone();
    } else {
      auto composite =shared_ptr<composite_indexstruct<I,d>>{ make_shared<composite_indexstruct<I,d>>() };
      composite->push_back(this->make_clone());
      composite->push_back(idx);
      return composite->force_simplify();
    }
  }
  /*
   * Case : union with indexed
   */
  indexed_indexstruct<I,d>* indexed = dynamic_cast<indexed_indexstruct<I,d>*>(idx.get());
  if (indexed!=nullptr) {
    if (trace) print("union indexed+indexed\n");
    auto merged = indexed->make_clone();
    for (auto v : indices)
      merged = merged->add_element(v);
    return merged->make_clone();
  }

  /*
   * Case : union with composite
   */
  composite_indexstruct<I,d>* composite = dynamic_cast<composite_indexstruct<I,d>*>(idx.get());
  if (composite!=nullptr) {
    if (trace) print("union indexed+composite\n");
    return idx->struct_union(this->shared_from_this());
  }

  throw(fmt::format("Unimplemented union: {} & {}",
		    type_as_string(),idx->type_as_string()));
};

/****
 **** Composite
 ****/

/*! Add a new structure to a composite, keeping the whole sorted on 
  the first index of the component.
*/
template<typename I,int d>
void composite_indexstruct<I,d>::push_back( shared_ptr<indexstruct<I,d>> idx ) {
  if (idx->is_empty()) return;
  if (structs.size()==0)
    structs.push_back(idx);
  else {
    //print("{} push {}\n",this->as_string(),idx->as_string());
    auto s=structs.end(); bool last{true};
    while (s!=structs.begin()) {
      --s;
      if (last && idx->first_index()>(*s)->first_index()) {
	structs.push_back(idx);
	break;
      }
      last = false;
      if (idx->first_index()<(*s)->first_index()) {
	structs.insert(s,idx);
	break;
      }
    }
    //print("... pushed {}\n",this->as_string());
  }
};

//! \todo the use of shared ptr here is dangerous. can we just copy the shared ptr?
template<typename I,int d>
shared_ptr<indexstruct<I,d>> composite_indexstruct<I,d>::make_clone() const {
  auto runion = shared_ptr<composite_indexstruct>{ make_shared<composite_indexstruct<I,d>>() };
  for ( auto s : structs ) 
    runion->push_back( s->make_clone() );
  return runion;
};

template<typename I,int d>
coordinate<I,d> composite_indexstruct<I,d>::first_index() const {
  //  throw("composite first index");
  if (structs.size()==0)
    throw(std::string("Can not get first from empty composite"));
  auto f = structs.at(0)->first_index();
  for (auto s : structs)
    f = coordmin<I,d>(f,s->first_index());
  return f;
};

template<typename I,int d>
coordinate<I,d> composite_indexstruct<I,d>::last_actual_index() const {
  if (structs.size()==0)
    throw(std::string("Can not get last from empty composite"));
  auto f = structs.at(0)->last_actual_index();
  for (auto s : structs)
    f = coordmax(f,s->last_actual_index());
  return f;
};

template<typename I,int d>
I composite_indexstruct<I,d>::volume() const {
  I siz=0;
  for (auto s : structs) {
    //printf("localsize: struct@%d\n",(long)(s.get()));
    siz += s->volume();
  }
  return siz;
};

//! \todo I have my doubts
template<typename I,int d>
coordinate<I,d> composite_indexstruct<I,d>::get_ith_element( const I i ) const {
  if (i>=volume())
    throw(fmt::format("Requested index {} out of bounds for {}",i,as_string()));
  if (structs.size()==1)
    return structs.at(0)->get_ith_element(i);
  I ilocal = i;
  coordinate<I,d> start_check;
  for ( auto s : structs ) {
    if (ilocal==i)
      start_check = s->first_index();
    else {
      auto sf = s->first_index();
      if (sf<start_check)
	print("WARNING composite not in increasing order: {}\n",as_string());
      start_check = sf;
    }
    if (ilocal>=s->volume())
      ilocal -= s->volume();
    else
      return s->get_ith_element(ilocal);
  }
  throw(fmt::format("still need to count down {} for ith element {} in composite {}",
		    ilocal,i,as_string()));
};

/*! Composite containment is tricky
  \todo is there a way to optimize the indexed case?
*/
template<typename I,int d>
bool composite_indexstruct<I,d>::contains( const shared_ptr<indexstruct<I,d>>& idx ) const {
  /*
   * Case: contains strided. 
   * test by chopping off pieces
   */
  strided_indexstruct<I,d>* strided = dynamic_cast<strided_indexstruct<I,d>*>(idx.get());
  if (strided!=nullptr) {
    auto skip = idx->make_clone();
    for ( auto s : structs ) {
      skip = skip->minus(s);
      if (skip->is_empty())
	return true;
    }
    return false;
  }

  /*
   * Case: contains indexed. 
   * test by chopping off pieces
   */
  indexed_indexstruct<I,d>* indexed = dynamic_cast<indexed_indexstruct<I,d>*>(idx.get());
  if (indexed!=nullptr) {
    auto skip = idx->make_clone();
    for ( auto s : structs ) {
      skip = skip->minus(s);
      if (skip->is_empty())
	return true;
    }
    return false;
  }

  /*
   * Case: contains composite. 
   * test by chopping off pieces
   */
  composite_indexstruct<I,d>* composite = dynamic_cast<composite_indexstruct<I,d>*>(idx.get());
  if (composite!=nullptr) {
    for ( auto s : composite->get_structs() ) {
      if (!contains(s))
	return false;
    }
    return true;;
  }

  throw(fmt::format("Unimplemented case: composite contains {}",idx->type_as_string()));
};

template<typename I,int d>
bool composite_indexstruct<I,d>::contains_element( const coordinate<I,d>& idx ) const {
  //for (auto s : structs)
  for (int is=0; is<structs.size(); is++) {
    auto s = structs.at(is);
    if (s->contains_element(idx))
      return true;
  }
  return false;
};

template<typename I,int d>
I composite_indexstruct<I,d>::find( coordinate<I,d> idx ) const {
  auto first = structs[0]->first_index();
  I accumulate{0};
  for ( auto s : structs ) {
    if (s->first_index()<first)
      throw(fmt::format("Composite not sorted: {}",as_string()));
    first = structs[0]->first_index();
    if (s->contains_element(idx)) {
      return accumulate+s->find(idx);
    } else
      accumulate += s->volume();
  }
  throw(format("Could not find {} in <<{}>>",idx[0],as_string()));
};

//! Disjointness test for composite. \todo bunch of unimplemented cases
template<typename I,int d>
bool composite_indexstruct<I,d>::disjoint( shared_ptr<indexstruct<I,d>> idx ) {
  bool range_disjoint = indexstruct<I,d>::disjoint(idx);
  if (range_disjoint==true)
    return true;
  else {
    throw(fmt::format("unimplemented disjoint: composite & {}",idx->type_as_string()));
  }
};

/*!
  The first structure is installed no matter what. 
  For every next we try to simplify the structure.
  \todo doesn't clone already make a shared ptr?
 */
template<typename I,int d>
shared_ptr<indexstruct<I,d>> composite_indexstruct<I,d>::struct_union
        ( shared_ptr<indexstruct<I,d>> idx,bool trace) {
  //print("composite union {} + {}\n",as_string(),idx->as_string());

  // already contained: return
  for (auto s : structs)
    if (s->contains(idx)) {
      if (trace) print("union composite and already contained\n");
      return this->make_clone();
    }

  // empty: should use push_back to add first member
  if (structs.size()==0) {
    throw(fmt::format("Should use push_back to insert first member in composite, not union"));
  }

  // try to merge with existing struct of same type
  {
    //print("  try to merge\n");
    bool can{false};
    for (auto s : structs) {
      if (s->can_merge_with_type(idx)) {
	//print("  can merge with {}\n",s->as_string());
	can = true;
      }
    }
    if (can) {
      auto composite = shared_ptr<composite_indexstruct>( make_shared<composite_indexstruct<I,d>>() );
      for (auto s : structs) {
	if (s->is_composite())
	  throw(fmt::format
		("How did a composite {} wind up in {}",s->as_string(),this->as_string()));
	if (s->can_merge_with_type(idx)) {
	  auto extend =  s->struct_union(idx);
	  // print("  merged with component {} giving {}\n",
	  // 	     s->as_string(),extend->as_string());
	  if (extend->is_composite())
	    throw(fmt::format("Illegal merge {} into {}",idx->as_string(),s->as_string()));
	  composite->push_back(extend);
	} else {
	  composite->push_back(s->make_clone());
	}
      }
      return composite;
    }
  }

  // try to merge with existing indexed
  if (idx->volume()==1) {
    //print("  merge single point {}\n",idx->as_string());
    bool was_merged{false};
    auto composite = shared_ptr<composite_indexstruct>( make_shared<composite_indexstruct<I,d>>() );
    for (auto s : structs) {
      if (s->is_indexed()) {
	composite->push_back( s->add_element( idx->first_index() ) );
	was_merged = true;
      } else {
	composite->push_back( s->make_clone() );
      }
    }
    if (!was_merged) {
      composite->push_back(idx);
    }
    return composite;
  }

  // merge in: weed out both existing members and the new contribution
  {
    auto composite = shared_ptr<composite_indexstruct>( make_shared<composite_indexstruct<I,d>>() );
    //print("  weeding out and add_ing\n");
    for (auto s : structs) {
      auto new_s = s->minus(idx);
      if (new_s->is_empty())
	continue;
      if (new_s->is_composite()) {
	auto s_comp = dynamic_cast<composite_indexstruct<I,d>*>(new_s.get());
	if (s_comp==nullptr)
	  throw(fmt::format("could not upcast new_s"));
	for ( auto ss : s_comp->get_structs() )
	  composite->push_back(ss);
      } else
	composite->push_back(new_s);
    }
    auto comp = dynamic_cast<composite_indexstruct<I,d>*>(idx.get());
    if (comp!=nullptr) {
      for ( auto s : comp->structs )
	composite->structs.push_back(s);
    } else     
      composite->push_back(idx);
    return composite; // no simplify
  }
};

template<typename I,int d>
shared_ptr<indexstruct<I,d>> composite_indexstruct<I,d>::intersect
    ( shared_ptr<indexstruct<I,d>> idx ) {
  // rule out the hard case
  composite_indexstruct<I,d>* composite = dynamic_cast<composite_indexstruct<I,d>*>(idx.get());
  if (composite!=nullptr)
    throw(std::string("Can not intersect composite-composite"));

  auto result = shared_ptr<indexstruct<I,d>>{ make_shared<empty_indexstruct<I,d>>() };
  for (auto s : structs) {
    result = result->struct_union( s->intersect(idx) );
  }
  return result;
};

//! Composite minus set does a minus on each component \todo make unittest
template<typename I,int d>
shared_ptr<indexstruct<I,d>> composite_indexstruct<I,d>::minus
    ( shared_ptr<indexstruct<I,d>> idx ) const {
  auto rstruct = shared_ptr<composite_indexstruct>{ make_shared<composite_indexstruct<I,d>>() };
  for (auto s : structs) {
    auto new_one = s->minus(idx);
    if (!new_one->is_empty()) {
      if (new_one->is_composite()) {
	composite_indexstruct<I,d>* new_comp = dynamic_cast<composite_indexstruct<I,d>*>(new_one.get());
      if (new_comp==nullptr)
	throw(fmt::format("could not upcast supposed composite (minus)"));
	for ( auto s : new_comp->get_structs() )
	  rstruct->push_back(s);
      } else
	rstruct->push_back(new_one);
    }
  }
  return rstruct->force_simplify();
};

template<typename I,int d>
shared_ptr<indexstruct<I,d>> composite_indexstruct<I,d>::relativize_to
    ( shared_ptr<indexstruct<I,d>> idx,bool trace ) {
  auto rel = shared_ptr<indexstruct<I,d>>( make_shared<empty_indexstruct<I,d>>() );
  for ( auto s : structs ) {
    try {
      rel = rel->struct_union( s->relativize_to(idx) );
    } catch (std::string c) {
      throw(fmt::format("Composite relativive: {}",c));
    }
  }
  return rel;
};

//! Simplify a composite. We cover only some cases.
template<typename I,int d>
shared_ptr<indexstruct<I,d>> composite_indexstruct<I,d>::force_simplify(bool trace) const {

  try {
    // first the obvious simplicitions: empty and just one member
    if (get_structs().size()==0)
      return shared_ptr<indexstruct<I,d>>(make_shared<empty_indexstruct<I,d>>());
    else if (get_structs().size()==1)
      return get_structs()[0]->make_clone();

    { // try to merge multiple contiguous structs
      auto structs = get_structs();
      int ns = structs.size();
      int ncont{0},a_cont{-1};
      for (int is=0; is<ns; is++)
	if (structs[is]->is_contiguous()) {
	  ncont++; a_cont = is; }
      if (ncont>1) {
	for (int ic=0; ic<ns-1; ic++) {
	  if (structs.at(ic)->is_contiguous()) {
	    for (int jc=ic+1; jc<ns; jc++) {
	      if (structs.at(jc)->is_contiguous()) {
		if (structs.at(ic)->can_merge_with_type(structs.at(jc))) {
		  auto composite = shared_ptr<composite_indexstruct>
		    ( make_shared<composite_indexstruct<I,d>>() );
		  auto new_struct = structs.at(ic)->struct_union(structs.at(jc));
		  for (int is=0; is<ns; is++) {
		    if (is==ic || is==jc) continue;
		    composite->push_back( structs.at(is) );
		  }
		  composite->push_back(new_struct);
		  return composite->force_simplify();
		} } } } }
      }
    }
  
    { // try to merge multiple indexed structs
      auto structs = get_structs();
      int ns = structs.size();
      int nind{0},an_ind{-1};
      for (int is=0; is<ns; is++)
	if (structs[is]->is_indexed()) {
	  nind++; an_ind = is; }
      if (nind>1) {
	// merge all the indexeds
	auto composite = shared_ptr<composite_indexstruct>( make_shared<composite_indexstruct<I,d>>() );
	auto indexed   = shared_ptr<indexstruct<I,d>>( make_shared<indexed_indexstruct<I,d>>() );
	for (int is=0; is<ns; is++) {
	  auto istruct = structs.at(is);
	  if (structs.at(is)->is_indexed())
	    indexed = indexed->struct_union(istruct);
	  else
	    composite->push_back(istruct);
	}
	indexed = indexed->force_simplify();
	if (indexed->is_composite()) {
	  composite_indexstruct<I,d>* icomp = dynamic_cast<composite_indexstruct<I,d>*>(indexed.get());
	  if (icomp==nullptr)
	    throw(fmt::format("could not upcast supposed composite (composite simplify)"));
	  for ( auto is : icomp->get_structs() )
	    composite->push_back(is);
	} else
	  composite->push_back(indexed);
	return composite->force_simplify();
      } else if (nind==1) { // see if outer elements can be merged in
	//print("{} has 1 indexed\n",as_string());
	auto indexed = structs.at(an_ind); bool can_merge{false};
	for (int is=0; is<ns; is++) {
	  if (is!=an_ind) {
	    auto istruct = structs.at(is);
	    for (int ii=0; ii<indexed->volume(); ii++) {
	      auto i = indexed->get_ith_element(ii);
	      can_merge = structs.at(is)->can_incorporate(i);
	      if (can_merge) goto do_merge;
	    }
	    // print("test incorporation {},{} in {}: {},{}\n",
	    // 	     left,right,istruct->as_string(),can_left,can_right);
	  }
	}
      do_merge:
	if (can_merge) {
	  auto composite = shared_ptr<composite_indexstruct>( make_shared<composite_indexstruct<I,d>>() );
	  for (int is=0; is<ns; is++) {
	    if (is==an_ind) continue;
	    auto istruct = structs.at(is);
	    for (int ii=0; ; ) {
	      auto i = indexed->get_ith_element(ii);
	      if (istruct->can_incorporate(i)) {
		auto i_struct = shared_ptr<indexstruct<I,d>>(make_shared<contiguous_indexstruct<I,d>>(i));
		istruct = istruct->struct_union(i_struct);
		indexed = indexed->minus(i_struct);
	      } else ii++;
	      if (ii>=indexed->volume()) break;
	    }
	    //print("push {}\n",istruct->as_string());
	    composite->push_back(istruct);
	  }
	  composite->push_back(indexed);
	  return composite->force_simplify(); // either this or loop over indexed outers
	}
      }
    }

    { // do we have a strided that can contribute to contiguous?
      int cani = -1;
      for (int is=0; is<structs.size(); is++) {
	auto istruct = structs.at(is);
	if (istruct->is_strided() && istruct->stride()>1) {
	  // try to find a mergeable i
	  auto first=istruct->first_index(),last = istruct->last_index();
	  int canj = -1;
	  for (int js=0; js<structs.size(); js++) {
	    // try to find a j that can incorporate
	    auto jstruct = structs.at(js);
	    if (is!=js && jstruct->is_contiguous() &&
		(jstruct->contains_element(first) || jstruct->contains_element(last)
		 || jstruct->can_incorporate(first) || jstruct->can_incorporate(last)) ) {
	      // found: break j loop
	      canj = js; break; }
	  }
	  if (canj>=0) {
	    // found: break i loop
	    cani = is; break; }
	}
      }
      if (cani>=0) { // merge set i into all others
	auto istruct = structs.at(cani);
	auto first=istruct->first_index(),last = istruct->last_index();
	bool left{true};
	auto composite = shared_ptr<composite_indexstruct>( make_shared<composite_indexstruct<I,d>>() );
	for (int js=0; js<structs.size(); js++) { // loop over all others
	  if (js==cani) continue;
	  auto jstruct = structs.at(js);
	  if (jstruct->is_contiguous()) {
	    if (left && ( jstruct->contains_element(first) || jstruct->can_incorporate(first) ) ) {
	      if (!jstruct->contains_element(first))
		jstruct = jstruct->add_element(first);
	      istruct = istruct->minus
		( shared_ptr<indexstruct<I,d>>( make_shared<contiguous_indexstruct<I,d>>(first) ) );
	      left = !istruct->is_empty();
	      if (left)
		first = istruct->first_index();
	    }
	  }
	  if (jstruct->is_contiguous()) {
	    if (left && ( jstruct->contains_element(last) || jstruct->can_incorporate(last) ) ) {
	      if (!jstruct->contains_element(last))
		jstruct = jstruct->add_element(last);
	      istruct = istruct->minus
		( shared_ptr<indexstruct<I,d>>( make_shared<contiguous_indexstruct<I,d>>(last) ) );
	      left = !istruct->is_empty();
	      if (left)
		last = istruct->last_index();
	    }
	  }
	  composite->push_back(jstruct);
	}
	if (!istruct->is_empty())
	  composite->push_back(istruct);
	// no guarantee of fully incorporating, so we repeat;
	// this also merges any contiguouses
	return composite->force_simplify();
      }
    }

    // default return self
    return this->make_clone(); //shared_from_this();
  } catch (string e) { print("Error: {}",e);
    throw("Could not simplify composite indexstruct");
  } catch( ... ) {
    throw("Could not simplify composite indexstruct");
  }

};

/*!
  If we have a composite with small gaps, try to plaster them shut.
 */
template<typename I,int d>
shared_ptr<indexstruct<I,d>> composite_indexstruct<I,d>::over_simplify() {
  { /*
     * Case: contiguous & { other contiguous or singleton } with a gap of size 1.
     */
    bool gap{false}; int s1{-1},s2{-1};
    for (int is=0; is<structs.size(); is++) {
      auto istruct = structs[is];
      if (!istruct->is_contiguous()) continue;
      for (int js=0; js<structs.size(); js++) {
	auto jstruct = structs[js];
	if (is==js) continue;
	if (jstruct->is_contiguous() || jstruct->volume()==1) {
	  if (istruct->last_index()==jstruct->first_index()-2) {
	    s1 = is; s2 = js; gap = true; }
	  if (jstruct->last_index()==istruct->first_index()-2) {
	    s1 = js; s2 = is; gap = true; }
	}
	if (gap) break;
      }
      if (gap) break;
    }
    if (gap) {
      auto composite = shared_ptr<composite_indexstruct>( make_shared<composite_indexstruct<I,d>>() );
      // merge the two almost-adjacent structs
      auto new_s = shared_ptr<indexstruct<I,d>>
	( make_shared<contiguous_indexstruct<I,d>>( structs[s1]->first_index(),structs[s2]->last_index() ) );
      composite->push_back(new_s);
      // all other structs go as they are
      for (int is=0; is<structs.size(); is++) {
	if (is!=s1 && is!=s2)
	  composite->push_back(structs[is]);
      }
      return composite->over_simplify()->force_simplify();
    }
  }
  return make_clone();
};

//! We convert the component structures in a row and union them.
template<typename I,int d>
shared_ptr<indexstruct<I,d>> composite_indexstruct<I,d>::convert_to_indexed() const {
  auto idx = shared_ptr<indexstruct<I,d>>( make_shared<indexed_indexstruct<I,d>>() );
  for ( auto s : structs ) {
    idx = idx->struct_union( s->convert_to_indexed() );
  }
  return idx;
};

template<typename I,int d>
std::string composite_indexstruct<I,d>::as_string() const {
  fmt::memory_buffer w;
  format_to(w.end(),"composite ({} members):",structs.size());
  //for (auto s : structs)
  for (int is=0; is<structs.size(); is++) {
    auto s = structs.at(is);
    format_to(w.end()," {}",s->as_string());
  }
  return to_string(w);
};

/****
 **** Composite iteration
 ****/

template<typename I,int d>
bool composite_indexstruct<I,d>::equals( const shared_ptr<indexstruct<I,d>>& idx ) const {
  if (structs.size()>1)
    return false;
  else
    return structs.at(0)->equals(idx);
};

/****
 **** indexstructure
 ****/
//! Push onto the structs of a composite
template<typename I,int d>
void indexstructure<I,d>::push_back( indexstructure<I,d> idx ) {
  auto comp = dynamic_cast<composite_indexstruct<I,d>*>(strct.get());
  if (comp!=nullptr) {
    comp->push_back( idx.strct );
  } else {
    print("Not composite; cannot push back\n");
    throw(fmt::format("Can not push into non-multi structure"));
  }
};

//! Get the structs of a composite
template<typename I,int d>
const vector<shared_ptr<indexstruct<I,d>>> indexstructure<I,d>::get_structs() const {
  auto comp = dynamic_cast<composite_indexstruct<I,d>*>(strct.get());
  if (comp!=nullptr) {
    return comp->get_structs();
  } else {
    print("Not composite; cannot get structs\n");
    throw(fmt::format("Can not get structs from non-multi structure"));
  }
};

template<typename I,int d>
string indexstructure<I,d>::as_string() const {
  if (strct==nullptr) {
    return string("indexstructure<zip>");
  } else {
    return format("indexstructure<{}><<{}>>",strct->type_as_string(),strct->as_string());
  }
};

/****************
 ****************
 **************** ioperator
 ****************
 ****************/

/*!
  ioperator construction from a character string is good for shortcuts
  - "none" is the identity mapping
  - ">>1" is rightshift by one, using modulo. This does not make sense without more information, but that is provided by the distribution
  - "<<1" leftshift; same
  - ">=1" rightshift, except that no modulo wrapping is used: the upper bound provided by the distributioni will not be exceeded
  - "*2" multiplication
  - "x2" also multiplication; the difference becomes apparent when we operate on indexstruct objects, then the former stretches the range, where as the latter multiplies both bounds and stride
  - "/2" division
  - ":2" division, but meant for multiple contiguous: instead of "last /= 2", we do "last = (last+stride)/2-newstride"
*/
template<typename I,int d>
ioperator<I,d>::ioperator( string op ) {
  if ( op==string("none") || op==string("no_op") ) {
    type = iop_type::NONE;
  } else if ( op[0]=='>' || op[0]=='<' ) {
    type = iop_type::SHIFT_REL;
    mod = op[1]!='=';
    by = constant_coordinate<I,d>( std::stoi(string(op.begin()+2,op.end())) );
    if (op[0]=='<') by = -by;
  } else if (op[0]=='*') {
    type = iop_type::MULT;
  } else if (op[0]=='x') { //throw(std::string("base mult temporarily disabled"));
    type = iop_type::MULT; baseop = 1;
  } else if (op[0]=='/') {
    type = iop_type::DIV;
  } else if (op[0]==':') {
    type = iop_type::CONTDIV;
  } else {
    print("unparsable iop\n");
    throw(string("Can not parse operator"));
  }
  if (type==iop_type::MULT || type==iop_type::DIV || type==iop_type::CONTDIV)
    by = constant_coordinate<I,d>( std::stoi(string(op.begin()+1,op.end())) );
};

/*!
  ioperator constructor from keyword and amount. Currently implemented:
  - "shift" : relative shift
  - "shiftto" : shift to specific point
*/
template<typename I,int d>
ioperator<I,d>::ioperator( string op,const coordinate<I,d>& amt ) {
  if ( op=="shift" ) {
    type = iop_type::SHIFT_REL; by = amt;
  } else if ( op=="shiftto" ) {
    type = iop_type::SHIFT_ABS; by = amt;
  } else {
    throw(string("Can not parse ioperator")); }
};

/*!
  Operate on any legitimate index. We do not at this point have a context for judging
  the result to be valid. 
 */
template<typename I,int d>
coordinate<I,d> ioperator<I,d>::operate( const coordinate<I,d>& i ) const {
  if (is_none_op())
    return i;
  else if (is_shift_op()) {
    coordinate<I,d> r = i+by;
    return r; // if (!is_modulo_op() && r<0) return 0; else 
  } else if (is_mult_op())
    return i*by;
  else if (is_div_op() || is_contdiv_op())
    return i/by;
  else if (is_function_op()) {
    auto r(i);
    for (int id=0; id<d; id++)
      r.data()[id] = func( r.data()[id] );
    return r;
  } else
    throw( string("unknown operate type for operate") );
};

//! Render `ioperator' as string
template<typename I,int d>
string ioperator<I,d>::as_string() const {
  if (is_none_op())
    return std::string("Id");
  else if (is_shift_op()) {
    if (by>=0)
      return fmt::format("+{}",by.as_string());
    else
      return fmt::format("-{}",by.as_string());
  } else if (is_mult_op()) {
    if (is_base_op())
      return fmt::format("x{}",by.as_string());
    else
      return fmt::format("*{}",by.as_string());
  } else if (is_div_op() || is_contdiv_op())
    return fmt::format("/{}",by.as_string());
  else if (is_function_op())
    return std::string("func");
  else
    throw(std::string("unknown operate type for string"));
};

/*!
  Operate on a scalar, and bump or wrap as needed.
*/
template<typename I,int d>
coordinate<I,d> ioperator<I,d>::operate( const coordinate<I,d>& i, const coordinate<I,d>& m ) const {
  auto r = operate(i);
  if (is_modulo_op()) {
    return coordmod( r,m+1 );
  } else {
    return coordmax( constant_coordinate<I,d>(0), coordmin(m,r) );
  }
};

//! Inverse operate on a scalar
template<typename I,int d>
coordinate<I,d> ioperator<I,d>::inverse_operate( const coordinate<I,d>& i ) const {
  if (i<0) {
    throw(std::string("I don't like to operate negative indices")); }
  if (is_none_op())
    return i;
  else if (is_left_shift_op())
    return i+1;
  else if (is_right_shift_op()) {
    if (i==0) {
      throw(std::string("Can't unrightshift zero"));
    } else return coordmax( constant_coordinate<I,d>(0),i-1);
  }
  throw(std::string("unknown operate type for inverse"));
};

template<typename I,int d>
coordinate<I,d> ioperator<I,d>::inverse_operate
    ( const coordinate<I,d>& i, const coordinate<I,d>& m ) const {
  throw("inverse operator not implemented");
#if 0
  if (i<0 || i>=m) {
    throw(std::string("index out of range")); }
  if (is_none_op())
    return i;
  else {
    auto ii=-1;
    if (is_left_shift_op()) {
      if (i==m-1) {
	if (is_modulo_op())
	  return coordinate<I,d>(0);
	else {
	  throw(std::string("Can not unleftshift last index")); }
      } else ii = i+coordinate<I,d>(1);
    } else if (is_right_shift_op()) {
      if (i==0) {
	if (is_modulo_op())
	  return m-coordinate<I,d>(1);
	else {
	  throw(std::string("Can not unrightshift zero")); }
      } else ii = i-coordinate<I,d>(1);
    }
    if (is_modulo_op())
      return coordmod( ii,m );
    else
      return coordmax( coordinate<I,d>(0), coordmin(m-1,ii) );
  }
#endif
};

/****
 **** Operate on indexstructs
 ****/

//! \todo lose a bunch of clones
template<typename I,int d>
shared_ptr<indexstruct<I,d>> strided_indexstruct<I,d>::operate( const ioperator<I,d>& op,bool trace ) const {
  shared_ptr<indexstruct<I,d>> operated{nullptr};
  if (op.is_none_op()) {
    return this->make_clone();
  } else if (op.is_shift_to()) {
    throw("shift to not implemented");
    // I amt = op.amount()-first;
    // ioperator shift = shift_operator<I,d>(amt);
    // operated = this->operate(shift);
  } else if (op.is_shift_op()) {
    auto
      f = op.operate(first_index()),
      l = f+(last_actual_index()-first_index());
    if (stride()==1)
      operated = shared_ptr<indexstruct<I,d>>
	{ make_shared<contiguous_indexstruct<I,d>>(f,l) };
    else
      operated = shared_ptr<indexstruct<I,d>>
	{ make_shared<strided_indexstruct<I,d>>(f,l,stride()) };
  } else if (op.is_mult_op() || op.is_div_op() || op.is_contdiv_op()) {
    if (trace) print("mult type op");
    coordinate<I,d>
      new_first  = op.operate(first_index()),
      new_last;
    coordinate<I,d> new_stride;
    if (op.is_base_op())
      new_stride  = stride();
    else
      new_stride = coordmax( constant_coordinate<I,d>(1),op.operate(stride()) );
    if (op.is_contdiv_op())
      new_last = coordmax<I,d>( new_first, op.operate(last_actual_index()+stride())-new_stride );
    else if (op.is_base_op())
      new_last = op.operate(last_actual_index()+1)-1;
    else {
      new_last = op.operate(last_actual_index());
      if (trace)
	print(", mult last={} gives {}",
	      last_actual_index().as_string(),new_last.as_string()); // formatter broken
    }
    if (new_stride==1)
      operated = shared_ptr<indexstruct<I,d>>
	{ make_shared<contiguous_indexstruct<I,d>>(new_first,new_last) };
    else
      operated = shared_ptr<indexstruct<I,d>>
	{ make_shared<strided_indexstruct<I,d>>(new_first,new_last,new_stride) };
  } else if (op.is_function_op()) {
    auto rstruct = shared_ptr<indexstruct<I,d>>{ make_shared<indexed_indexstruct<I,d>>() };
    //for (auto i : *this) { VLE `this' is modified, which contradicts the const
    for (I ii=0; ii<volume(); ii++) {
      auto i = get_ith_element(ii);
      rstruct = rstruct->add_element( op.operate(i) );
    }
    operated = rstruct;
  } else throw(std::string("Can not operate contiguous/strided struct"));
  if (operated==nullptr)
    throw(std::string("Strided indexstruct operate: missing case"));
  if (trace) print("\n");
  return operated;
};

/*!
  We apply a sigma operator by taking the union over all indices.
*/
template<typename I,int d>
shared_ptr<indexstruct<I,d>> strided_indexstruct<I,d>::operate( const sigma_operator<I,d> &op ) const {
  if (op.is_struct_operator()) {
    return op.struct_apply(*this);
  } else if (op.is_point_operator()) {
    auto pop = op.point_operator();
    return operate(pop);
  } else {
    auto rstruct = shared_ptr<indexstruct<I,d>>{ make_shared<empty_indexstruct<I,d>>() };
    //for ( auto i : *this ) {
    throw( "operate on strided");
    // for ( auto i=first_index(); i<=last_index(); i+=stride() ) {
    //   rstruct = rstruct->struct_union( op.operate(i) );
    // }
    return rstruct;
  }
};

template<typename I,int d>
shared_ptr<indexstruct<I,d>> indexed_indexstruct<I,d>::operate( const ioperator<I,d>& op,bool trace ) const {
  auto shifted = shared_ptr<indexstruct<I,d>>( make_shared<indexed_indexstruct<I,d>>() );
  shifted->reserve(this->volume());
  if (op.is_shift_op()) {
    for (I i=0; i<indices.size(); i++)
      shifted->add_element( op.operate( indices[i] ) );
    return shifted;
  } else throw(std::string("operation not defined on indexed"));
};

//! \todo just use the result of operate?
template<typename I,int d>
shared_ptr<indexstruct<I,d>> composite_indexstruct<I,d>::operate( const ioperator<I,d>& op,bool trace ) const {
  auto rstruct = shared_ptr<composite_indexstruct>{ make_shared<composite_indexstruct<I,d>>() };
  for (auto s : structs)
    rstruct->push_back( s->operate(op)->make_clone() );
  return rstruct;
};

/****
 **** Operators
 ****/

template<typename I,int d>
shared_ptr<indexstruct<I,d>> sigma_operator<I,d>::operate( coordinate<I,d> i) const {
  if (lambda_s)
    throw(std::string("Can not operate on point: only defined for structs"));
  if (is_point_operator()) {
    return shared_ptr<indexstruct<I,d>>{make_shared<contiguous_indexstruct<I,d>>( point_func.operate(i) )};
  } else {
    return func(i);
  }
};

/*! \todo find occurrences of the lambda_i case. can they be done by storing an ioperator?
  \todo print("sigma by point operator is dangerous\n");
*/
template<typename I,int d>
shared_ptr<indexstruct<I,d>> sigma_operator<I,d>::operate( shared_ptr<indexstruct<I,d>> i) const {
  if (lambda_s) {
    try {
      return sfunc(*i);
    } catch (std::string c) {
      throw(fmt::format("Error <<{}>> applying structure sigma",c));
    };
  } else if (lambda_p) {
    return shared_ptr<indexstruct<I,d>>
      { make_shared<contiguous_indexstruct<I,d>>
	( point_func.operate(i->first_index()),
	  point_func.operate(i->last_actual_index()) ) };
  } else if (lambda_i) {
    //print("sigma by point operator is dangerous\n");
    return shared_ptr<indexstruct<I,d>>
      { make_shared<contiguous_indexstruct<I,d>>
	( func(i->first_index())->first_index(),
	  func(i->last_actual_index())->last_actual_index() ) };
  } else {
    throw(std::string("sigma::operate(struct) weird case"));
  }
};

template<typename I,int d>
string sigma_operator<I,d>::as_string() const {
  memory_buffer w;
  format_to(w.end(),"Sigma operator");
  if (is_point_operator())
    format_to(w.end()," from ioperator \"{}\"",point_func.as_string());
  return to_string(w);
};

template struct fmt::formatter<shared_ptr<indexstruct<int,1>>>;
template struct fmt::formatter<shared_ptr<indexstruct<index_int,1>>>;
template struct fmt::formatter<shared_ptr<indexstruct<int,2>>>;
template struct fmt::formatter<shared_ptr<indexstruct<index_int,2>>>;
template struct fmt::formatter<shared_ptr<indexstruct<int,3>>>;
template struct fmt::formatter<shared_ptr<indexstruct<index_int,3>>>;

template class contiguous_indexstruct<int,1>;
template class contiguous_indexstruct<index_int,1>;
template class strided_indexstruct<int,1>;
template class strided_indexstruct<index_int,1>;
template class indexed_indexstruct<int,1>;
template class indexed_indexstruct<index_int,1>;

template class contiguous_indexstruct<int,2>;
template class contiguous_indexstruct<index_int,2>;
template class strided_indexstruct<int,2>;
template class strided_indexstruct<index_int,2>;
template class indexed_indexstruct<int,2>;
template class indexed_indexstruct<index_int,2>;

template class contiguous_indexstruct<int,3>;
template class contiguous_indexstruct<index_int,3>;
template class strided_indexstruct<int,3>;
template class strided_indexstruct<index_int,3>;
template class indexed_indexstruct<int,3>;
template class indexed_indexstruct<index_int,3>;

template class indexstructure<int,1>;
template class indexstructure<index_int,1>;
template class indexstructure<int,2>;
template class indexstructure<index_int,2>;
template class indexstructure<int,3>;
template class indexstructure<index_int,3>;

template class ioperator<int,1>;
template class ioperator<index_int,1>;
template class shift_operator<int,1>;
template class shift_operator<index_int,1>;
template class sigma_operator<int,1>;
template class sigma_operator<index_int,1>;

template class ioperator<int,2>;
template class ioperator<index_int,2>;
template class shift_operator<int,2>;
template class shift_operator<index_int,2>;
template class sigma_operator<int,2>;
template class sigma_operator<index_int,2>;

template class ioperator<int,3>;
template class ioperator<index_int,3>;
template class shift_operator<int,3>;
template class shift_operator<index_int,3>;
template class sigma_operator<int,3>;
template class sigma_operator<index_int,3>;
