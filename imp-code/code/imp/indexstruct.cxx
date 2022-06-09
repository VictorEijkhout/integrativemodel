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

#include "coord.hpp"
#include "indexstruct.hpp"

using fmt::format;
using fmt::print;
using fmt::format_to;
using fmt::to_string;

using std::function;
using std::move;
using std::make_shared;
using std::shared_ptr;

using std::string;

using std::vector;

/****************
 ****************
 **************** structs
 ****************
 ****************/

template<int d>
int indexstruct<d>::type_as_int() const {
  if (is_empty()) return 1;
  else if (is_contiguous()) return 2;
  else if (is_strided()) return 3;
  else if (is_indexed()) return 4;
  else if (is_composite()) return 5;
  else
    throw(format("Type can not be converted to int"));
};

template<int d>
void indexstruct<d>::report_unimplemented( const char *c ) const {
  if (!is_known())
    throw(fmt::format("Trying to use query <<{}>> on undefined indexstruct",c));
  else
    throw(fmt::format("Routine {} not implemented for type {}",c,type_as_string()));
};

template<int d>
bool indexstruct<d>::equals( std::shared_ptr<indexstruct<d>> idx ) const {
  throw(fmt::format("Equals not implemented for <<{}>> and <<{}>>",
		    type_as_string(),idx->type_as_string()));
};

//! Operate, then cut to let lo/hi be the lowest/highest index, inclusive.
template<int d>
shared_ptr<indexstruct<d>> indexstruct<d>::operate
    ( const ioperator &op, index_int lo,index_int hi ) const {
  shared_ptr<indexstruct<d>>
    noleft = this->operate(op)->truncate_left(lo),
    noright = noleft->truncate_right(hi);
  return noright;
};

template<int d>
bool indexstruct<d>::contains_element_in_range(index_int idx) const {
  return idx>=first_index() && idx<=last_index();
};

//! Base test for disjointness; derived classes will build on this.
template<int d>
bool indexstruct<d>::disjoint( shared_ptr<indexstruct<d>> idx ) {
  return first_index()>idx->last_index() || last_index()<idx->first_index();
};

template<int d>
shared_ptr<indexstruct<d>> indexstruct<d>::operate( const sigma_operator &op, index_int lo,index_int hi ) const {
  shared_ptr<indexstruct<d>>
    noleft = this->operate(op)->truncate_left(lo),
    noright = noleft->truncate_right(hi);
  return noright;
};

//! Operate, then cut boundaries to fit within `outer'.
template<int d>
shared_ptr<indexstruct<d>> indexstruct<d>::operate
    ( const ioperator &op,shared_ptr<indexstruct<d>> outer ) const {
  index_int lo = outer->first_index(), hi = outer->last_index();
  return this->operate(op,lo,hi);
};

//! Operate, then cut boundaries to fit within `outer'.
template<int d>
shared_ptr<indexstruct<d>> indexstruct<d>::operate
    ( const sigma_operator &op,shared_ptr<indexstruct<d>> outer ) const {
  index_int lo = outer->first_index(), hi = outer->last_index();
  return this->operate(op,lo,hi);
};

//! Let `trunc' be the first index in the truncated struct.
template<int d>
shared_ptr<indexstruct<d>> indexstruct<d>::truncate_left( index_int trunc ) {
  if (trunc<=first_index()) {
    return this->make_clone();
    //return shared_ptr<indexstruct<d>>( this->make_clone() ); //shared_from_this();
  } else {
    auto truncated = this->minus
      ( shared_ptr<indexstruct<d>>
	( new contiguous_indexstruct(this->first_index(),trunc-1) ) );
    return truncated;
  }
};

//! Let `trunc' be the last index in the truncated struct.
template<int d>
shared_ptr<indexstruct<d>> indexstruct<d>::truncate_right( index_int trunc ) {
  if (trunc>=last_index()) {
    return this->make_clone(); //shared_from_this();
    //return shared_ptr<indexstruct<d>>( this->make_clone() ); //shared_from_this();
  } else {
    auto truncated = this->minus
      ( shared_ptr<indexstruct<d>>
	( new contiguous_indexstruct(trunc+1,this->last_index()) ) );
    return truncated;
  }
};

/****
 **** Empty indexstruct
 ****/
template<int d>
shared_ptr<indexstruct<d>> empty_indexstruct<d>::add_element( const index_int idx ) {
  //  return shared_ptr<indexstruct<d>>( new indexed_indexstruct(1,&idx) );
  return shared_ptr<indexstruct<d>>( make_shared<indexed_indexstruct>(1,&idx) );
};

template<int d>
shared_ptr<indexstruct<d>> empty_indexstruct<d>::struct_union
    ( shared_ptr<indexstruct<d>> idx ) {
  return idx;
};

/****
 **** Strided indexstruct
 ****/

template<int d>
bool strided_indexstruct<d>::equals( shared_ptr<indexstruct<d>> idx ) const {
  if (idx->is_empty())
    return false;
  strided_indexstruct *strided = dynamic_cast<strided_indexstruct*>(idx.get());
  if (strided!=nullptr)
    return first==strided->first_index() && last==strided->last_index()
      && stride_amount==strided->stride();
  else {
    auto simple = idx->force_simplify();
    if (simple->is_strided())
      return equals(simple);
    else return false;
  }
};

template<int d>
shared_ptr<indexstruct<d>> strided_indexstruct<d>::add_element( const index_int idx ) {
  if (contains_element(idx))
    return this->make_clone(); //shared_from_this();
  else if (idx==first-stride_amount) {
    first = idx; return  this->make_clone();
  } else if (idx==last+stride_amount) {
    last = idx;
    return this->make_clone();
  } else {
    auto indexed = shared_ptr<indexstruct<d>>( make_shared<indexed_indexstruct>(this) );
    return indexed->add_element(idx);
  }
};

template<int d>
bool strided_indexstruct<d>::contains_element( index_int idx ) const {
  return idx>=first && idx<=last && (idx-first)%stride_amount==0;
};

template<int d>
index_int strided_indexstruct<d>::find( index_int idx ) const {
  if (!contains_element(idx))
    throw(fmt::format("Index {} to find is out of range <<{}>>",idx,this->as_string()));
  return (idx-first)/stride_amount;
};

template<int d>
index_int strided_indexstruct<d>::get_ith_element( const index_int i ) const {
  if (i<0 || i>=local_size())
    throw(fmt::format("Index {} out of range for {}",i,as_string()));
  return first+i*stride_amount;
};

template<int d>
bool strided_indexstruct<d>::contains( shared_ptr<indexstruct<d>> idx ) const {
  if (idx->local_size()==0) return true;
  if (this->local_size()==0) return false;
  if (idx->local_size()==1)
    return contains_element( idx->first_index() );

  /*
   * Case : contains other strided
   */
  strided_indexstruct *strided = dynamic_cast<strided_indexstruct*>(idx.get());
  if (strided!=nullptr)
    return first<=strided->first && last>=strided->last
      && (strided->first-first)%stride_amount==0 && strided->stride_amount%stride_amount==0;

  /*
   * Case: contains indexed
   */
  indexed_indexstruct *indexed = dynamic_cast<indexed_indexstruct*>(idx.get());
  if (indexed!=nullptr) { // strided & indexed
    //auto nonconst = make_clone();
    return contains_element(indexed->first_index())
      && contains_element(indexed->last_index());
  }

  /*
   * Case: contains composite
   */
  composite_indexstruct *composite = dynamic_cast<composite_indexstruct*>(idx.get());
  if (composite!=nullptr) { // strided & composite
    for ( auto s : composite->get_structs() )
      if (!contains(s))
	return false;
    return true;
  }

  throw(std::string("unimplemented strided contains case"));
};

/*! Disjointness test for strided.
  \todo bunch of unimplemented cases \todo unittest for indexed csae
*/
template<int d>
//bool strided_indexstruct<d>::disjoint( indexstruct *idx ) {
template<int d>
bool strided_indexstruct<d>::disjoint( shared_ptr<indexstruct<d>> idx ) {
  template<int d>
  bool range_disjoint = indexstruct<d>::disjoint(idx);
  if (range_disjoint==true)
    return true;
  else {
    // disjoint with other strided
    strided_indexstruct *strided = dynamic_cast<strided_indexstruct*>(idx.get());
    if (strided!=nullptr) {
      if (stride()==idx->stride())
	return first_index()%stride()!=idx->first_index()%stride();
      else
	throw(fmt::format("unimplemented disjoint for different strides"));
    }
    // disjoint from indexed
    indexed_indexstruct *indexed = dynamic_cast<indexed_indexstruct*>(idx.get());
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

template<int d>
bool strided_indexstruct<d>::has_intersect( shared_ptr<indexstruct<d>> idx ) {
  if (idx->is_empty())
    return false;
  if (contains(idx))
    return true;
  if (idx->contains(this->shared_from_this()))
    return true; 

  /*
   * Case : intersect with other strided
   */
  strided_indexstruct *strided = dynamic_cast<strided_indexstruct*>(idx.get());
  if (strided!=nullptr) { // strided & strided
    if (stride_amount==strided->stride_amount) {
      if (first%stride_amount!=strided->first%strided->stride_amount) {
	return false;
      } else {
	index_int mn = MAX(first,strided->first), mx = MIN(last,strided->last);
	if (mn>mx)
	  return false;
	else
	  return true;
      }
    } else if (idx->local_size()<local_size()) { // case of different strides, `this' s/b small
      return idx->has_intersect(this->shared_from_this());
    } else { // case of different strides
      return true;
    }
  }
  /*
   * Case: intersect with indexed => reverse
   */
  indexed_indexstruct *indexed = dynamic_cast<indexed_indexstruct*>(idx.get());
  if (indexed!=nullptr) { // strided & indexed
    return indexed->has_intersect(this->shared_from_this());
  }
  throw(fmt::format("Unimplemented has_intersect <<{}>> and <<{}>>",
		    this->as_string(),idx->as_string()));
};

template<int d>
shared_ptr<indexstruct<d>> strided_indexstruct<d>::translate_by( index_int shift ) {
  return shared_ptr<indexstruct<d>>{
    new strided_indexstruct(first+shift,last+shift,stride_amount) };
};

template<int d>
shared_ptr<indexstruct<d>> strided_indexstruct<d>::intersect
    ( shared_ptr<indexstruct<d>> idx ) {
  if (idx->is_empty())
    return shared_ptr<indexstruct<d>>( make_shared<empty_indexstruct>() );
  if (contains(idx))
    return idx->make_clone();
  if (idx->contains(this->shared_from_this()))
    return this->make_clone();
  if (first_index()>idx->last_index() || last_index()<idx->first_index() )
    return shared_ptr<indexstruct<d>>( make_shared<empty_indexstruct>() );

  /*
   * Case : intersect with other strided
   */
  strided_indexstruct *strided = dynamic_cast<strided_indexstruct*>(idx.get());
  if (strided!=nullptr) { // strided & strided
    if (stride_amount==strided->stride_amount) {
      if (first%stride_amount!=strided->first%strided->stride_amount) {
	return shared_ptr<indexstruct<d>>( make_shared<empty_indexstruct>() );
      } else {
	index_int mn = MAX(first,strided->first), mx = MIN(last,strided->last);
	if (mn>mx)
	  return shared_ptr<indexstruct<d>>( make_shared<empty_indexstruct>() );
	else if (stride_amount==1)
	  return shared_ptr<indexstruct<d>>( make_shared<contiguous_indexstruct>(mn,mx) );
	else
	  return shared_ptr<indexstruct<d>>
	    ( make_shared<strided_indexstruct>(mn,mx,stride_amount) );
      }
    } else if (idx->local_size()<local_size()) { // case of different strides, `this' s/b small
      return idx->intersect(this->shared_from_this());
    } else { // case of different strides
      auto rstruct = shared_ptr<indexstruct<d>>( make_shared<empty_indexstruct>() );
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
  indexed_indexstruct *indexed = dynamic_cast<indexed_indexstruct*>(idx.get());
  if (indexed!=nullptr) { // strided & indexed
    return indexed->make_clone()->intersect(this->shared_from_this());
  }

  /*
   * Case: intersect with composite => reverse
   */
  composite_indexstruct *composite = dynamic_cast<composite_indexstruct*>(idx.get());
  if (composite!=nullptr) { // strided & composite
    return composite->intersect(this->shared_from_this());
  }

  throw(fmt::format("Unimplemented intersect <<{}>> and <<{}>>",
		    this->as_string(),idx->as_string()));
};

//! \todo make unittesting of minus composite
//! \todo weird clone because of a const problem.
template<int d>
shared_ptr<indexstruct<d>> strided_indexstruct<d>::minus
    ( shared_ptr<indexstruct<d>> idx ) const {
  // easy cases
  if (idx->is_empty())
    return this->make_clone();
  //return shared_ptr<indexstruct<d>>( this->make_clone());
  if (idx->contains(this->make_clone())) // this->shared_from_this()))
    return shared_ptr<indexstruct<d>>( make_shared<empty_indexstruct>() );

  /*
   * Case: the idx set is in the interior
   */
  if (this->first_index()<idx->first_index() && this->last_index()>idx->last_index()) {
    shared_ptr<indexstruct<d>> left,right,mid;
    index_int 
      midfirst = idx->first_index(), midlast = idx->last_index();
    if (stride_amount==1) {
      left = shared_ptr<indexstruct<d>>
	( make_shared<contiguous_indexstruct>(this->first_index(),midfirst-1) );
      right = shared_ptr<indexstruct<d>>
	( make_shared<contiguous_indexstruct>(midlast+1,this->last_index()) );
    } else {
      // left
      left = shared_ptr<indexstruct<d>>
	( make_shared<strided_indexstruct>(this->first_index(),midfirst-1,stride_amount) );
      // right
      index_int rfirst = midlast+1;
      while (rfirst%stride_amount!=last%stride_amount) rfirst++;
      right = shared_ptr<indexstruct<d>>
	( make_shared<strided_indexstruct>(rfirst,this->last_index(),stride_amount) );
    }
    auto comp = left->struct_union(right);
    return comp;
  }

  /*
   * Case : minus contiguous. Easy.
   */
  contiguous_indexstruct *contiguous = dynamic_cast<contiguous_indexstruct*>(idx.get());
  if (contiguous!=nullptr) {
    index_int ifirst = contiguous->first_index(),ilast = contiguous->last_index();
    shared_ptr<indexstruct<d>> contmin;
    if (ilast<first || ifirst>last) // disjoint 
      return this->make_clone(); //shared_from_this();
    else if (ifirst<=first) { // cut left part
      if (stride_amount==1)
	contmin = shared_ptr<indexstruct<d>>{
					  make_shared<contiguous_indexstruct>( ilast+stride_amount,last ) };
      else
	contmin = shared_ptr<indexstruct<d>>{
					  make_shared<strided_indexstruct>( ilast+stride_amount,last,stride_amount ) };
    } else { // cut right part
      if (stride_amount==1)
	contmin = shared_ptr<indexstruct<d>>{
					  make_shared<contiguous_indexstruct>( first,ifirst-1/*stride_amount*/ ) };
      else
	contmin = shared_ptr<indexstruct<d>>{
					  make_shared<strided_indexstruct>( first,ifirst-1/*stride_amount*/,stride_amount ) };
    }
    return contmin;
  }
  
  /*
   * Case : minus other strided
   */
  strided_indexstruct *strided = dynamic_cast<strided_indexstruct*>(idx.get());
  if (strided!=nullptr) { // strided & strided
    if (stride_amount==strided->stride_amount) {
      shared_ptr<indexstruct<d>> stridmin{nullptr};
      if (first%stride_amount!=strided->first%strided->stride_amount) {
	return this->make_clone(); // interleaving case
      } else {
	if (contains_element(idx->first_index()) && contains_element(idx->last_index()))
	  throw(std::string("should yield composite; unimplemented"));
	index_int f = idx->first_index(), l = idx->last_index();
	if (f<=first) { // cut on the left
	  f = MAX(l+1,first); l = last;
	} else { // cut on the right
	  l = MIN(f-1,last); f = first;
	}
	if (stride_amount==1)
	  stridmin = shared_ptr<indexstruct<d>>{ make_shared<contiguous_indexstruct>(f,l) };
	else
	  stridmin = shared_ptr<indexstruct<d>>{ make_shared<strided_indexstruct>(f,l,stride_amount) };
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
  indexed_indexstruct *indexed = dynamic_cast<indexed_indexstruct*>(idx.get());
  if (indexed!=nullptr) { // strided & indexed
    auto
      this_index = this->convert_to_indexed();
    indexed_indexstruct *test_index = dynamic_cast<indexed_indexstruct*>(this_index.get());
    if (test_index==nullptr) // remove this test after a while
      throw(fmt::format("failed conversion of <<{}>> to indexed",this->as_string()));
    return this_index->minus(idx); //(shared_ptr<indexstruct<d>>(indexed));
  }

  /*
   * Minus composite
   */
  composite_indexstruct *composite = dynamic_cast<composite_indexstruct*>(idx.get());
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
template<int d>
shared_ptr<indexstruct<d>> strided_indexstruct<d>::relativize_to
    ( shared_ptr<indexstruct<d>> idx,bool trace ) {
  if (trace)
    print("attempt to relativize {} to {}\n",this->as_string(),idx->as_string());
  if (!idx->contains(this->shared_from_this())) {
    if (trace) print("Not contained; throwing\n");
    throw(fmt::format("Need containment for relativize {} to {}",
                      this->as_string(),idx->as_string()));
  }
  if (local_size()==1)
    return shared_ptr<indexstruct<d>>{
      make_shared<contiguous_indexstruct>( idx->find( first_index() ) ) };

  /*
   * Case : relativize against other strided
   */
  strided_indexstruct *strided = dynamic_cast<strided_indexstruct*>(idx.get());
  if (strided!=nullptr) { // strided & strided
    if (stride_amount==strided->stride_amount) { // same stride
      if (first%stride_amount!=strided->first%strided->stride_amount) { // interleaved
        return shared_ptr<indexstruct<d>>{ make_shared<empty_indexstruct>() };
      } else {
        index_int mn = MAX(first,strided->first), mx = MIN(last,strided->last);
        if (mn>mx)
	  return shared_ptr<indexstruct<d>>{ make_shared<empty_indexstruct>() };
        else {
          return shared_ptr<indexstruct<d>>{ make_shared<contiguous_indexstruct
						      >((mn-strided->first)/stride_amount,(mx-strided->first)/stride_amount) };
        }
      }
    }
  }

  /*
   * Case : relativize against indexed
   */
  indexed_indexstruct *indexed = dynamic_cast<indexed_indexstruct*>(idx.get());
  if (indexed!=nullptr) {
    auto relext = shared_ptr<indexstruct<d>>( make_shared<indexed_indexstruct>() );
    for (auto i : *this) {
      relext->addin_element( idx->find(i) );
    }
    return relext;
  }

  /*
   * Case : relativize against composite
   */
  composite_indexstruct *composite = dynamic_cast<composite_indexstruct*>(idx.get());
  if (composite!=nullptr) {
    // easy case: contained in one member of the composite
    auto structs = composite->get_structs(); int found_s{-1}; index_int shift{0};
    for (int is=0; is<structs.size(); is++) {
      auto istruct = structs.at(is);
      if (istruct->contains(idx)) {
	auto rstruct = relativize_to(istruct);
	return rstruct->operate( shift_operator(shift) );
      }
      shift += istruct->local_size();
    }
    // general case
    auto rel = shared_ptr<indexstruct<d>>( make_shared<empty_indexstruct>() );
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
template<int d>
shared_ptr<indexstruct<d>> strided_indexstruct<d>::convert_to_indexed() const {
  auto indexed = shared_ptr<indexstruct<d>>( make_shared<indexed_indexstruct>() );
  indexed->reserve( local_size() );
  auto this_copy = *this;
  for (auto v : this_copy)
    indexed = indexed->add_element(v);
  return indexed;
};

template<int d>
shared_ptr<indexstruct<d>> strided_indexstruct<d>::struct_union
    ( shared_ptr<indexstruct<d>> idx ) {
  if (contains(idx))
    return this->shared_from_this();
  else if (idx->contains(this->shared_from_this()))
    return idx;
  
  /*
   * Case : union with other strided
   */
  strided_indexstruct *strided = dynamic_cast<strided_indexstruct*>(idx.get());
  if (strided!=nullptr) {
    int amt = idx->stride();
    index_int frst = idx->first_index(), lst = idx->last_index();
    if (stride_amount==amt && first%stride_amount==frst%amt &&
	frst<=last+stride_amount && lst>=first-stride_amount) {
      index_int mn = MIN(first,frst), mx = MAX(last,lst);
      if (stride_amount==1)
	return shared_ptr<indexstruct<d>>{make_shared<contiguous_indexstruct>(mn,mx)};
      else
	return shared_ptr<indexstruct<d>>{make_shared<strided_indexstruct>(mn,mx,stride_amount)};
    } else if (local_size()<5 && idx->local_size()<5) {
      auto indexed = convert_to_indexed();
      return indexed->struct_union( idx->convert_to_indexed() );
    } else {
      auto composite = make_shared<composite_indexstruct>();
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
  indexed_indexstruct *indexed = dynamic_cast<indexed_indexstruct*>(idx.get());
  if (indexed!=nullptr) {
    auto extstrided = this->make_clone();
    for (auto v : *indexed) {
      if (extstrided->contains_element(v))
	continue;
      else if (extstrided->can_incorporate(v))
	extstrided = extstrided->add_element(v);
      
      else { // an index struct can incorporate arbitrary crap
	//print("extending strided {} by indexed {}\n",this->as_string(),indexed->as_string());
	auto indexplus = indexed->struct_union(this->shared_from_this());
	//print("  extended : {}\n",indexplus->as_string());
	indexplus = indexplus->force_simplify();
	//print("  simplified : {}\n",indexplus->as_string());
	return indexplus;
      }
    }
    return shared_ptr<indexstruct<d>>( extstrided->force_simplify() );
  }

  /*
   * Case : union with composite
   */
  composite_indexstruct *composite = dynamic_cast<composite_indexstruct*>(idx.get());
  if (composite!=nullptr) {
    return idx->struct_union(this->shared_from_this());
  }

  throw(fmt::format("Unimplemented union: {} & {}",
		    type_as_string(),idx->type_as_string()));
};

//! Split along the start/end points of a second indexstruct
template<int d>
shared_ptr<indexstruct<d>> strided_indexstruct<d>::split
    ( shared_ptr<indexstruct<d>> idx ) {
  int s = stride();
  auto res = make_shared<composite_indexstruct>();
  auto f = first_index(), ff = idx->first_index(), l = last_index(), ll = idx->last_index();
  shared_ptr<indexstruct<d>> s1,s2;
  if (contains_element_in_range(ff)) {
    s1 = shared_ptr<indexstruct<d>>(make_shared<contiguous_indexstruct>(f,ff-1));
    s2 = shared_ptr<indexstruct<d>>(make_shared<contiguous_indexstruct>(ff,l));
  } else if (contains_element_in_range(ll)) {
    s1 = shared_ptr<indexstruct<d>>(make_shared<contiguous_indexstruct>(f,ll));
    s2 = shared_ptr<indexstruct<d>>(make_shared<contiguous_indexstruct>(ll+1,l));
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
template<int d>
indexed_indexstruct<d>::indexed_indexstruct( const index_int len,const index_int *idxs ) {
  indices.reserve(len);
  index_int iold;
  for (int i=0; i<len; i++) { index_int inew = idxs[i];
    if (i>0 && inew<=iold) throw(std::string("Only construct from sorted array"));
    indices.push_back(inew);
    iold = inew;
  }
};

/*!
  Add an element to an indexed indexstruct. We don't simplify: that
  can be done through \ref force_simplify.
*/
template<int d>
shared_ptr<indexstruct<d>> indexed_indexstruct<d>::add_element( const index_int idx ) {
  if (indices.size()==0 || idx>indices.at(indices.size()-1)) {
    indices.push_back(idx);
  } else {
    // this loop can not be ranged....
    for (auto it=indices.begin(); it!=indices.end(); ++it) {
      if (*it==idx) break;
      else if (*it>idx) { indices.insert(it,idx); break; }
    }
  }
  return this->make_clone();
};

//! Add an element into this structure.
template<int d>
void indexed_indexstruct<d>::addin_element( const index_int idx ) {
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

template<int d>
shared_ptr<indexstruct<d>> indexed_indexstruct<d>::translate_by( index_int shift ) {
  auto translated = shared_ptr<indexstruct<d>>{
    new indexed_indexstruct() };
  for (int loc=0; loc<indices.size(); loc++)
    translated->addin_element( indices.at(loc)+shift );
  return translated;
};

//! See if we can turn in indexed into contiguous, but cautiously.
template<int d>
shared_ptr<indexstruct<d>> indexed_indexstruct<d>::simplify() {
  if (local_size()>SMALLBLOCKSIZE)
    return force_simplify();
  else
    return this->make_clone();
};

/*!
  See if we can turn in indexed into contiguous.
*/
template<int d>
shared_ptr<indexstruct<d>> indexed_indexstruct<d>::force_simplify() const {
  try {
    if (0) {
    } else if (local_size()==0) {
      return shared_ptr<indexstruct<d>>( make_shared<empty_indexstruct>() );
    } else if (local_size()==outer_size()) {
      // easy simplification to contiguous
      return
	shared_ptr<indexstruct<d>>( make_shared<contiguous_indexstruct>(first_index(),last_index()) );
    } else {
      //    throw(string("arbitrary throw"));	
      int ileft = 0, iright = local_size()-1;
      // try detect strided
      int stride;
      if (is_strided_between_indices(ileft,iright,stride)) {
	index_int first = get_ith_element(ileft), last = get_ith_element(iright);
	//print("detecting stride {} between {}-{}\n",stride,first,last);
	if (stride==1)
	  return shared_ptr<indexstruct<d>>( make_shared<contiguous_indexstruct>(first,last) );
	else
	  return shared_ptr<indexstruct<d>>( make_shared<strided_indexstruct>(first,last,stride) );
      } else {
	// find strided subsections
	index_int first = get_ith_element(ileft), last = get_ith_element(iright);
	for (int find_left=ileft; find_left<iright; find_left++) {
	  int top_right = iright; if (find_left==ileft) top_right--;
	  for (int find_right=top_right; find_right>find_left+1; find_right--) {
	    // if we find one section, we replace it by a 3-composite. ultimately recursive?
	    if (is_strided_between_indices(find_left,find_right,stride)) {
	      // print("{}: found strided stretch {}-{}\n",
	      // 	       this->as_string(),find_left,find_right);
	      auto comp = shared_ptr<composite_indexstruct>( make_shared<composite_indexstruct>() );
	      index_int found_left = get_ith_element(find_left),
		found_right = get_ith_element(find_right);
	      if (find_left>ileft) { // there is stuff to the left
		index_int left = get_ith_element(find_left);
		auto left_part = this->minus
		  ( shared_ptr<indexstruct<d>>(make_shared<contiguous_indexstruct>(left,last)) );
		//print("composing with left: {}\n",left_part->as_string());
		comp->push_back(left_part);
		//print(" .. gives {}\n",comp->as_string());			 
	      } 
	      shared_ptr<indexstruct<d>> strided;
	      if (stride==1)
		strided = shared_ptr<indexstruct<d>>
		  ( make_shared<contiguous_indexstruct>(found_left,found_right) );
	      else
		strided = shared_ptr<indexstruct<d>>
		  ( make_shared<strided_indexstruct>(found_left,found_right,stride) );
	      //print("with middle: {}\n",strided->as_string());
	      comp->push_back(strided);
	      //print(" .. gives {}\n",comp->as_string());
	      if (find_right<iright) { // there is stuff to the right
		index_int right = get_ith_element(find_right);
		auto right_part = this->minus
		  ( shared_ptr<indexstruct<d>>(make_shared<contiguous_indexstruct>(first,right)) );
		right_part = right_part->force_simplify();
		//print("composing with right: {}\n",right_part->as_string());
		if (right_part->is_composite()) {
		  composite_indexstruct *new_right =
		    dynamic_cast<composite_indexstruct*>(right_part.get());
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
template<int d>
bool indexed_indexstruct<d>::is_strided_between_indices(int ileft,int iright,int &stride) const {
  index_int first = get_ith_element(ileft), last = get_ith_element(iright),
    n_index = iright-ileft+1;
  if (n_index==1)
    throw(fmt::format("Single point should have been caught"));
  // if this is strided, what would the stride be?
  stride = (last-first)/(n_index-1);
  // let's see if the bounds are proper for the stride
  if (first+(n_index-1)*stride!=last)
    return false;
  // test if everything in between is also strided
  for (int inext=ileft+1; inext<iright; inext++) {
    index_int elt = get_ith_element(inext);
    //print("elt {} : {}, testing with stride {}\n",inext,elt,stride);
    if ( (elt-first)%stride!=0 )
      return false;
  }
  return true; // iright>ileft+1; // should be at least 3 elements
};

template<int d>
index_int indexed_indexstruct<d>::get_ith_element( const index_int i ) const {
  if (i<0 || i>=local_size())
    throw(fmt::format("Index {} out of range for {}",i,as_string()));
  return indices[i];
};

template<int d>
shared_ptr<indexstruct<d>> indexed_indexstruct<d>::intersect
    ( shared_ptr<indexstruct<d>> idx ) {
  /*
   * Case : intersect with strided
   */
  strided_indexstruct *strided = dynamic_cast<strided_indexstruct*>(idx.get());
  if (strided!=nullptr) { // indexed & strided
    auto limited = shared_ptr<indexstruct<d>>( make_shared<indexed_indexstruct>() );
    index_int first = strided->first_index(), last = strided->last_index();
    for (auto v : indices)
      if (v>=first && v<=last)
    	limited = limited->add_element(v);
    return limited;
  }
  /*
   * Case: intersect with indexed => reverse
   */
  indexed_indexstruct *indexed = dynamic_cast<indexed_indexstruct*>(idx.get());
  if (indexed!=nullptr) { // strided & indexed
    auto limited = shared_ptr<indexstruct<d>>( make_shared<indexed_indexstruct>() );
    for (auto v : indices)
      if (indexed->contains_element(v))
	limited = limited->add_element(v);
    return limited;
  }
  throw(std::string("Unimplemented intersect with indexed"));
};

//! \todo lose those clones
template<int d>
bool indexed_indexstruct<d>::contains( shared_ptr<indexstruct<d>> idx ) const {
  if (idx->local_size()==0) return true;
  if (this->local_size()==0) return false;

  /*
   * Case : contains strided
   */
  strided_indexstruct *strided = dynamic_cast<strided_indexstruct*>(idx.get());
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
  indexed_indexstruct *indexed = dynamic_cast<indexed_indexstruct*>(idx.get());
  if (indexed!=nullptr) {
    auto nonconst = make_clone();
    for (auto v : indexed->indices)
      if (!nonconst->contains_element(v)) return false;
    return true;
  }

  /*
   * Case: contains composite
   */
  composite_indexstruct *composite = dynamic_cast<composite_indexstruct*>(idx.get());
  if (composite!=nullptr) {
    //    auto nonconst = make_clone();
    for (auto s : composite->get_structs())
      if (!/*nonconst->*/contains(s)) return false;
    return true;
  }
  
  throw(fmt::format("indexed contains case not implemented: {}",idx->type_as_string()));
};

//! \todo weird clone because of a const problem.
template<int d>
shared_ptr<indexstruct<d>> indexed_indexstruct<d>::minus
    ( shared_ptr<indexstruct<d>> idx ) const{
  if (idx->is_empty())
    return this->make_clone();
  if (idx->contains(this->make_clone())) //(this->shared_from_this()))
    return shared_ptr<indexstruct<d>>(make_shared<empty_indexstruct>());
    
  /*
   * Case : minus strided
   */
  strided_indexstruct *strided = dynamic_cast<strided_indexstruct*>(idx.get());
  if (strided!=nullptr) {
    auto indexminus = shared_ptr<indexstruct<d>>( make_shared<indexed_indexstruct>() );
    for (auto v : indices)
      if (!idx->contains_element(v))
	indexminus = indexminus->add_element(v);
    if (indexminus->local_size()==0)
      return shared_ptr<indexstruct<d>>(make_shared<empty_indexstruct>());
    else
      return indexminus;
  }
  /*
   * Case: minus other indexed
   */
  indexed_indexstruct *indexed = dynamic_cast<indexed_indexstruct*>(idx.get());
  if (indexed!=nullptr) {
    auto indexminus = shared_ptr<indexstruct<d>>( make_shared<indexed_indexstruct>() );
    for (auto v : indices)
      if (!idx->contains_element(v))
	indexminus = indexminus->add_element(v);
    if (indexminus->local_size()==0)
      return shared_ptr<indexstruct<d>>(make_shared<empty_indexstruct>());
    else
      return indexminus;
  }
  /*
   * Case: minus other composite
   */
  composite_indexstruct *composite = dynamic_cast<composite_indexstruct*>(idx.get());
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
template<int d>
bool indexed_indexstruct<d>::disjoint( shared_ptr<indexstruct<d>> idx ) {
  template<int d>
  //bool indexed_indexstruct<d>::disjoint( indexstruct *idx ) {
    template<int d>
  bool range_disjoint = indexstruct<d>::disjoint(idx);
  if (range_disjoint==true)
    return true;
  else {
    strided_indexstruct *strided = dynamic_cast<strided_indexstruct*>(idx.get());
    if (strided!=nullptr) {
      return strided->disjoint(this->shared_from_this());
    } else 
      throw(fmt::format("unimplemented disjoint: indexed & {}",idx->type_as_string()));
  }
};

//! \todo what do we need that nonconst for? equals and contains are const methods.
template<int d>
bool indexed_indexstruct<d>::equals( shared_ptr<indexstruct<d>> idx ) const {
  auto nonconst = make_clone();
  return nonconst->contains(idx) && idx->contains(nonconst);
};

template<int d>
shared_ptr<indexstruct<d>> indexed_indexstruct<d>::relativize_to
    ( shared_ptr<indexstruct<d>> idx,bool trace ) {
  if (!idx->contains(this->shared_from_this()))
    throw(fmt::format("Need containment for relativize {} to {}",
		      this->as_string(),idx->as_string()));

  /*
   * Case : relativize against other strided
   */
  strided_indexstruct *strided = dynamic_cast<strided_indexstruct*>(idx.get());
  if (strided!=nullptr) {
    index_int shift = strided->first_index();
    return shared_ptr<indexstruct<d>>{ this->translate_by( -shift ) };
  }

  /*
   * Case : relativize against indexed
   */
  indexed_indexstruct *indexed = dynamic_cast<indexed_indexstruct*>(idx.get());
  if (indexed!=nullptr) {
    auto relative = shared_ptr<indexstruct<d>>{ make_shared<indexed_indexstruct>() };
    int count = 0;
    for (auto v : indices)
      //relative = relative->add_element( count++ );
      relative = relative->add_element( indexed->find(v) );
    return relative;
  }

  /*
   * Case : relativize against composite
   */
  composite_indexstruct *composite = dynamic_cast<composite_indexstruct*>(idx.get());
  if (composite!=nullptr) {
    auto relative = shared_ptr<indexstruct<d>>{ make_shared<indexed_indexstruct>() };
    int count = 0;
    for (auto v : indices)
      relative = relative->add_element( composite->find(v) );
    return relative;
  }
  
  throw(fmt::format("Unimplemented indexed relativize to {}",idx->type_as_string()));
};

//! \todo can we lose that clone?
template<int d>
shared_ptr<indexstruct<d>> indexed_indexstruct<d>::struct_union
    ( shared_ptr<indexstruct<d>> idx ) {
  if (contains(idx)) {
    return this->make_clone();
  } else if (idx->contains(this->shared_from_this())) {
    return idx->make_clone();
  }

  /*
   * Case : union with strided
   */
  strided_indexstruct *strided = dynamic_cast<strided_indexstruct*>(idx.get());
  if (strided!=nullptr) {
    if (strided->local_size()<SMALLBLOCKSIZE) {
      auto indexed = this->make_clone();
      int s = strided->stride();
      for (auto i : *strided)
	indexed = indexed->add_element(i);
      return indexed->make_clone();
    } else {
      auto composite =shared_ptr<composite_indexstruct>{ make_shared<composite_indexstruct>() };
      composite->push_back(this->make_clone());
      composite->push_back(idx);
      return composite->force_simplify();
    }
  }
  /*
   * Case : union with indexed
   */
  indexed_indexstruct *indexed = dynamic_cast<indexed_indexstruct*>(idx.get());
  if (indexed!=nullptr) {
    auto merged = indexed->make_clone();
    for (auto v : indices)
      merged = merged->add_element(v);
    return merged->make_clone();
  }

  /*
   * Case : union with composite
   */
  composite_indexstruct *composite = dynamic_cast<composite_indexstruct*>(idx.get());
  if (composite!=nullptr) {
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
template<int d>
void composite_indexstruct<d>::push_back( shared_ptr<indexstruct<d>> idx ) {
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
template<int d>
shared_ptr<indexstruct<d>> composite_indexstruct<d>::make_clone() const {
  auto runion = shared_ptr<composite_indexstruct>{ make_shared<composite_indexstruct>() };
  for ( auto s : structs ) 
    runion->push_back( s->make_clone() );
  return runion;
};

template<int d>
index_int composite_indexstruct<d>::first_index() const {
  if (structs.size()==0)
    throw(std::string("Can not get first from empty composite"));
  index_int f = structs.at(0)->first_index();
  for (auto s : structs)
    f = MIN(f,s->first_index());
  return f;
};

template<int d>
index_int composite_indexstruct<d>::last_index() const {
  if (structs.size()==0)
    throw(std::string("Can not get last from empty composite"));
  index_int f = structs.at(0)->last_index();
  for (auto s : structs)
    f = MAX(f,s->last_index());
  return f;
};

template<int d>
index_int composite_indexstruct<d>::local_size() const {
  index_int siz=0;
  for (auto s : structs) {
    //printf("localsize: struct@%d\n",(long)(s.get()));
    siz += s->local_size();
  }
  return siz;
};

//! \todo I have my doubts
template<int d>
index_int composite_indexstruct<d>::get_ith_element( const index_int i ) const {
  if (i>=local_size())
    throw(fmt::format("Requested index {} out of bounds for {}",i,as_string()));
  if (structs.size()==1)
    return structs.at(0)->get_ith_element(i);
  index_int ilocal = i, start_check;
  for ( auto s : structs ) {
    if (ilocal==i)
      start_check = s->first_index();
    else {
      index_int sf = s->first_index();
      if (sf<start_check)
	print("WARNING composite not in increasing order: {}\n",as_string());
      start_check = sf;
    }
    if (ilocal>=s->local_size())
      ilocal -= s->local_size();
    else
      return s->get_ith_element(ilocal);
  }
  throw(fmt::format("still need to count down {} for ith element {} in composite {}",
		    ilocal,i,as_string()));
  //  throw(fmt::format("Cannot get index from true composite"));
};

/*! Composite containment is tricky
  \todo is there a way to optimize the indexed case?
*/
template<int d>
bool composite_indexstruct<d>::contains( shared_ptr<indexstruct<d>> idx ) const {
  /*
   * Case: contains strided. 
   * test by chopping off pieces
   */
  strided_indexstruct *strided = dynamic_cast<strided_indexstruct*>(idx.get());
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
  indexed_indexstruct *indexed = dynamic_cast<indexed_indexstruct*>(idx.get());
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
  composite_indexstruct *composite = dynamic_cast<composite_indexstruct*>(idx.get());
  if (composite!=nullptr) {
    for ( auto s : composite->get_structs() ) {
      if (!contains(s))
	return false;
    }
    return true;;
  }

  throw(fmt::format("Unimplemented case: composite contains {}",idx->type_as_string()));
};

template<int d>
bool composite_indexstruct<d>::contains_element( index_int idx ) const {
  //for (auto s : structs)
  for (int is=0; is<structs.size(); is++) {
    auto s = structs.at(is);
    if (s->contains_element(idx))
      return true;
  }
  return false;
};

template<int d>
index_int composite_indexstruct<d>::find( index_int idx ) const {
  index_int
    first = structs[0]->first_index(),
    accumulate = 0;
  for ( auto s : structs ) {
    if (s->first_index()<first)
      throw(fmt::format("Composite not sorted: {}",as_string()));
    first = structs[0]->first_index();
    if (s->contains_element(idx)) {
      return accumulate+s->find(idx);
    } else
      accumulate += s->local_size();
  }
  throw(format("Could not find {} in <<{}>>",idx,as_string()));
};

//! Disjointness test for composite. \todo bunch of unimplemented cases
template<int d>
bool composite_indexstruct<d>::disjoint( shared_ptr<indexstruct<d>> idx ) {
  template<int d>
  //bool composite_indexstruct<d>::disjoint( indexstruct *idx ) {
    template<int d>
  bool range_disjoint = indexstruct<d>::disjoint(idx);
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
template<int d>
shared_ptr<indexstruct<d>> composite_indexstruct<d>::struct_union
    ( shared_ptr<indexstruct<d>> idx) {
  //print("composite union {} + {}\n",as_string(),idx->as_string());

  // already contained: return
  for (auto s : structs)
    if (s->contains(idx)) {
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
      auto composite = shared_ptr<composite_indexstruct>( make_shared<composite_indexstruct>() );
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
  if (idx->local_size()==1) {
    //print("  merge single point {}\n",idx->as_string());
    bool was_merged{false};
    auto composite = shared_ptr<composite_indexstruct>( make_shared<composite_indexstruct>() );
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
    auto composite = shared_ptr<composite_indexstruct>( make_shared<composite_indexstruct>() );
    //print("  weeding out and adding\n");
    for (auto s : structs) {
      auto new_s = s->minus(idx);
      if (new_s->is_empty())
	continue;
      if (new_s->is_composite()) {
	auto s_comp = dynamic_cast<composite_indexstruct*>(new_s.get());
	if (s_comp==nullptr)
	  throw(fmt::format("could not upcast new_s"));
	for ( auto ss : s_comp->get_structs() )
	  composite->push_back(ss);
      } else
	composite->push_back(new_s);
    }
    auto comp = dynamic_cast<composite_indexstruct*>(idx.get());
    if (comp!=nullptr) {
      for ( auto s : comp->structs )
	composite->structs.push_back(s);
    } else     
      composite->push_back(idx);
    return composite; // no simplify
  }
};

template<int d>
shared_ptr<indexstruct<d>> composite_indexstruct<d>::intersect
    ( shared_ptr<indexstruct<d>> idx ) {
  // rule out the hard case
  composite_indexstruct *composite = dynamic_cast<composite_indexstruct*>(idx.get());
  if (composite!=nullptr)
    throw(std::string("Can not intersect composite-composite"));

  auto result = shared_ptr<indexstruct<d>>{ make_shared<empty_indexstruct>() };
  for (auto s : structs) {
    result = result->struct_union( s->intersect(idx) );
  }
  return result;
};

//! Composite minus set does a minus on each component \todo make unittest
template<int d>
shared_ptr<indexstruct<d>> composite_indexstruct<d>::minus
    ( shared_ptr<indexstruct<d>> idx ) const {
  auto rstruct = shared_ptr<composite_indexstruct>{ make_shared<composite_indexstruct>() };
  for (auto s : structs) {
    auto new_one = s->minus(idx);
    if (!new_one->is_empty()) {
      if (new_one->is_composite()) {
	composite_indexstruct *new_comp = dynamic_cast<composite_indexstruct*>(new_one.get());
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

template<int d>
shared_ptr<indexstruct<d>> composite_indexstruct<d>::relativize_to
    ( shared_ptr<indexstruct<d>> idx,bool trace ) {
  auto rel = shared_ptr<indexstruct<d>>( make_shared<empty_indexstruct>() );
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
template<int d>
shared_ptr<indexstruct<d>> composite_indexstruct<d>::force_simplify() const {

  try {
    // first the obvious simplicitions: empty and just one member
    if (get_structs().size()==0)
      return shared_ptr<indexstruct<d>>(make_shared<empty_indexstruct>());
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
		    ( make_shared<composite_indexstruct>() );
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
	auto composite = shared_ptr<composite_indexstruct>( make_shared<composite_indexstruct>() );
	auto indexed   = shared_ptr<indexstruct<d>>( make_shared<indexed_indexstruct>() );
	for (int is=0; is<ns; is++) {
	  auto istruct = structs.at(is);
	  if (structs.at(is)->is_indexed())
	    indexed = indexed->struct_union(istruct);
	  else
	    composite->push_back(istruct);
	}
	indexed = indexed->force_simplify();
	if (indexed->is_composite()) {
	  composite_indexstruct *icomp = dynamic_cast<composite_indexstruct*>(indexed.get());
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
	    for (int ii=0; ii<indexed->local_size(); ii++) {
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
	  auto composite = shared_ptr<composite_indexstruct>( make_shared<composite_indexstruct>() );
	  for (int is=0; is<ns; is++) {
	    if (is==an_ind) continue;
	    auto istruct = structs.at(is);
	    for (int ii=0; ; ) {
	      auto i = indexed->get_ith_element(ii);
	      if (istruct->can_incorporate(i)) {
		auto i_struct = shared_ptr<indexstruct<d>>(make_shared<contiguous_indexstruct>(i));
		istruct = istruct->struct_union(i_struct);
		indexed = indexed->minus(i_struct);
	      } else ii++;
	      if (ii>=indexed->local_size()) break;
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
	  index_int first=istruct->first_index(),last = istruct->last_index();
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
	index_int first=istruct->first_index(),last = istruct->last_index();
	bool left{true};
	auto composite = shared_ptr<composite_indexstruct>( make_shared<composite_indexstruct>() );
	for (int js=0; js<structs.size(); js++) { // loop over all others
	  if (js==cani) continue;
	  auto jstruct = structs.at(js);
	  if (jstruct->is_contiguous()) {
	    if (left && ( jstruct->contains_element(first) || jstruct->can_incorporate(first) ) ) {
	      if (!jstruct->contains_element(first))
		jstruct = jstruct->add_element(first);
	      istruct = istruct->minus
		( shared_ptr<indexstruct<d>>( make_shared<contiguous_indexstruct>(first) ) );
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
		( shared_ptr<indexstruct<d>>( make_shared<contiguous_indexstruct>(last) ) );
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
template<int d>
shared_ptr<indexstruct<d>> composite_indexstruct<d>::over_simplify() {
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
	if (jstruct->is_contiguous() || jstruct->local_size()==1) {
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
      auto composite = shared_ptr<composite_indexstruct>( make_shared<composite_indexstruct>() );
      // merge the two almost-adjacent structs
      auto new_s = shared_ptr<indexstruct<d>>
	( make_shared<contiguous_indexstruct>( structs[s1]->first_index(),structs[s2]->last_index() ) );
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
template<int d>
shared_ptr<indexstruct<d>> composite_indexstruct<d>::convert_to_indexed() const {
  auto idx = shared_ptr<indexstruct<d>>( make_shared<indexed_indexstruct>() );
  for ( auto s : structs ) {
    idx = idx->struct_union( s->convert_to_indexed() );
  }
  return idx;
};

template<int d>
std::string composite_indexstruct<d>::as_string() const {
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

template<int d>
bool composite_indexstruct<d>::equals( shared_ptr<indexstruct<d>> idx ) const {
  if (structs.size()>1)
    return false;
  else
    return structs.at(0)->equals(idx);
};

// //! Initialize an iterator by starting at the first component;
template<int d>
// void composite_indexstruct<d>::init_cur() {
//   cur_struct = 0; structs.at(cur_struct)->init_cur();
// };

template<int d>
// bool composite_indexstruct<d>::operator!=( indexstruct rr ) {
//   cur_struct < structs.size();
// };

template<int d>
// void composite_indexstruct<d>::operator++() {
//   // if (cmp_itr!=cur_cmp->end())
//   //   ++cmp_itr;
//   // else {
//   //   cur++; cur_cmp = structs.at(cur).get(); cmp_itr = cur_cmp->begin();
//   // }
// };

/****
 **** indexstructure
 ****/
void indexstructure::push_back( contiguous_indexstruct &&idx ) {
  print("pushing cnt <<{}>>\n",idx.as_string());
  print(".. onto <<{}>>\n",this->as_string());
  auto comp = dynamic_cast<composite_indexstruct*>(strct.get());
  if (comp!=nullptr) {
    auto cnt_ptr = std::shared_ptr<indexstruct<d>>( make_shared<contiguous_indexstruct>(idx) );
    print("pushing cnt ptr <<{}>>\n",cnt_ptr->as_string());
    comp->push_back(cnt_ptr);
  } else {
    print("Not composite; cannot push back\n");
    throw(fmt::format("Can not push into non-multi structure"));
  }
};

string indexstructure::as_string() const {
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
ioperator::ioperator( string op ) {
  if ( op==string("none") || op==string("no_op") ) {
    type = iop_type::NONE;
  } else if ( op[0]=='>' || op[0]=='<' ) {
    type = iop_type::SHIFT_REL;
    mod = op[1]!='=';
    by = std::stoi(string(op.begin()+2,op.end()));
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
    by = std::stoi(string(op.begin()+1,op.end()));
};

/*!
  ioperator constructor from keyword and amount. Currently implemented:
  - "shift" : relative shift
  - "shiftto" : shift to specific point
*/
ioperator::ioperator( string op,index_int amt ) {
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
index_int ioperator::operate( index_int i ) const {
  if (is_none_op())
    return i;
  else if (is_shift_op()) {
    index_int r = i+by;
    return r; // if (!is_modulo_op() && r<0) return 0; else 
  } else if (is_mult_op())
    return i*by;
  else if (is_div_op() || is_contdiv_op())
    return i/by;
  else if (is_function_op())
    return func(i);
  else
    throw(std::string("unknown operate type for operate"));
};

std::string ioperator::as_string() const {
  if (is_none_op())
    return std::string("Id");
  else if (is_shift_op()) {
    if (by>=0)
      return fmt::format("+{}",by);
    else
      return fmt::format("-{}",-by);
  } else if (is_mult_op()) {
    if (is_base_op())
      return fmt::format("x{}",by);
    else
      return fmt::format("*{}",by);
  } else if (is_div_op() || is_contdiv_op())
    return fmt::format("/{}",by);
  else if (is_function_op())
    return std::string("func");
  else
    throw(std::string("unknown operate type for string"));
};

/*!
  Operate on an index, and bump or wrap as needed.
*/
index_int ioperator::operate( index_int i, index_int m ) const {
  index_int r = operate(i);
  if (is_modulo_op()) {
    return MOD( r,m+1 );
  } else {
    return MAX( 0, MIN(m,r) );
  }
};

index_int ioperator::inverse_operate( index_int i ) const {
  if (i<0) {
    throw(std::string("I don't like to operate negative indices")); }
  if (is_none_op())
    return i;
  else if (is_left_shift_op())
    return i+1;
  else if (is_right_shift_op()) {
    if (i==0) {
      throw(std::string("Can't unrightshift zero"));
    } else return MAX(0,i-1);
  }
  throw(std::string("unknown operate type for inverse"));
};

index_int ioperator::inverse_operate( index_int i, index_int m ) const {
  if (i<0 || i>=m) {
    throw(std::string("index out of range")); }
  if (is_none_op())
    return i;
  else {
    index_int ii=-1;
    if (is_left_shift_op()) {
      if (i==m-1) {
	if (is_modulo_op())
	  return 0;
	else {
	  throw(std::string("Can not unleftshift last index")); }
      } else ii = i+1;
    } else if (is_right_shift_op()) {
      if (i==0) {
	if (is_modulo_op())
	  return m-1;
	else {
	  throw(std::string("Can not unrightshift zero")); }
      } else ii = i-1;
    }
    if (is_modulo_op())
      return MOD( ii,m );
    else
      return MAX( 0, MIN(m-1,ii) );
  }
};

/****
 **** Operate on indexstructs
 ****/

//! \todo lose a bunch of clones
template<int d>
shared_ptr<indexstruct<d>> strided_indexstruct<d>::operate( const ioperator &op ) const {
  shared_ptr<indexstruct<d>> operated{nullptr};
  if (op.is_none_op()) {
    return this->make_clone();
  } else if (op.is_shift_to()) {
    index_int amt = op.amount()-first;
    ioperator shift = shift_operator(amt);
    operated = this->operate(shift);
  } else if (op.is_shift_op()) {
    int f = op.operate(first_index()),
      l = f+(last_index()-first_index());
    if (stride()==1)
      operated = shared_ptr<indexstruct<d>>{ make_shared<contiguous_indexstruct>(f,l) };
    else
      operated = shared_ptr<indexstruct<d>>{ make_shared<strided_indexstruct>(f,l,stride()) };
  } else if (op.is_mult_op() || op.is_div_op() || op.is_contdiv_op()) {
    int
      new_first  = op.operate(first_index()),
      new_last, new_stride;
    if (op.is_base_op())
      new_stride  = stride();
    else
      new_stride = MAX(1,op.operate(stride()));
    if (op.is_contdiv_op())
      new_last = MAX( new_first, op.operate(last_index()+stride())-new_stride );
    else if (op.is_base_op())
      new_last = op.operate(last_index()+1)-1;
    else
      new_last = op.operate(last_index());
    if (new_stride==1)
      operated =
	shared_ptr<indexstruct<d>>{ make_shared<contiguous_indexstruct>(new_first,new_last) };
    else
      operated =
	shared_ptr<indexstruct<d>>{ make_shared<strided_indexstruct>(new_first,new_last,new_stride) };
  } else if (op.is_function_op()) {
    auto rstruct = shared_ptr<indexstruct<d>>{ make_shared<indexed_indexstruct>() };
    //for (auto i : *this) { VLE `this' is modified, which contradicts the const
    for (int ii=0; ii<local_size(); ii++) {
      index_int i = get_ith_element(ii);
      rstruct = rstruct->add_element( op.operate(i) );
    }
    operated = rstruct;
  } else throw(std::string("Can not operate contiguous/strided struct"));
  if (operated==nullptr)
    throw(std::string("Strided indexstruct operate: missing case"));
  return operated;
};

template<int d>
shared_ptr<indexstruct<d>> strided_indexstruct<d>::operate( const ioperator &&op ) const {
  shared_ptr<indexstruct<d>> operated{nullptr};
  if (op.is_none_op()) {
    return this->make_clone();
  } else if (op.is_shift_to()) {
    index_int amt = op.amount()-first;
    auto shift = shift_operator(amt);
    operated = this->operate(shift);
  } else if (op.is_shift_op()) {
    int f = op.operate(first_index()),
      l = f+(last_index()-first_index());
    if (stride()==1)
      operated = shared_ptr<indexstruct<d>>{ make_shared<contiguous_indexstruct>(f,l) };
    else
      operated = shared_ptr<indexstruct<d>>{ make_shared<strided_indexstruct>(f,l,stride()) };
  } else if (op.is_mult_op() || op.is_div_op() || op.is_contdiv_op()) {
    int
      new_first  = op.operate(first_index()),
      new_last, new_stride;
    if (op.is_base_op())
      new_stride  = stride();
    else
      new_stride = MAX(1,op.operate(stride()));
    if (op.is_contdiv_op())
      new_last = MAX( new_first, op.operate(last_index()+stride())-new_stride );
    else if (op.is_base_op())
      new_last = op.operate(last_index()+1)-1;
    else
      new_last = op.operate(last_index());
    if (new_stride==1)
      operated =
	shared_ptr<indexstruct<d>>{ make_shared<contiguous_indexstruct>(new_first,new_last) };
    else
      operated =
	shared_ptr<indexstruct<d>>{ make_shared<strided_indexstruct>(new_first,new_last,new_stride) };
  } else if (op.is_function_op()) {
    auto rstruct = shared_ptr<indexstruct<d>>{ make_shared<indexed_indexstruct>() };
    //for (auto i : *this) { VLE `this' is modified, which contradicts the const
    for (int ii=0; ii<local_size(); ii++) {
      index_int i = get_ith_element(ii);
      rstruct = rstruct->add_element( op.operate(i) );
    }
    operated = rstruct;
  } else throw(std::string("Can not operate contiguous/strided struct"));
  if (operated==nullptr)
    throw(std::string("Strided indexstruct operate: missing case"));
  return operated;
};

/*!
  We apply a sigma operator by taking the union over all indices.
*/
template<int d>
shared_ptr<indexstruct<d>> strided_indexstruct<d>::operate( const sigma_operator &op ) const {
  if (op.is_struct_operator()) {
    return op.struct_apply(*this);
  } else if (op.is_point_operator()) {
    auto pop = op.point_operator();
    return operate(pop);
  } else {
    auto rstruct = shared_ptr<indexstruct<d>>{ make_shared<empty_indexstruct>() };
    //for ( auto i : *this ) {
    for ( auto i=first_index(); i<=last_index(); i+=stride() ) {
      rstruct = rstruct->struct_union( op.operate(i) );
    }
    return rstruct;
  }
};

template<int d>
shared_ptr<indexstruct<d>> indexed_indexstruct<d>::operate( const ioperator &op ) const {
  auto shifted = shared_ptr<indexstruct<d>>( make_shared<indexed_indexstruct>() );
  shifted->reserve(this->local_size());
  if (op.is_shift_op()) {
    for (index_int i=0; i<indices.size(); i++)
      shifted->add_element( op.operate( indices[i] ) );
    return shifted;
  } else throw(std::string("operation not defined on indexed"));
};

//! \todo just use the result of operate?
template<int d>
shared_ptr<indexstruct<d>> composite_indexstruct<d>::operate( const ioperator &op ) const {
  auto rstruct = shared_ptr<composite_indexstruct>{ make_shared<composite_indexstruct>() };
  for (auto s : structs)
    rstruct->push_back( s->operate(op)->make_clone() );
  return rstruct;
};

/****
 **** Multi indexstructs
 ****/

//! Constructor, just specifying the dimensionality. Components are set later.
template<int d>
multi_indexstruct<d>::multi_indexstruct( int d ) {
  if (d<=0)
    throw(fmt::format("multi indexstruct dimension {} s/b >=1",d));
  dim = d;
  for (int dm=0; dm<d; dm++)
    components.push_back( shared_ptr<indexstruct<d>>( make_shared<empty_indexstruct>() ) );
  stored_local_size = domain_coordinate(d);
};

//! Constructor from coordinate: make structures in each dimension
template<int d>
multi_indexstruct<d>::multi_indexstruct( domain_coordinate c )
  : multi_indexstruct(c.get_dimensionality()) {
  int dim = c.get_dimensionality();
  stored_local_size = domain_coordinate_allones(c.get_dimensionality());
  for (int id=0; id<dim; id++)
    set_component(id,shared_ptr<indexstruct<d>>( make_shared<contiguous_indexstruct>(c[id])) );
};

template<int d>
multi_indexstruct<d>::multi_indexstruct( domain_coordinate f,domain_coordinate l )
  : multi_indexstruct(f.get_dimensionality()) {
  int dim = f.get_same_dimensionality(l.get_dimensionality());
  stored_local_size = l-f+domain_coordinate_allones(dim);
  for (int id=0; id<dim; id++)
    set_component(id,shared_ptr<indexstruct<d>>( make_shared<contiguous_indexstruct>(f[id],l[id]) ));
};

//! Constructor by wrapping up a single \ref indexstruct as the only dimension
template<int d>
multi_indexstruct<d>::multi_indexstruct( shared_ptr<indexstruct<d>> d1struct )
  : multi_indexstruct(1) {
  set_component(0,d1struct);
};

//! Constructor from individual \ref indexstruct objects. \todo should we clone?
template<int d>
multi_indexstruct<d>::multi_indexstruct( std::vector< shared_ptr<indexstruct<d>> > structs )
  : multi_indexstruct(structs.size()) {
  for ( int is=0; is<structs.size(); is++ ) {
    set_component(is,structs.at(is));
  }
};

template<int d>
multi_indexstruct<d>::multi_indexstruct( std::vector<index_int> sizes )
  : multi_indexstruct( domain_coordinate_zero(sizes.size()),
		       domain_coordinate(sizes)-1 ) {
};

/*!
  A multi-indexstruct is known if all its coordinates are known
  \todo it's kinda suspicious for only one component to be unknown, right?
*/
template<int d>
bool multi_indexstruct<d>::is_known() const {
  int d = get_dimensionality();
  for (int id=0; id<d; id++)
    if ( !get_component(id)->is_known() )
      return false;
  return true;
};

template<int d>
int multi_indexstruct<d>::type_as_int() const {
  int type{-1};
  int d = get_dimensionality();
  for (int id=0; id<d; id++) {
    if (type<0)
      type = get_component(id)->type_as_int();
    else
      if (type!=get_component(id)->type_as_int())
	throw(format("Can not deduce single type for multi indexstruct"));
  }
  return type;
};

/*!
  A multi-indexstruct is strided if all its coordinates are strided
  \todo it's kinda suspicious for only one component to be unstrided, right?
*/
template<int d>
bool multi_indexstruct<d>::is_strided() const {
  int d = get_dimensionality();
  for (int id=0; id<d; id++)
    if ( !get_component(id)->is_strided() )
      return false;
  return true;
};

/*!
  Deep copy of a multi_indexstruct;
  we copy both the immediate content and the multi pointers
*/
template<int d>
shared_ptr<multi_indexstruct> multi_indexstruct<d>::make_clone() const {
  int dim = get_dimensionality();
  auto cstruct = make_shared<multi_indexstruct>(dim) ;
  if (is_multi()) {
    for ( auto m : multi ) {
      cstruct->multi.push_back(m->make_clone());
    }
  } else {
    for (int id=0; id<dim; id++) {
      shared_ptr<indexstruct<d>> idx = get_component(id), nidx;
      if (idx==nullptr)
	nidx = idx;
      else
	nidx = idx->make_clone();
      cstruct->set_component( id,nidx );
    }
  }
  return cstruct;
};

//! Return dimensionality, but has to be same as someone else's.
template<int d>
int multi_indexstruct<d>::get_same_dimensionality(int d) const {
  int dim = get_dimensionality();
  if (dim!=d)
    throw(fmt::format("Multi idx dimensionality mismatch {} vs {}\n",dim,d));
  return dim;
};

/*!
  Set the indexstruct as component. This does not make a copy: that 
  is for instance done in the copy constructor.
*/
template<int d>
void multi_indexstruct<d>::set_component( int d, shared_ptr<indexstruct<d>> cmp ) {
  if (multi.size()>0)
    throw(fmt::format("set_component multi {}+{} needs multi",
    		      multi.at(0)->as_string(),multi.at(1)->as_string()));
  if (d<0 || d>=dim)
    throw(fmt::format("Component dimension index {} out of bounds 0--{}",d,dim));
  components.at(d) = cmp;
  if (cmp->is_known())
    stored_local_size.set(d,cmp->local_size());
  set_needs_recomputing();
};

template<int d>
shared_ptr<indexstruct<d>> multi_indexstruct<d>::get_component(int d) const {
  if (multi.size()>0) {
    throw(fmt::format("unimplemented get_component in {}",as_string()));
  }
  if (d<0 || d>=dim)
    throw(fmt::format("Component dimension index {} out of bounds 0--{}",d,dim));
  return components.at(d);
};

//! Compute volume; return stored volume if available.
template<int d>
index_int multi_indexstruct<d>::volume() const {
  if (multi.size()>0) {
    return enclosing_structure_r().volume();
  }
  if (stored_volume<0) {
    index_int s = 1;
    try {
      for ( auto c : components ) {
	auto ss = c->local_size();
	//print("component {} size {}\n",c->as_string(),ss);
	s *= ss;
      }
      //print("computed volume {} : {}\n",as_string(),s);
    } catch (std::string c) { throw(fmt::format("Error <<{}>> computing volume",c)); };
    stored_volume = s;
  }
  return stored_volume;
};

//! Return a vector of local sizes \todo can we do this and the next bunch without a star?
template<int d>
const domain_coordinate &multi_indexstruct<d>::local_size_r() const {
  if (multi.size()>0) throw(std::string("local_size_r needs multi"));
  return stored_local_size;
};

/*! Return a vector of first indices
  We memoize it as a st::share_ptr, hoping that regeneration will make it deallocate.
  However, we return a naked pointer to the calling environment, hoping that will
  not make a copy or otherwise remember it.
*/
template<int d>
const domain_coordinate &multi_indexstruct<d>::first_index_r() const {
  if (stored_first_index==nullptr) {
    //    compute_first_index();
    if (multi.size()==0) {
      //auto idx =
      stored_first_index =
	make_shared<domain_coordinate>( get_dimensionality() ) ;
      int id=0;
      for ( auto c : components )
	stored_first_index->set( id++,c->first_index() );
    } else {
      stored_first_index = 
	shared_ptr<domain_coordinate>
	( make_shared<domain_coordinate>( multi.at(0)->first_index_r() ) );
      for ( auto m : multi )
	stored_first_index->min_with( m->first_index_r() );
    }
  }
  return *stored_first_index;
};

/*! Return a vector of last indices
  We memoize it as a st::share_ptr, hoping that regeneration will make it deallocate.
  However, we return a naked pointer to the calling environment, hoping that will
  not make a copy or otherwise remember it.
*/
template<int d>
const domain_coordinate &multi_indexstruct<d>::last_index_r() const {
  if (stored_last_index==nullptr) {
    //    compute_last_index();
    if (multi.size()==0) {
      //auto idx =
      stored_last_index =
	shared_ptr<domain_coordinate>( make_shared<domain_coordinate>( get_dimensionality() ) );
      int id=0;
      for ( auto c : components )
	stored_last_index->set( id++,c->last_index() );
    } else {
      stored_last_index =
	shared_ptr<domain_coordinate>{
				      make_shared<domain_coordinate>( multi.at(0)->last_index_r() ) };
      //    print("multi last starts as {}\n",stored_last_index->as_string());
      for ( auto m : multi ) {
	stored_last_index->max_with( m->last_index_r() );
	// print("multi extended by {} to {}\n",
	// 		 m->last_index()->as_string(),stored_last_index->as_string());
      }
    }
  }
  return *stored_last_index;
};

template<int d>
std::vector<domain_coordinate> multi_indexstruct<d>::get_corners() const {
  int dim = get_dimensionality();
  std::vector<domain_coordinate> corners;
  { auto corner = first_index_r(); corners.push_back(corner); }
  for (int nd=0; nd<dim-1; nd++) {
    {
    // set nd first dimensions to max
      auto corner = domain_coordinate(first_index_r());
      for (int id=0; id<=nd; id++)
	corner.set( id, last_index_r().coord(id) );
      corners.push_back(corner);
    }
    {
    // set nd lat dimensions to max
      auto corner = domain_coordinate(first_index_r());
      for (int id=0; id<=nd; id++)
	corner.set( dim-1-id,  last_index_r().coord(dim-1-id) );
      corners.push_back(corner);
    }
  }
  { auto corner = last_index_r(); corners.push_back(corner); }
  return corners;
};

template<int d>
const multi_indexstruct &multi_indexstruct<d>::enclosing_structure_r() const {
  if (stored_enclosing_structure==nullptr)
    stored_enclosing_structure =
      shared_ptr<multi_indexstruct>
      ( make_shared<contiguous_multi_indexstruct>( first_index_r(),last_index_r() ) );
  return *(stored_enclosing_structure.get());
};

template<int d>
const shared_ptr<multi_indexstruct> multi_indexstruct<d>::enclosing_structure() const {
  if (stored_enclosing_structure==nullptr)
    stored_enclosing_structure =
      shared_ptr<multi_indexstruct>
      ( make_shared<contiguous_multi_indexstruct>( first_index_r(),last_index_r() ) );
  return stored_enclosing_structure;
};

/*!
  Return a vector of strides. If not every dimension is strided, constructing
  this will throw an exception.
*/
template<int d>
domain_coordinate *multi_indexstruct<d>::stride() const {
  int dim = get_dimensionality();
  domain_coordinate *strid = new domain_coordinate(dim);
  for (int id=0; id<dim; id++)
    strid->set(id,get_component(id)->stride());
  return strid;
};

//! A multi-dimensional indexstruct is empty if at least one dimension is empty
template<int d>
bool multi_indexstruct<d>::is_empty() const {
  int count = 0;
  if (multi_size()>0) {
    for ( auto s : multi )
      count += s->is_empty();
  } else {
    for ( auto c : components )
      count += c->is_empty();
  }
  return count>0;
};

template<int d>
bool multi_indexstruct<d>::is_contiguous() const { bool is=1;
  for (int id=0; id<get_dimensionality(); id++)
    is = is && get_component(id)->is_contiguous();
  return is;
};

template<int d>
std::string multi_indexstruct<d>::type_as_string() const {
  if (is_contiguous())
    return std::string("contiguous");
  else if (get_dimensionality()==0)
    return std::string("zero-dimensional");
  else
    return fmt::format("(not implemented, maybe {})",components.at(0)->type_as_string());
};

/*!
  Give the coordinate of a structure within this one.
template<int d>
  Containment testing is inherited from indexstruct<d>::location_of.

  \todo why the hell do we still have a naked pointer here?
*/
template<int d>
domain_coordinate *multi_indexstruct<d>::location_of
    ( shared_ptr<multi_indexstruct> inner ) const {
  int dim = get_same_dimensionality( inner->get_dimensionality() );
  auto loc = new domain_coordinate(dim);
  for (int id=0; id<dim; id++) {
    index_int iloc = this->get_component(id)->location_of( inner->get_component(id) );
    loc->set( id,iloc);
  }
  return loc;
};

/*! Where is this indexstruct located in a surrounding one?
  \todo the simplification should be done outside
*/
template<int d>
shared_ptr<multi_indexstruct> multi_indexstruct<d>::relativize_to
    ( shared_ptr<multi_indexstruct> other ) {
  auto simple = force_simplify();
  auto somple = other->force_simplify();

  int dim = get_same_dimensionality( other->get_dimensionality() );
  auto rstruct = make_shared<multi_indexstruct>(dim) ;
  for (int id=0; id<dim; id++) {
    try { rstruct->set_component
	( id,simple->get_component(id)->relativize_to( somple->get_component(id) ) );
    } catch (std::string c) { print("Error <<{}>> in component {}\n",c,id);
      throw(fmt::format("Could not relativize multi_indexstruct {} to {}",
			this->as_string(),other->as_string()));
    }
  }
  return rstruct;
};

/*!
  Find the linear location of a structure in this one.
  We don't do any testing on proper containment.
  \todo probably lose this in favour of the next
*/
template<int d>
index_int multi_indexstruct<d>::linear_location_of( shared_ptr<multi_indexstruct> idx ) const {
  return idx->first_index_r().linear_location_in( this->first_index_r(),this->last_index_r() );
};

template<int d>
index_int multi_indexstruct<d>::linear_location_in( shared_ptr<multi_indexstruct> idx ) const {
  auto find = first_index_r(),
    first = idx->first_index_r(), last = idx->last_index_r();
  index_int s = find.linear_location_in(first,last);
  return s;
};

template<int d>
index_int multi_indexstruct<d>::linear_location_in( const multi_indexstruct &idx ) const {
  auto find = first_index_r(),
    first = idx.first_index_r(), last = idx.last_index_r();
  index_int s = find.linear_location_in(first,last);
  return s;
};

template<int d>
domain_coordinate multi_indexstruct<d>::linear_offsets(shared_ptr<multi_indexstruct> inner) const {
  int dim = get_same_dimensionality(inner->get_dimensionality());
  domain_coordinate offsets(dim),
    ofirst = this ->first_index_r(), osize = this ->local_size_r(), 
    ifirst = inner->first_index_r(), isize = inner->local_size_r();
  return offsets;				    
};

/*! Operate the same operator on every dimension of a multi_indexstruct.
  There is a mechanism for limiting on what dimensions we operate.
  \todo write a unitttest for this
*/
template<int d>
shared_ptr<multi_indexstruct> multi_indexstruct<d>::operate( const ioperator &op ) const {
  int dim = get_dimensionality();
  auto rstruct = make_shared<multi_indexstruct>(dim) ;
  for (int id=0; id<dim; id++) {
    rstruct->set_component( id,get_component(id)->operate(op) );
  }
  return rstruct->force_simplify();
};
template<int d>
shared_ptr<multi_indexstruct> multi_indexstruct<d>::operate( const ioperator &&op ) const {
  int dim = get_dimensionality();
  auto rstruct = make_shared<multi_indexstruct>(dim) ;
  for (int id=0; id<dim; id++) {
    rstruct->set_component( id,get_component(id)->operate(op) );
  }
  return rstruct->force_simplify();
};

template<int d>
/*! Same as \ref multi_indexstruct<d>::operate but with truncation of the result.
  Treatment of the dimension is weird. Make dimension explicit

  \todo Treatment of the dimension is weird. Make dimension explicit
*/
template<int d>
shared_ptr<multi_indexstruct> multi_indexstruct<d>::operate
    ( const ioperator &op,shared_ptr<multi_indexstruct> truncation ) const {
  return operate(op,*(truncation.get())); };
template<int d>
shared_ptr<multi_indexstruct> multi_indexstruct<d>::operate
    ( const ioperator &op,const multi_indexstruct &truncation ) const {
  int dim = get_dimensionality(),opdim = op.get_dimension();
  auto rstruct = make_shared<multi_indexstruct>(dim) ;
  for (int id=0; id<dim; id++)
    rstruct->set_component
      ( id,get_component(id)->operate(op,truncation.get_component(id)) );
  return rstruct->force_simplify();
};

//! That looks like a dangerous use of get() on the operator!!
template<int d>
shared_ptr<multi_indexstruct> multi_indexstruct<d>::operate
    ( multi_ioperator *op,shared_ptr<multi_indexstruct> truncation ) const {
  return operate(op,*(truncation.get()));
};

template<int d>
shared_ptr<multi_indexstruct> multi_indexstruct<d>::operate
    ( multi_ioperator *op,const multi_indexstruct &truncation ) const {
  int dim = get_same_dimensionality(op->get_dimensionality());
  auto rstruct = make_shared<multi_indexstruct>(dim) ;
  for (int id=0; id<dim; id++)
    rstruct->set_component
      ( id,get_component(id)
	->operate(op->get_operator(id),truncation.get_component(id)));
  return rstruct->force_simplify();
};

template<int d>
shared_ptr<multi_indexstruct> multi_indexstruct<d>::operate( multi_ioperator *op ) const {
  int dim = op->get_dimensionality(), sim = get_dimensionality();
  if (dim>sim)
    throw(fmt::format("Operator dimensionality {} greater than struct: {}",dim,sim));
  auto s = make_shared<multi_indexstruct>(sim) ;
  for (int id=0; id<dim; id++)
    s->set_component(id, get_component(id)->operate( op->get_operator(id) ) );
  for (int id=dim+1; id<sim; id++)
    s->set_component(id, get_component(id)->make_clone() );
  return s->force_simplify();
};

//! \todo this is a synonym of multi_sigma_operator::operate : lose this.
template<int d>
shared_ptr<multi_indexstruct> multi_indexstruct<d>::operate( const multi_sigma_operator &op ) const {
  return op.operate(this->shared_from_this())->force_simplify();
};

//! Translating is an operation in place.
template<int d>
void multi_indexstruct<d>::translate_by(int d,index_int amt) {
  get_component(d)->translate_by(amt); set_needs_recomputing();
};

/*!
  Two base cases: both structs are non-multi, or the second one is non-multi.
 */
template<int d>
shared_ptr<multi_indexstruct> multi_indexstruct<d>::struct_union
    (shared_ptr<multi_indexstruct> other) {
  int dim = get_same_dimensionality(other->get_dimensionality());

  // empty cases
  if (is_empty()) {
    return other->make_clone();
  } else if (other->is_empty()) {
    return make_clone();
  } else if (other->is_multi() && !this->is_multi()) {
    // if any, the other is not multi
    return other->struct_union(this->shared_from_this());
  } else if (contains(other)) {
    // containment
    return make_clone();
  }

  // two multis then gradually merge the whole thing together
  if (other->is_multi() && this->is_multi()) {
    auto un = shared_ptr<multi_indexstruct>( make_shared<empty_multi_indexstruct>(dim) );
    for ( auto o : multi )
      un = un->struct_union(o);
    for ( auto o : other->multi )
      un = un->struct_union(o);
    return un;
  }

  //print("Union {} & {}\n", this->as_string(),other->as_string());
  {
    int diff = -1; shared_ptr<multi_indexstruct> un;
    un = shared_ptr<multi_indexstruct>( make_shared<empty_multi_indexstruct>(dim) );
    // first store yourself
    if (!is_multi()) {
      un->multi.push_back(make_clone());
    } else {
      for ( auto m : multi )  {
	auto uni =  m->make_clone(); un->multi.push_back(uni);
      }
    }
    // then store the other
    if ( !other->is_multi() ) {
      un->multi.push_back(other->make_clone());
    } else {
      for ( auto o : other->multi ) {
	un->multi.push_back( o->make_clone() );
      }
    }
    return un->force_simplify(); // can we detect mergability earlier? e.g. in 1D?
  }
};

/*!
  Test whether the `other' argument can be merged into `this'. If so, return 
  the dimension.
*/
template<int d>
bool multi_indexstruct<d>::can_union_in_place
    (shared_ptr<multi_indexstruct> other,int &diff) const {
  diff = -1;
  if (is_multi() || other->is_multi())
    return false;

  // try to make an actual union
  for (int d=0; d<get_same_dimensionality( other->get_dimensionality() ); d++) {
    // find dimensions that don't fit
    if ( !get_component(d)->equals( other->get_component(d) ) ) {
      if (diff>=0) {
	// if we already found a differing dimension, we have to make a multi
	return false;
      } else // record the differing dimension
	diff = d;
    }
  }
  return true;
};

/*!
  The `other' struct can be merged along dimension `diff', so yield a 
  expanded struct with the other struct merged in.
*/
template<int d>
shared_ptr<multi_indexstruct> multi_indexstruct<d>::struct_union_in_place
    (shared_ptr<multi_indexstruct> other,int diff) {
  int dim = get_same_dimensionality(other->get_dimensionality());
  if (other->is_multi())
    throw(std::string("Argument of union in place should not be multi"));
  auto un = shared_ptr<multi_indexstruct>( make_shared<empty_multi_indexstruct>(dim) );
  for (int d=0; d<dim; d++) {
    if (d!=diff)
      un->set_component( d,get_component(d)->make_clone() );
    else {
      auto d1 = get_component(d), d2 = other->get_component(d);
      try {
	auto new_un = d1->struct_union(d2);
	un->set_component( d,new_un );
      } catch (std::string c) {
	throw(fmt::format("struct union in place, dim {}: {} & {} : {}",
			  d,d1->as_string(),d2->as_string(),c));
      }
    }
  }
  return un;
};

template<int d>
shared_ptr<multi_indexstruct> multi_indexstruct<d>::split_along_dim
    (int splitdim,shared_ptr<indexstruct<d>> induce) const {
  int dim = get_dimensionality();
  if (splitdim<0 || splitdim>=dim)
    throw(fmt::format("invalid dim={} for {}",splitdim,as_string()));
  //print("Split {} on dim {} along {}\n",this->as_string(),dim,induce->as_string());
  auto res = shared_ptr<multi_indexstruct>( make_shared<empty_multi_indexstruct>(dim) );
  if (is_multi()) {
    for ( auto m : multi ) {
      auto msplit = m->split_along_dim(splitdim,induce);
      //print("split component to be unioned: {}\n",msplit->as_string());
      if (msplit->is_multi()) {
	for ( auto split_comp : msplit->multi )
	  res->multi.push_back(split_comp);
      } else
	res->multi.push_back(msplit);
      //res = res->struct_union(msplit);
    }
  } else {
    auto cmp = get_component(splitdim)->split(induce);
    composite_indexstruct *comp = dynamic_cast<composite_indexstruct*>( cmp.get() );
    if (comp==nullptr)
      throw(fmt::format("could not upcast supposed composite (split along dim)"));
    for ( auto c : comp->get_structs() ) {
      auto s = make_shared<multi_indexstruct>(dim) ;
      for (int id=0; id<dim; id++) {
	if (id==splitdim)
	  s->set_component( id,c );
	else
	  s->set_component( id,get_component(id) );
      }
      res->multi.push_back(s); // = res->struct_union(s);
    }
  }
  return res;
};

/*!
  This function tries to simplify a union.
  There are still cases being missed, mostly non-orthogonal ones, such as
   ____
  |_   |
  | |__|
  |____|
  Hm. Can this really exist?
*/
template<int d>
shared_ptr<multi_indexstruct> multi_indexstruct<d>::force_simplify(bool trace) const {
  try {
    if (is_multi()) {
      if (trace) print("Simplifying multi: {}\n",this->as_string());
      if (multi_size()==1) {
	if (trace)
	  print("Returning unique member {}\n",multi.at(0)->as_string());
	return multi.at(0);
      } else {
	auto rstruct = shared_ptr<multi_indexstruct>( make_shared<empty_multi_indexstruct>(dim) );
	multi_indexstruct cstruct(*this);
	bool merged{false};
	// go through components, merging
	for (int im=0; im<multi_size(); im++) {
	  bool can{false}; int diff = -1;
	  // see if it can be merged with a previous
	  for (int jm=0 ;jm<im; jm++) { // if can union with previous, then was already
	    can = cstruct.multi.at(im)->can_union_in_place(cstruct.multi.at(jm),diff);
	    if (can) {
	      if (trace)
		print("multi {} can be merged back with {}\n",
		      cstruct.multi.at(im)->as_string(),cstruct.multi.at(jm)->as_string());
	      try {
		cstruct.multi.at(jm) =
		  cstruct.multi.at(jm)->struct_union_in_place(cstruct.multi.at(im),diff);
	      } catch (std::string c) { print("Error: {}",c);
		throw(fmt::format("Union in place1 failed {} & {}",
				  cstruct.multi.at(jm)->as_string(),
				  cstruct.multi.at(im)->as_string()));
	      }
	      break;
	    }
	  }
	  if (!can) {
	    // can not be merged with previous, so merge with future and push.
	    auto imstruct = cstruct.multi.at(im)->make_clone();
	    if (trace)
	      print("multi {} is new\n",imstruct->as_string());
	    for (int jm=im+1; jm<multi.size(); jm++) {
	      auto jmstruct = cstruct.multi.at(jm);
	      if (imstruct->can_union_in_place(jmstruct,diff)) {
		if (trace) print(".. merging along dim {}: {} & {}\n",diff,im,jm);
		try {
		  imstruct = imstruct->struct_union_in_place(jmstruct,diff);
		} catch (std::string c) { print("Error: {}",c);
		  throw(fmt::format("Union in place2 {} & {}",
				    imstruct->as_string(),jmstruct->as_string()));
		}
		if (trace) print(".. merging into {}, giving {}\n",
				 jmstruct->as_string(),imstruct->as_string());
		merged = true;
	      }
	    }
	    if (trace) print(".. pushing {}\n",imstruct->as_string());
	    rstruct->multi.push_back(imstruct);
	  }
	}
	if (rstruct->multi_size()==1) {
	  if (trace)
	    print("Returning unique member {}\n",rstruct->multi.at(0)->as_string());
	  return rstruct->multi.at(0);
	} else {
	  if (trace) print("simplify non-multi\n");
	  if (merged) {
	    if (trace) print("do another pass over {}\n",rstruct->as_string());
	    rstruct = rstruct->force_simplify();
	    if (trace) print("giving {}\n",rstruct->as_string());
	  }
	  return rstruct;
	}
      }
    } else {
      //print("Simplifying non-multi: {}\n",as_string());
      int dim = get_dimensionality();
      auto rstruct = make_shared<multi_indexstruct>(dim) ;
      for (int id=0; id<dim; id++)
	rstruct->set_component(id,get_component(id)->force_simplify());
      //print(".. simplified to: {}\n",rstruct->as_string());
      return rstruct;
    }
  } catch (string e) { print("Error: {}",e);
    throw("Could not simplify multi_indexstruct");
  } catch( ... ) {
    throw("Could not simplify multi_indexstruct");
  }
}

/*!
  A multi-intersection is the simultaneous intersection in all components.
  \todo add unittest for strided case to 102
  \todo needs to return shared_ptr
*/
template<int d>
shared_ptr<multi_indexstruct> multi_indexstruct<d>::intersect
    ( shared_ptr<multi_indexstruct> other ) {
  return intersect( *(other.get()) );
};

template<int d>
shared_ptr<multi_indexstruct> multi_indexstruct<d>::intersect
    ( const multi_indexstruct &other ) {
  int dim = get_same_dimensionality( other.get_dimensionality() );

  if (other.is_empty())
    return shared_ptr<multi_indexstruct>( make_shared<empty_multi_indexstruct>(dim) );
  if (contains(other))
    return other.make_clone();
  if (other.contains(this->shared_from_this()))
    return this->make_clone();
  if (first_index_r()>other.last_index_r() || last_index_r()<other.first_index_r() )
    return shared_ptr<multi_indexstruct>( make_shared<empty_multi_indexstruct>(dim) );

  auto rstruct = make_shared<multi_indexstruct>(dim) ;
  for (int id=0; id<dim; id++)
    rstruct->set_component
      (id,shared_ptr<indexstruct<d>>
       ( get_component(id)->intersect(other.get_component(id))) );
  return rstruct;
};

template<int d>
bool multi_indexstruct<d>::contains( shared_ptr<multi_indexstruct> other ) const {
  return contains( *(other.get()) );
};

//! Multi_indexstruct containment requires all dimensions to contain
template<int d>
bool multi_indexstruct<d>::contains( const multi_indexstruct &other ) const {
  bool trace{false};
  int dim  = get_same_dimensionality( other.get_dimensionality() );
  if (trace)
    print("Testing containment {} > {} ?\n",this->as_string(),other.as_string());
  if (is_empty())
    return other.is_empty();

  if (!is_multi()) {
    if (other.is_multi()) {
      if (trace)
	print("   test non-multi {} contains multi {}\n",as_string(),other.as_string());
      // simple & multi: test if you contain all multis
      for ( auto o : other.multi ) {
	if (!contains(o))
	  return false;
      } // .... it contains all other multis
      return true;
    } else { // both not multi: we use an orthogonal containment test
      if (trace)
	print("   test non-multi {} contains non-multi {}\n",as_string(),other.as_string());
      for (int id=0; id<get_same_dimensionality(other.get_dimensionality()); id++) {
	if (get_component(id)->contains(other.get_component(id))) {
	  if (trace) print("   true in component {}\n",id);
	} else {
	  if (trace) print("   false because component {}\n",id);
	  return false; }
      }
      if (trace) print("   true\n\n");
      return true;
    }
  } else { // this is_multi
    if (other.is_multi()) {
      if (trace) print("   test multi contains multi\n");
      for ( auto o : other.multi ) { // check all others
	if (!contains(o))
	  return false;
      }
      return true;
    } else { // multi, other not
      if (trace) print("   test multi contains non-multi\n");
      for ( auto &p : other.get_corners() ) {
	if (contains_element(p)) {
	if (trace) print("      test contains corner {}: true\n",p.as_string());
	} else {
	if (trace) print("      test contains corner {}: false\n",p.as_string());
	  return false;
	}
      }
      return true;
    }
  }
};

//! \todo weird clone because of a const problem.
template<int d>
shared_ptr<multi_indexstruct> multi_indexstruct<d>::minus
    ( shared_ptr<multi_indexstruct> idx,bool trace ) const {
  int dim = get_same_dimensionality( idx->get_dimensionality() );

  fmt::memory_buffer w;
  if (trace) {
    format_to(w.end(),"Minus {}-{}",as_string(),idx->as_string());
    print("{}\n",to_string(w)); }

  // easy cases
  if (!is_multi() && !idx->is_multi() && idx->contains(make_clone()/*shared_from_this()*/)) {
    if (trace) print("{} easy case: contain in other\n",to_string(w));
    return shared_ptr<multi_indexstruct>( make_shared<empty_multi_indexstruct>(dim) );
  }
  if (!is_multi() && !idx->is_multi()) {
    if (trace) print("{} easy case: disjoint in some dimension\n",to_string(w));
    for (int id=0; id<dim; id++) {
      if (get_component(id)->disjoint(idx->get_component(id))) {
	if (trace) print(".. {} simple/simple disjoint in dim={}\n",to_string(w),id);
	return make_clone();
      }
    }
    if (trace) print(".. {} no easy disjoint case\n",to_string(w));
  }

  if (idx->is_multi()) {
    if (trace) print("{} Subtract multis from this:\n",to_string(w));
    auto res = shared_ptr<multi_indexstruct>( make_shared<empty_multi_indexstruct>(dim) );
    for ( auto o : idx->multi ) {
      auto omin = minus(o); res = res->struct_union(omin);
      if (trace) print(".. {} subtracted multi: {}\n",to_string(w),omin->as_string());
    }
    return res;
  } else if (is_multi()) {
    if (trace) print("{} Subtract from multi:\n",to_string(w));
    auto res = shared_ptr<multi_indexstruct>( make_shared<empty_multi_indexstruct>(dim) );
    for ( auto m : multi ) {
      auto mm = m->minus(idx); res = res->struct_union(mm);
      if (trace) print(".. {} subtracted multi: {}\n",to_string(w),mm->as_string());
    }
    return res;
  } else {
    if (trace) print("{} Subtract non-multi from non-multi:\n",to_string(w));
    // Case: other contains in all dimensions but one
    int noncontain = -1;
    for (int id=0; id<dim; id++) {
      if (!idx->get_component(id)->contains(get_component(id))) {
	if (noncontain<0) {
	  if (trace) print("non-containment in dimension {}\n",id);
	  noncontain = id;
	} else { // we have already found one non-containing dimension
	  if (trace) print("{} needs splitting in dim={}\n",to_string(w),noncontain);
	  auto s = split_along_dim(noncontain,idx->get_component(noncontain));
	  if (trace) print("Retry minus with: {}\n",s->as_string());
	  // s = s->split_along_dim(id,idx->get_component(id));
	  // if (trace) print("split twice {}\n",s->as_string());
	  return s->minus(idx);
	}
      }
    }
    if (noncontain>=0) {
      auto mn = make_shared<multi_indexstruct>(dim) ;
      for (int id=0; id<dim; id++) {
	if (id==noncontain)
	  mn->set_component(id,get_component(id)->minus(idx->get_component(id)));
	else
	  mn->set_component(id,get_component(id)->make_clone());
      }
      return mn;
    }
  }
  template<int d>
  throw(fmt::format("Unimplemented multi_indexstruct<d>::minus"));
}

//! Multi_indexstruct containment requires all dimensions to contain \todo write unit test
template<int d>
bool multi_indexstruct<d>::contains_element( const domain_coordinate &i ) const {
  if (is_multi()) {
    for ( auto s : multi )
      if (s->contains_element(i))
	return true;
    return false;
  } else {
    for (int id=0; id<get_same_dimensionality(i.get_dimensionality()); id++)
      if ( !(get_component(id)->contains_element(i.coord(id))) )
	return false;
    return true;
  }
};

//! Multi_indexstruct containment requires all dimensions to contain \todo write unit test
template<int d>
bool multi_indexstruct<d>::contains_element( const domain_coordinate &&i ) const {
  if (is_multi()) {
    for ( auto s : multi )
      if (s->contains_element(i))
	return true;
    return false;
  } else {
    for (int id=0; id<get_same_dimensionality(i.get_dimensionality()); id++)
      if ( !(get_component(id)->contains_element(i.coord(id))) )
	return false;
    return true;
  }
};

//! Multi_indexstruct equals requires all dimensions to contain \todo write unit test
template<int d>
bool multi_indexstruct<d>::equals( shared_ptr<multi_indexstruct> other ) const {
  if (is_multi() || other->is_multi()) {
    throw(std::string("multi multi equals not implemented"));
  } else {
    for (int d=0; d<get_dimensionality(); d++)
      if ( !(get_component(d)->equals(other->get_component(d))) )
	return false;
    return true;
  }
};
template<int d>
bool multi_indexstruct<d>::operator==( const multi_indexstruct &other ) const {
  if (is_multi() || other.is_multi()) {
    throw(std::string("multi multi equals not implemented"));
  } else {
    for (int d=0; d<get_dimensionality(); d++)
      if ( !(get_component(d)->equals(other.get_component(d))) )
	return false;
    return true;
  }
};

//! Find an element expressed in linearized coordinates
template<int d>
index_int multi_indexstruct<d>::linearfind( index_int i ) {
  int dim = get_dimensionality();
  if (dim==1)
    return get_component(0)->find(i);
  else throw(std::string("Can not linearfind in multi-d"));
};

template<int d>
std::string multi_indexstruct<d>::as_string() const {
  int dim = get_dimensionality();
  fmt::memory_buffer w; format_to(w.end(),"Dim={} ",dim);
  if (multi.size()==0) {
    for (int id=0; id<dim; id++)
      format_to(w.end(),"{}:[{}]",id,get_component(id)->as_string());
  } else {
    format_to(w.end(),"M{}:",multi.size());
    for (int im=0; im<multi.size(); im++) {
      format_to(w.end(),"{}",multi.at(im)->as_string());
      if (im<multi.size()-1)
	format_to(w.end(),"+");
    }
  }
  return to_string(w);
};

/*!
  We begin iteration by giving the first coordinate.
  We store the current iterate as a private domain_coordinate: `cur_coord'.
  Iteration is done C-style: the last coordinate varies quickest.

  Note: iterating is only defined for bricks.
*/
template<int d>
multi_indexstruct &multi_indexstruct<d>::begin() {
  if (!is_contiguous())
    throw(fmt::format("Iteration only defined for contiguous"));
  cur_coord = first_index_r();
  //print("multi::begin: {}\n",cur_coord.as_string());
  return *this;
};

/*!
  Since we are iterating C-style (row-major),
  the iteration endpoint is like the first coordinate but with the 
  zero component increased.
  In row major this would be the first iterated point that is not in the brick.
*/
template<int d>
multi_indexstruct &multi_indexstruct<d>::end() {
  cur_coord = domain_coordinate( first_index_r() );
  cur_coord.set(0,last_index_r()[0]+1);
  //print("multi::end: {}\n",cur_coord.as_string());
  return *this;
};

/*!
  Here's how to iterate: 
  - from last to first dimensions, find the dimension where you are not at the far edge
  - increase the coordinate in that dimension
  - all higher dimensions are reset to the first coordinate.
*/
template<int d>
void multi_indexstruct<d>::operator++() {
  int dim = first_index_r().get_dimensionality();
  for (int id=dim-1; id>=0; id--) {
    if (cur_coord[id]<last_index_r()[id] || id==0) {
      cur_coord.set(id,cur_coord.at(id)+1); break;
    } else
      cur_coord.set(id,first_index_r()[id]);
  }
};

template<int d>
bool multi_indexstruct<d>::operator!=( multi_indexstruct &other ) {
  bool
    f = first_index_r()!=other.first_index_r(),
    l = last_index_r()!=other.last_index_r(),
    c = cur_coord!=other.cur_coord; // what does this test?
  // print("mult:neq {}@{} vs {}@{} : {}, {}, {}\n",
  // 	     as_string(),cur_coord.as_string(),
  // 	     other.as_string(),other.cur_coord.as_string(),
  // 	     f,l,c);
  return f || l || c;
};

template<int d>
bool multi_indexstruct<d>::operator==( multi_indexstruct &other ) {
  bool
    f = first_index_r()==other.first_index_r(),
    l = last_index_r()==other.last_index_r(),
    c = cur_coord==other.cur_coord;
  // print("mult:eq {}@{} vs {}@{} : {}, {}, {}\n",
  // 	     as_string(),cur_coord.as_string(),
  // 	     other.as_string(),other.cur_coord.as_string(),
  // 	     f,l,c);
  return f && l && c;
};

template<int d>
domain_coordinate &multi_indexstruct<d>::operator*() {
  //print("multi::deref: {}\n",cur_coord.as_string());
  return cur_coord;
};

/****
 **** Operators
 ****/

shared_ptr<indexstruct<d>> sigma_operator::operate(index_int i) const {
  if (lambda_s)
    throw(std::string("Can not operate on point: only defined for structs"));
  if (is_point_operator()) {
    return shared_ptr<indexstruct<d>>{make_shared<contiguous_indexstruct>( point_func.operate(i) )};
  } else {
    return func(i);
  }
};

/*! \todo find occurrences of the lambda_i case. can they be done by storing an ioperator?
  \todo print("sigma by point operator is dangerous\n");
*/
shared_ptr<indexstruct<d>> sigma_operator::operate( shared_ptr<indexstruct<d>> i) const {
  if (lambda_s) {
    try {
      return sfunc(*i);
    } catch (std::string c) {
      throw(fmt::format("Error <<{}>> applying structure sigma",c));
    };
  } else if (lambda_p) {
    return shared_ptr<indexstruct<d>>
      {make_shared<contiguous_indexstruct
		   >( point_func.operate(i->first_index()),point_func.operate(i->last_index()) )};
  } else if (lambda_i) {
    //print("sigma by point operator is dangerous\n");
    return shared_ptr<indexstruct<d>>
      ( make_shared<contiguous_indexstruct
	>( func(i->first_index())->first_index(),
	  func(i->last_index())->last_index() ) );
  } else {
    throw(std::string("sigma::operate(struct) weird case"));
  }
};

std::string sigma_operator::as_string() const {
  fmt::memory_buffer w;
  format_to(w.end(),"Sigma operator");
  if (is_point_operator())
    format_to(w.end()," from ioperator \"{}\"",point_func.as_string());
  return to_string(w);
};

void multi_ioperator::set_operator(int id,ioperator &op) {
  if (id<0 || id>=get_dimensionality())
    throw(fmt::format("set component invalid dimension {}: s/b 0--{}",
		      id,0,get_dimensionality()));
  operators.at(id) = op;
};
void multi_ioperator::set_operator(int id,ioperator &&op) {
  if (id<0 || id>=get_dimensionality())
    throw(fmt::format("set component invalid dimension {}: s/b 0--{}",
		      id,0,get_dimensionality()));
  operators.at(id) = op;
};

/*!
  Operating on a domain coordinate gives a new domain_coordinate
  by plain Cartesian product of the operate results in each dimension.
*/
domain_coordinate *multi_ioperator::operate( domain_coordinate *c ) {
  int dim = c->get_dimensionality();
  if (operator_based) {
    domain_coordinate *idx = new domain_coordinate(dim);
    for (int id=0; id<dim; id++)
      idx->set(id, get_operator(id).operate(c->coord(id)) );
    return idx;
  } else if (point_based) {
    return pointf(c);
  } else
    throw(std::string("What type is this multi_ioperator?"));
};

/*! A multi operator is modulo if all components are modulo.
  \todo can we be more clever about the function case?
*/
bool multi_ioperator::is_modulo_op() {
  if (operator_based) {
    for (int id=0; id<get_dimensionality(); id++)
      if (!get_operator(id).is_modulo_op())
	return false;
    return true;
  } else return false;
};

//! A multi operator is shift if all components are shift.
bool multi_ioperator::is_shift_op() {
  if (operator_based) {
    for (int id=0; id<get_dimensionality(); id++)
      if (!get_operator(id).is_shift_op())
	return false;
    return true;
  } else return false;
};

//! A multi operator is restrict if all components are restrict.
bool multi_ioperator::is_restrict_op() {
  if (operator_based) {
    for (int id=0; id<get_dimensionality(); id++)
      if (!get_operator(id).is_restrict_op())
	return false;
    return true;
  } else return false;
};

/****
 **** Multi sigma
 ****/

//! Constructor from list of sigma operators
multi_sigma_operator::multi_sigma_operator( vector<sigma_operator> ops )
  : multi_sigma_operator(ops.size()) {
  print("setting vector implementation from sigmas\n");
  operator_implementation = shared_ptr<multi_sigma_operator_implementation>
    ( make_shared<multi_sigma_operator_impl_vector>(ops) );
};

multi_sigma_operator::multi_sigma_operator( vector<ioperator> ops )
  : multi_sigma_operator(ops.size()) {
  try {
    operator_implementation = shared_ptr<multi_sigma_operator_implementation>
      ( make_shared<multi_sigma_operator_impl_vector>(ops) );
  } catch (string c) {
    print("Could not create impl_vector: <<{}>>\n",c);
    throw("Error setting vector implementation from iops");
  }
};

multi_sigma_operator_impl_vector::multi_sigma_operator_impl_vector
    ( std::vector<sigma_operator> ops )
  : multi_sigma_operator_implementation(ops.size()),operators(ops) {
};

multi_sigma_operator_impl_vector::
multi_sigma_operator_impl_vector( vector<ioperator> ops )
  : multi_sigma_operator_implementation(ops.size()) {
  int dim = ops.size();
  for (int id=0; id<dim; id++)
    operators.push_back(sigma_operator(ops.at(id)));
};

shared_ptr<multi_indexstruct> multi_sigma_operator_impl_vector::
operate( const domain_coordinate &point ) const {
  int dim = get_same_dimensionality(point.get_dimensionality());
  print("applying operators\n");
  auto idx = make_shared<multi_indexstruct>(dim) ;
  for (int id=0; id<dim; id++) { auto op = operators.at(id);
    try { idx->set_component(id, op.operate(point.coord(id)) );
    } catch (...) {
      throw(format("Could not operate <<{}>> set component {}",
			op.as_string(),id));
    }
  }
  return idx;
};

shared_ptr<multi_indexstruct> multi_sigma_operator_impl_vector::operate
    ( const multi_indexstruct &idx ) const {
  int dim = get_same_dimensionality(idx.get_dimensionality());
  auto opidx = make_shared<multi_indexstruct>(dim) ;
    for (int id=0; id<dim; id++) { 
      auto cpt = idx.get_component(id); auto op = operators.at(id);
      try {
	opidx->set_component(id,op.operate(cpt));
      } catch (std::string c) {
	print("Error <<{}>> applying <<{}>> in dimension {}",
		   c,op.as_string(),id,cpt->as_string());
	throw(fmt::format("Could not appy multi_sigma by operators on <<{}>>",
			  idx.as_string()));
      }
    }
    return opidx;
};

//! Constructor from coordinate operator
multi_sigma_operator::multi_sigma_operator
    ( int dim, function< domain_coordinate(const domain_coordinate&) > f )
  : multi_sigma_operator(dim) {
  operator_implementation = shared_ptr<multi_sigma_operator_implementation>
    ( make_shared<multi_sigma_operator_impl_coord>(dim,f) );
};

std::shared_ptr<multi_indexstruct> multi_sigma_operator_impl_coord::operate
    ( const domain_coordinate &point ) const {
  int dim = get_same_dimensionality(point.get_dimensionality());
  print("applying coord\n");
  try {
    return std::make_shared<multi_indexstruct>( coord_oper(point) ) ;
  } catch (std::string c) {
    throw(fmt::format("Multi sigma op failed on coord: {}",c)); }      
};

shared_ptr<multi_indexstruct> multi_sigma_operator_impl_coord::operate
    ( const multi_indexstruct &idx ) const {
  //print("applying coord\n");
  if (idx.is_contiguous()) {
    try {
      auto
	ofirst = coord_oper(idx.first_index_r()),
	olast = coord_oper(idx.last_index_r());
      return shared_ptr<multi_indexstruct>
	( make_shared<contiguous_multi_indexstruct>( ofirst,olast ) );
    } catch (std::string c) {
      throw(fmt::format("Multi sigma failed by coord on struct {}: {}",idx.as_string(),c)); }
  } else
    throw(std::string("Can not point operate on non-contiguous"));
};

//! Constructor from coordinate-to-struct operator
multi_sigma_operator::multi_sigma_operator
    //( int dim,shared_ptr<multi_indexstruct>(*f)(const domain_coordinate&) )
    ( int dim, function< shared_ptr<multi_indexstruct>(const domain_coordinate&) > f )
  : multi_sigma_operator(dim) {
  operator_implementation = shared_ptr<multi_sigma_operator_implementation>
    ( make_shared<multi_sigma_operator_impl_sigma>(dim,f) );
};

std::shared_ptr<multi_indexstruct> multi_sigma_operator_impl_sigma::operate
    ( const domain_coordinate &point ) const {
  int dim = get_same_dimensionality(point.get_dimensionality());
  print("applying sigma\n");
  try { return sigma_oper(point);
  } catch (std::string c) {
    throw(fmt::format("Multi sigma op failed sigma based: {}",c)); }
};

shared_ptr<multi_indexstruct> multi_sigma_operator_impl_sigma::operate
    ( const multi_indexstruct &idx ) const {
  try {
    auto
      ofirst = sigma_oper( idx.first_index_r() )->first_index_r(),
      olast = sigma_oper( idx.last_index_r() )->last_index_r();
    return shared_ptr<multi_indexstruct>
      ( make_shared<contiguous_multi_indexstruct>( ofirst,olast ) );
  } catch (std::string c) {
    throw(fmt::format("Multi sigma failed by sigma on struct {}: {}",idx.as_string(),c)); }
};

//! Constructor from struct to struct
multi_sigma_operator::multi_sigma_operator
    ( int dim,function<shared_ptr<multi_indexstruct>(const multi_indexstruct&) > f )
    //( int dim,function< shared_ptr<multi_indexstruct>(const multi_indexstruct&) > f )
  : multi_sigma_operator(dim) {
  operator_implementation = shared_ptr<multi_sigma_operator_implementation>
    ( make_shared<multi_sigma_operator_impl_struct>(dim,f) );
};

std::shared_ptr<multi_indexstruct> multi_sigma_operator_impl_struct::operate
    ( const domain_coordinate &point ) const {
  int dim = get_same_dimensionality(point.get_dimensionality());
  print("applying struct\n");
  try {
    auto pointstruct = multi_indexstruct(point);
    return struct_oper(pointstruct);
    //return struct_oper( std::shared_ptr<multi_indexstruct>(pointstruct) );
  } catch (std::string c) {
    throw(fmt::format("Multi sigma op failed struct based: {}",c)); }      
};

shared_ptr<multi_indexstruct> multi_sigma_operator_impl_struct::operate
    ( const multi_indexstruct &idx ) const {
  try {
    return struct_oper(idx);
  } catch (std::string c) {
    throw(fmt::format("Multi sigma failed by struct on struct {}: {}",idx.as_string(),c)); }
};

int multi_sigma_operator::get_dimensionality() const {
  return opdim;
};
int multi_sigma_operator::get_same_dimensionality(int dim) const {
  if (get_dimensionality()!=dim)
    throw(fmt::format("Sigma operator dimensionality mismatch: {} vs {}",
		      get_dimensionality(),dim));
  return dim;
};

#if 0
/*! Set a sigma operator in a specific dimension.
  This defines the multi sigma as operator based
*/
void multi_sigma_operator::set_operator(int id,sigma_operator op) {
  if (coord_based || sigma_based || struct_based)
    throw(string("Can not set operator: already of function type"));
  if (!op_based)
    operators = vector<sigma_operator>(opdim);
  op_based = true;
  operators.at(id) = op;
};

//! Set the operator in a dimension by converting from an ioperator
void multi_sigma_operator::set_operator(int id,ioperator op) {
  set_operator( id, sigma_operator(op) );
};

const sigma_operator &multi_sigma_operator::get_operator(int id) const {
  if (!op_based)
    throw(std::string("Can not get operator: not operator based"));
  return operators.at(id);
};

//! Define from coord-to-coord operator
multi_sigma_operator::multi_sigma_operator
    ( int dim, domain_coordinate(*c2c)(const domain_coordinate&) )
  : multi_sigma_operator(dim) {
  coord_oper = c2c; coord_based = true;
};

//! Define from coord-to-struct operator
multi_sigma_operator::multi_sigma_operator
( int dim, std::function< shared_ptr<multi_indexstruct>(const domain_coordinate&) > multisigma )
  : multi_sigma_operator(dim) {
  sigma_oper = multisigma; sigma_based = true;
};

//! Define from struct-to-struct operator
multi_sigma_operator::multi_sigma_operator
( int dim,std::function< shared_ptr<multi_indexstruct>(shared_ptr<multi_indexstruct>) > multisigma ) : multi_sigma_operator(dim) {
  struct_oper = multisigma; struct_based = true;
};
#endif

// bool multi_sigma_operator::is_point_operator() {
//   if (coord_oper!=nullptr) return true;
//   else if (sigma_oper!=nullptr || struct_oper!=nullptr) return false;
//   else {
//     for ( const auto &o : operators )
//       if (!o.is_point_operator())
// 	return false;
//     return true;
//   }
// };

/*!
  Operating on a domain coordinate gives a \ref multi_indexstruct
  by plain Cartesian product of the operate results in each dimension.
*/
#if 0
shared_ptr<multi_indexstruct> multi_sigma_operator::operate
    ( const domain_coordinate &point ) const {
  return operator_implementation(point);
  int dim = get_same_dimensionality(point.get_dimensionality());

  if (0) {
  } else if (op_based) {
    print("applying operators\n");
    auto idx = make_shared<multi_indexstruct>(dim) ;
    for (int id=0; id<dim; id++) { auto op = get_operator(id);
      try { idx->set_component(id, op.operate(point.coord(id)) );
      } catch (...) {
	throw(fmt::format("Could not operate <<{}>> set component {}",
			  op.as_string(),id));
      }
    }
    return idx;
  } else if (coord_based) {
    print("applying coord\n");
    try {
      return make_shared<multi_indexstruct>( coord_oper(point) ) ;
    } catch (std::string c) {
      throw(fmt::format("Multi sigma op failed on coord: {}",c)); }      
  } else if (sigma_based) {
    print("applying sigma\n");
    try { return sigma_oper(point);
    } catch (std::string c) {
      throw(fmt::format("Multi sigma op failed sigma based: {}",c)); }
  } else if (struct_based) {
    print("applying struct\n");
    try {
      return struct_oper( make_shared<multi_indexstruct>(point) ) ;
    } catch (std::string c) {
      throw(fmt::format("Multi sigma op failed struct based: {}",c)); }      
  } else
    throw(std::string("Unimplemented case multi_sigma operate"));
};

/*!
  A \ref multi_sigma_operator is strictly mapping coordinate to struct,
  but most of the time we will optimize this by mapping struct to struct.
  This routine covers three cases:
  - if \ref struct_oper is set, apply this once
  - if \ref coord_oper is set, we can apply this to contiguous by transforming the 
    first and last coordinate
  - if we have an array of single operators we apply those, one per dimension

  \todo the coord->indexstruct variant is not efficient. can we make shortcuts?
*/
shared_ptr<multi_indexstruct> multi_sigma_operator::operate
    ( shared_ptr<multi_indexstruct> idx ) {
  int dim = get_same_dimensionality(idx->get_dimensionality());
  if (0) {
  } else if (coord_based) {
    print("applying coord\n");
    if (idx->is_contiguous()) {
      try {
	auto
	  ofirst = coord_oper(idx->first_index_r()),
	  olast = coord_oper(idx->last_index_r());
	return shared_ptr<multi_indexstruct>
	  ( make_shared<contiguous_multi_indexstruct>( ofirst,olast ) );
      } catch (std::string c) {
	throw(fmt::format("Multi sigma failed by coord on struct {}: {}",idx->as_string(),c)); }
    } else
      throw(std::string("Can not point operate on non-contiguous"));
  } else if (sigma_based) {
    try {
      auto
	ofirst = sigma_oper( idx->first_index_r() )->first_index_r(),
	olast = sigma_oper( idx->last_index_r() )->last_index_r();
      return shared_ptr<multi_indexstruct>
	( make_shared<contiguous_multi_indexstruct>( ofirst,olast ) );
    } catch (std::string c) {
      throw(fmt::format("Multi sigma failed by sigma on struct {}: {}",idx->as_string(),c)); }
  } else if (struct_based) {
    try {
      return struct_oper(idx);
    } catch (std::string c) {
      throw(fmt::format("Multi sigma failed by struct on struct {}: {}",idx->as_string(),c)); }
  } else if (op_based) {
    auto opidx = make_shared<multi_indexstruct>(dim) ;
    for (int id=0; id<dim; id++) { 
      auto cpt = idx->get_component(id); auto op = get_operator(id);
      try {
	opidx->set_component(id,op.operate(cpt));
      } catch (std::string c) {
	print("Error <<{}>> applying <<{}>> in dimension {}",
		   c,op.as_string(),id,cpt->as_string());
	throw(fmt::format("Could not appy multi_sigma by operators on <<{}>>",
			  idx->as_string()));
      }
    }
    return opidx;
  } else if (1) {
    throw(std::string("And this is as far as we implemented it"));
  }
#if 0
  else if (is_coord_struct_operator()) {
    auto f = idx->first_index_r(), l = idx->last_index_r();
    if (is_shift_op()) {
      auto op_first = operate(f), op_last = operate(l);
      if (idx->is_contiguous()) {
	return new contiguous_multi_indexstruct
	  ( op_first->first_index_r(), op_last->last_index_r() );
      } else
	throw(fmt::format("Unimplemented case coord-struct on type {}",idx->type_as_string()));
    } else {
      throw(std::string("Really bad idea to enumerate coord-struct op"));
      shared_ptr<multi_indexstruct> r = new empty_multi_indexstruct(dim);
      domain_coordinate *ii = new domain_coordinate(dim);
      if (dim==1) {
	for (index_int i=f->coord(0); i<=l->coord(0); i++) {
	  ii->set(0,i);
	  r = r->struct_union( sigma_oper(ii) );
	}
      } else if (dim==2) {
	for (index_int i=f->coord(0); i<=l->coord(0); i++) {
	  ii->set(0,i);
	  for (index_int j=f->coord(1); j<=l->coord(1); j++) {
	    ii->set(1,j);
	    r = r->struct_union( sigma_oper(ii) );
	  }
	}
      } else if (dim==3) {
	for (index_int i=f->coord(0); i<=l->coord(0); i++) {
	  ii->set(0,i);
	  for (index_int j=f->coord(1); j<=l->coord(1); j++) {
	    ii->set(1,j);
	    for (index_int k=f->coord(2); k<=l->coord(2); k++) {
	      ii->set(2,k);
	      r = r->struct_union( sigma_oper(ii) );
	    }
	  }
	}
      } else
	throw(std::string("Cannot coord->indexstruct operate in dim>3"));
      return r;
    }
  } else {
    auto s = shared_ptr<multi_indexstruct>( new multi_indexstruct(dim) ); //(sim);
    for (int id=0; id<dim; id++) {
      shared_ptr<indexstruct<d>>
	oldstruct = idx->get_component(id),
	newstruct = shared_ptr<indexstruct<d>>( oldstruct->operate( operators[id] ) );
      s->set_component(id,newstruct);
    }
    return s;
  }
#endif
};
#endif
