/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2023
 ****
 **** indexstruct and ioperator package headers
 ****
 ****************************************************************/

#ifndef INDEXSTRUCT_H
#define INDEXSTRUCT_H

// C++ stuff
#include <functional>
#include <memory>
#include <vector>

// cppformat
#include "fmt/format.h"

// imp stuff
#include "utils.h"
#include "imp_coord.h"

// forward definitions
template<typename I,int d>
class empty_indexstruct;
template<typename I,int d>
class strided_indexstruct;
template<typename I,int d>
class contiguous_indexstruct;
template<typename I,int d>
class indexed_indexstruct;
template<typename I,int d>
class composite_indexstruct;
template<typename I,int d>
class ioperator;
template<typename I,int d>
class sigma_operator;
// hm.
template<typename I,int d>
class parallel_indexstruct;

/****************
 ****************
 **************** structs
 ****************
 ****************/

//! Size beyond which we detect contiguous in indexed and such
#define SMALLBLOCKSIZE 10

//! Sometimes we don't know what an indexstruct is.
enum class indexstruct_status { KNOWN, UNKNOWN };

template<typename I,int d>
class indexstruct : public std::enable_shared_from_this<indexstruct<I,d>> {
protected:
  indexstruct_status known_status{indexstruct_status::KNOWN};
public:
  indexstruct() {};
  bool is_known()             const { return known_status==indexstruct_status::KNOWN; };
  virtual bool is_empty( )     const { return volume()==0; };
  virtual bool is_contiguous() const { return 0; };
  virtual bool is_strided()    const { return 0; };
  virtual bool is_indexed()    const { return 0; };
  virtual bool is_composite()  const { return 0; };
  virtual std::string type_as_string() const { return std::string("none"); };
  int type_as_int() const;
  virtual void reserve( I s ) { return; };
  //  void report_unimplemented( char const * const c ) const;
  void report_unimplemented( std::string ) const;

  /*
   * Statistics
   */
  virtual coordinate<I,d> first_index() const {
    report_unimplemented("first_index"); return 0; };
  virtual coordinate<I,d> last_index()  const{
    report_unimplemented("last_index"); return 0; };
  virtual coordinate<I,d> last_actual_index()  const{
    report_unimplemented("last_actual_index"); return 0; };
  virtual I volume() const {
    report_unimplemented("volume"); return 0; };
  virtual I outer_volume() const;
  virtual const coordinate<I,d>& stride() const {
    throw(std::string("Indexstruct has no stride")); };
  virtual bool equals( const std::shared_ptr<indexstruct<I,d>>& idx ) const;
  
  virtual std::shared_ptr<indexstruct<I,d>> make_strided(bool=false) const;
  virtual I find( coordinate<I,d> ) const { throw(std::string("Can not be found")); };
  I location_of( std::shared_ptr<indexstruct<I,d>> inner ) const {
    return find(inner->first_index()); };
  //! Test for element containment; this can not be const because of optimizations.
  virtual bool contains_element( const coordinate<I,d>& idx ) const { return false; };
  virtual bool can_incorporate( coordinate<I,d> v ) const {
    report_unimplemented("can_incorporate"); return false; };
  bool contains_element_in_range( const coordinate<I,d>& idx) const;
  virtual bool contains( const std::shared_ptr<indexstruct<I,d>>& idx ) const {
    report_unimplemented("contains"); return false; };
  virtual bool is_strided_between_indices(I,I,I&,bool=false) const {
    report_unimplemented("strided between"); return false; };
  virtual bool disjoint( std::shared_ptr<indexstruct<I,d>> idx );
  virtual coordinate<I,d> get_ith_element( const I i ) const {
    throw(std::string("Get ith: not implemented")); };

  /*
   * Operations that yield a new indexstruct
   */
  virtual std::shared_ptr<indexstruct<I,d>> make_clone() const {
    throw(std::string("make_clone: Not implemented")); };
  virtual std::shared_ptr<indexstruct<I,d>> simplify() {
    return std::shared_ptr<indexstruct<I,d>>( make_clone() ); }; //shared_from_this();
  //! \todo try the shared_from_this again
  virtual std::shared_ptr<indexstruct<I,d>> force_simplify(bool=false) const {
    return std::shared_ptr<indexstruct<I,d>>( make_clone() ); }; //shared_from_this();
  virtual std::shared_ptr<indexstruct<I,d>> over_simplify() {
    return std::shared_ptr<indexstruct<I,d>>( make_clone() ); }; //shared_from_this();
  virtual std::shared_ptr<indexstruct<I,d>> add_element( coordinate<I,d> idx ) const;
  virtual void add_in_element( coordinate<I,d> idx );
  virtual std::shared_ptr<indexstruct<I,d>> translate_by( coordinate<I,d> shift ) const {
    report_unimplemented("translate_by"); return nullptr; };
  virtual bool has_intersect( std::shared_ptr<indexstruct<I,d>> idx ) {
    report_unimplemented("has_intersect"); return false; };

  // union and intersection
  virtual std::shared_ptr<indexstruct<I,d>> struct_union( std::shared_ptr<indexstruct<I,d>> idx,bool=false ) {
    throw(fmt::format("shared struct_union: Not implemented for {}",type_as_string())); };
  virtual std::shared_ptr<indexstruct<I,d>> struct_union( indexstruct* idx ,bool=false ) {
    return struct_union( idx->make_clone() ); };
  //return struct_union( std::shared_ptr<indexstruct<I,d>>{idx->make_clone()} ); };
  virtual std::shared_ptr<indexstruct<I,d>> split( std::shared_ptr<indexstruct<I,d>> idx ) {
    throw(fmt::format("shared split: Not implemented for {}",type_as_string())); };
  virtual std::shared_ptr<indexstruct<I,d>> intersect( std::shared_ptr<indexstruct<I,d>> idx ) {
    report_unimplemented("intersect"); return nullptr; };

  //! \todo make a variant that takes idx by const reference: it is often constructed ad-hoc
  virtual std::shared_ptr<indexstruct<I,d>> minus( std::shared_ptr<indexstruct<I,d>> idx ) const {
    report_unimplemented("minus"); return nullptr; };
  virtual std::shared_ptr<indexstruct<I,d>> truncate_left( coordinate<I,d> i );
  virtual std::shared_ptr<indexstruct<I,d>> truncate_right( coordinate<I,d>i );
  virtual std::shared_ptr<indexstruct<I,d>> relativize_to( std::shared_ptr<indexstruct<I,d>>,bool=false) {
    report_unimplemented("relativize_to"); return nullptr; };
  virtual std::shared_ptr<indexstruct<I,d>> convert_to_indexed() const {
    report_unimplemented("convert_to_indexed"); return nullptr; };
  virtual bool can_merge_with_type( std::shared_ptr<indexstruct<I,d>> idx) {
    return false; };

  // operate
  virtual std::shared_ptr<indexstruct<I,d>> operate( const ioperator<I,d> &op,bool=false ) const {
    throw(fmt::format("ioperate: Not implemented for struct type <<{}>>",type_as_string())); };
  virtual std::shared_ptr<indexstruct<I,d>> operate
      ( const ioperator<I,d>&,coordinate<I,d>,coordinate<I,d>) const;

  virtual std::shared_ptr<indexstruct<I,d>> operate( const sigma_operator<I,d> &op ) const {
    throw(fmt::format("sigma operate: Not implemented for struct type <<{}>>",type_as_string()));
  };
  virtual std::shared_ptr<indexstruct<I,d>> operate
      ( const sigma_operator<I,d> &op, coordinate<I,d>,coordinate<I,d>) const;
  virtual std::shared_ptr<indexstruct<I,d>> operate( const ioperator<I,d>&,std::shared_ptr<indexstruct<I,d>> ) const;
  virtual std::shared_ptr<indexstruct<I,d>> operate
      ( const sigma_operator<I,d>&,std::shared_ptr<indexstruct<I,d>> outer ) const;

  // Iterable functions
protected:
  int current_iterate{-1},last_iterate{-2};
  std::function< coordinate<I,d>(I) > ith_iterate
    { [] (I i) -> coordinate<I,d> { fmt::print("No ith_iterate defined\n"); return 0; } };
public:
  virtual indexstruct<I,d>& begin() {
    ith_iterate = [&] ( I i ) { return get_ith_element(i); };
    init_cur(); return *this;
  }
  virtual indexstruct<I,d>& end() { last_iterate = -1; return *this; }
  // the next 4 need to be pure virtual
  virtual void init_cur() { current_iterate = 0; last_iterate = -1; };
  virtual bool operator!=( indexstruct idx ) {
    //fmt::print("Comparing {} to {}\n",current_iterate,idx.last_iterate);
    return current_iterate<idx.last_iterate;
  };
  virtual coordinate<I,d> operator*() { return ith_iterate(current_iterate); };
  virtual void operator++() { current_iterate++; };

  // Stuff
  virtual std::string as_string() const {
    throw(std::string("as_string: Not implemented")); };
  virtual void debug_on() {};
  virtual void debug_off() {};
};

template<typename I,int d>
class unknown_indexstruct : public indexstruct<I,d> {
public:
  unknown_indexstruct() : indexstruct<I,d>() {
    this->known_status = indexstruct_status::UNKNOWN;
  };
  virtual std::string as_string() const override { return std::string("unknown"); };
};

template<typename I,int d>
class empty_indexstruct : public indexstruct<I,d> {
public:
  empty_indexstruct() {};
  virtual bool is_empty( ) const override { return true; };
  virtual std::string type_as_string() const override { return std::string("empty"); };
  virtual I volume() const override { return 0; };
  virtual std::shared_ptr<indexstruct<I,d>> make_clone() const override {
    return std::shared_ptr<indexstruct<I,d>>{ new empty_indexstruct() }; };
  virtual std::shared_ptr<indexstruct<I,d>> add_element( coordinate<I,d> idx ) const override;
  virtual void add_in_element( coordinate<I,d> idx ) override;
  virtual coordinate<I,d> first_index() const override {
    throw(std::string("No first index for empty")); };
  virtual coordinate<I,d> last_index()  const override {
    throw(std::string("No last index for empty")); };
  virtual coordinate<I,d> last_actual_index()  const override {
    throw(std::string("No last_actual index for empty")); };
  virtual coordinate<I,d> get_ith_element( const I i ) const override {
    throw(fmt::format("Can not get ith <<{}>> in empty",i)); };
  virtual std::shared_ptr<indexstruct<I,d>> translate_by( coordinate<I,d> shift ) const override {
    return this->make_clone(); };
  virtual bool has_intersect( std::shared_ptr<indexstruct<I,d>> idx ) override { return false; };
  virtual std::shared_ptr<indexstruct<I,d>> relativize_to( std::shared_ptr<indexstruct<I,d>>,bool=false) override{
    return std::shared_ptr<indexstruct<I,d>>( new empty_indexstruct() );
  };
  virtual std::shared_ptr<indexstruct<I,d>> minus( std::shared_ptr<indexstruct<I,d>> idx ) const override {
    return std::shared_ptr<indexstruct<I,d>>( make_clone() ); }; //shared_from_this(); };
  virtual std::shared_ptr<indexstruct<I,d>> truncate_left( coordinate<I,d> i ) override {
    return std::shared_ptr<indexstruct<I,d>>( make_clone() ); }; //shared_from_this(); };
  virtual std::shared_ptr<indexstruct<I,d>> truncate_right( coordinate<I,d> i ) override {
    return std::shared_ptr<indexstruct<I,d>>( make_clone() ); }; //shared_from_this(); };

  //! Operating on empty struct give the struct itself.
  virtual std::shared_ptr<indexstruct<I,d>> operate( const ioperator<I,d> &op ,bool=false) const override {
    return std::shared_ptr<indexstruct<I,d>>( make_clone() ); };
  virtual std::shared_ptr<indexstruct<I,d>> operate
      ( const ioperator<I,d>&,coordinate<I,d>,coordinate<I,d>)
    const override {
    return std::shared_ptr<indexstruct<I,d>>( make_clone() ); };

  virtual std::shared_ptr<indexstruct<I,d>> operate( const sigma_operator<I,d> &op ) const override {
    return std::shared_ptr<indexstruct<I,d>>( make_clone() ); };
  //! Operating on empty struct give the struct itself.
  virtual std::shared_ptr<indexstruct<I,d>> operate
      ( const sigma_operator<I,d> &op, coordinate<I,d>,coordinate<I,d> )
    const override {
    return std::shared_ptr<indexstruct<I,d>>( make_clone() ); };

  virtual std::shared_ptr<indexstruct<I,d>> struct_union( std::shared_ptr<indexstruct<I,d>> idx ,bool=false ) override;
  virtual bool can_merge_with_type( std::shared_ptr<indexstruct<I,d>> idx) override {
    return true; };
  virtual bool equals( const std::shared_ptr<indexstruct<I,d>>& idx ) const override {
    return idx->is_empty(); };
  virtual std::string as_string() const override { return std::string("empty"); };

  /*
   * Iterator functions
   */
public:
  virtual void init_cur() override {
    this->current_iterate = 0; this->last_iterate = volume(); };
  virtual indexstruct<I,d>& end() override {
    this->last_iterate = volume(); return *this; }
  // virtual empty_indexstruct& begin() override { return *this; };
  // virtual empty_indexstruct& end() override { return *this; };
  // virtual bool operator!=( indexstruct rr ) override { return 0; };
  // virtual void operator++() override { return; };
};

template<typename I,int d>
class strided_indexstruct : public indexstruct<I,d> {
protected:
  coordinate<I,d> first,last,stride_amount{coordinate<I,d>(1)};
public:
  strided_indexstruct(const I f,const I l,const I s);
  strided_indexstruct(const std::array<I,d> f,const std::array<I,d>  l,const I s);
  strided_indexstruct(const coordinate<I,d>&,const coordinate<I,d>&,const coordinate<I,d>&);
  strided_indexstruct(const coordinate<I,d> f,const coordinate<I,d>  l )
    : strided_indexstruct( f,l,coordinate<I,d>(1) ) {};
  /*
   * Statistics
   */
  coordinate<I,d> first_index() const override { return first; };
  //! non-inclusive upper bound, mostly to be correct for contiguous
  coordinate<I,d> last_index()  const override { return last+1; };
  //! actually last contained index
  coordinate<I,d> last_actual_index()  const override { return last; };
  virtual I outer_volume() const override;
  virtual I volume()  const override;
  virtual const coordinate<I,d>& stride() const override { return stride_amount; };
  virtual bool is_strided() const override { return true; };
  virtual std::string type_as_string() const override { return std::string("strided"); };
  virtual I find( coordinate<I,d> idx ) const override;
  virtual bool contains_element( const coordinate<I,d>& idx ) const override;
  virtual coordinate<I,d> get_ith_element( const I i ) const override;
  virtual bool can_incorporate( coordinate<I,d> v ) const override {
    return ( v==first-stride_amount || v==last+stride_amount); };
  //  virtual bool contains( indexstruct *idx ) override;
  virtual bool contains( const std::shared_ptr<indexstruct<I,d>>& idx ) const override;
  //  virtual bool disjoint( indexstruct *idx ) override;
  virtual bool disjoint( std::shared_ptr<indexstruct<I,d>> idx ) override;
  virtual bool equals( const std::shared_ptr<indexstruct<I,d>>& idx ) const override;

  /*
   * Operations that yield a new indexstruct
   */
  virtual std::shared_ptr<indexstruct<I,d>> convert_to_indexed() const override;
  virtual std::shared_ptr<indexstruct<I,d>> make_clone() const override {
    return std::shared_ptr<indexstruct<I,d>>{ new strided_indexstruct(first,last,stride_amount) };
  };
  virtual std::shared_ptr<indexstruct<I,d>> add_element( coordinate<I,d> idx ) const override;
  virtual std::shared_ptr<indexstruct<I,d>> translate_by( coordinate<I,d> shift ) const override;
  virtual bool has_intersect( std::shared_ptr<indexstruct<I,d>> idx ) override;
  virtual std::shared_ptr<indexstruct<I,d>> intersect( std::shared_ptr<indexstruct<I,d>> idx ) override;
  virtual std::shared_ptr<indexstruct<I,d>> minus( std::shared_ptr<indexstruct<I,d>> idx ) const override;
  virtual std::shared_ptr<indexstruct<I,d>> relativize_to( std::shared_ptr<indexstruct<I,d>>,bool=false) override;
  virtual std::shared_ptr<indexstruct<I,d>> struct_union( std::shared_ptr<indexstruct<I,d>> idx ,bool=false ) override;
  virtual bool can_merge_with_type( std::shared_ptr<indexstruct<I,d>> idx) override;
  virtual std::shared_ptr<indexstruct<I,d>> split( std::shared_ptr<indexstruct<I,d>> idx ) override;

  /*
   * Iterator functions
   */
  virtual indexstruct<I,d>& end() override {
    this->last_iterate = volume(); return *this; }
  virtual void init_cur() override {
    this->current_iterate = 0; this->last_iterate = volume();
    //fmt::print("Setting last iterate to {}\n",last_iterate);
  };

  /*
   * Operate
   */
  virtual std::shared_ptr<indexstruct<I,d>> operate( const ioperator<I,d> &op ,bool=false) const override;
  virtual std::shared_ptr<indexstruct<I,d>> operate( const sigma_operator<I,d> &op ) const override;

  virtual std::string as_string() const override;
};

template<typename I,int d>
class contiguous_indexstruct : public strided_indexstruct<I,d> {
public:
  //! Constructor from first/last delegates to strided with stride 1
  contiguous_indexstruct(const coordinate<I,d> s,const coordinate<I,d> l)
    : strided_indexstruct<I,d>(s,l,1) {};
  //! Constructing from a single coordinate means using it as first and last.
  contiguous_indexstruct(const coordinate<I,d> f)
    : contiguous_indexstruct<I,d>(f,f) {};
  //! Constructor from first/last arrays delegates to strided. \todo delegate to coordinate version?
  contiguous_indexstruct( const std::array<I,d> s,const std::array<I,d> l )
  //  : strided_indexstruct<I,d>(s,l,1) {};
    : contiguous_indexstruct( coordinate<I,d>(s), coordinate<I,d>(l) ) {};

  virtual bool is_contiguous() const override { return true; };
  virtual std::string type_as_string() const override { return std::string("contiguous"); };
  std::shared_ptr<indexstruct<I,d>> make_clone() const override {
    return std::shared_ptr<indexstruct<I,d>>
      { new contiguous_indexstruct(this->first,this->last) };
  };
  virtual std::string as_string() const override;
};

/*!
  An indexed indexstruct contains a standard vector of indices.
 */
template<typename I,int d>
class indexed_indexstruct : public indexstruct<I,d> {
protected:
  std::vector<coordinate<I,d>> indices;
public:
  indexed_indexstruct() {}; //!< Create an empty indexed struct
  indexed_indexstruct( const std::vector<coordinate<I,d>> idxs );
  indexed_indexstruct( const std::vector<I> idxs );
  bool is_indexed() const override { return true; };
  virtual std::shared_ptr<indexstruct<I,d>> simplify() override;
  virtual std::shared_ptr<indexstruct<I,d>> force_simplify(bool=false) const override;
  virtual std::string type_as_string() const override { return std::string("indexed"); };
  virtual void reserve( I s ) override { indices.reserve(s); };

  /*
   * Iterator functions
   */
  virtual void init_cur() override {
    this->current_iterate = 0; this->last_iterate = volume(); };
  virtual indexstruct<I,d>& end() override {
    this->last_iterate = volume(); return *this; }

  /*
   * Statistics
   */
  coordinate<I,d> first_index() const override {
    if (indices.size()==0) throw(std::string("Can not ask first for empty indexed"));
    return indices.at(0); };
  // coordinate<I,d> last_index() needs to stay undefined for indexed
  //! actually last contained index
  coordinate<I,d> last_actual_index() const override {
    if (indices.size()==0) throw(std::string("Can not ask last_actual for empty indexed"));
    return indices.back(); };
  I volume() const override { return indices.size(); };
  virtual I find( coordinate<I,d> idx ) const override {
    for (int i=0; i<indices.size(); i++)
      if (indices[i]==idx) return i;
    // for (auto loc=this->begin_at_value(idx); loc!=this->end(); ++loc)
    //   if (*loc==idx) return loc.search_loc();
    throw(std::string("Index to find is out of range"));
  };
  virtual bool contains_element( const coordinate<I,d>& idx ) const override {
    if (indices.size()==0) return false;
    for (int i=0; i<indices.size(); i++)
      if (indices[i]==idx) return true;
    return false;
  };
  virtual coordinate<I,d> get_ith_element( const I i ) const override;
  virtual std::shared_ptr<indexstruct<I,d>> make_strided(bool=false) const override;
  virtual bool is_strided_between_indices(I,I,I&,bool=false) const override;
  virtual bool contains( const std::shared_ptr<indexstruct<I,d>>& idx ) const override;
  virtual bool can_incorporate( coordinate<I,d> v ) const override { return true; } //!< We can always add an index.
  virtual bool disjoint( std::shared_ptr<indexstruct<I,d>> idx ) override;
  virtual bool equals( const std::shared_ptr<indexstruct<I,d>>& idx ) const override;

  /*
   * Operations that yield a new indexstruct
   */
  virtual std::shared_ptr<indexstruct<I,d>> make_clone() const override {
    return std::shared_ptr<indexstruct<I,d>>{ new indexed_indexstruct(indices) }; };
  virtual std::shared_ptr<indexstruct<I,d>> convert_to_indexed() const override {
    return std::shared_ptr<indexstruct<I,d>>( this->make_clone() ); }; //shared_from_this(); };
  virtual std::shared_ptr<indexstruct<I,d>> add_element( coordinate<I,d> idx ) const override;
  virtual void add_in_element( coordinate<I,d> idx ) override;
  virtual std::shared_ptr<indexstruct<I,d>> translate_by( coordinate<I,d> shift ) const override;

  virtual std::shared_ptr<indexstruct<I,d>> minus( std::shared_ptr<indexstruct<I,d>> idx ) const override;
  virtual std::shared_ptr<indexstruct<I,d>> relativize_to( std::shared_ptr<indexstruct<I,d>>,bool=false) override;
  virtual std::shared_ptr<indexstruct<I,d>> struct_union( std::shared_ptr<indexstruct<I,d>> idx ,bool=false ) override;
  virtual bool can_merge_with_type( std::shared_ptr<indexstruct<I,d>> idx) override {
    return idx->is_indexed(); };
  virtual std::shared_ptr<indexstruct<I,d>> intersect( std::shared_ptr<indexstruct<I,d>> idx ) override;

  /*
   * Operate
   */
  virtual std::shared_ptr<indexstruct<I,d>> operate( const ioperator<I,d> &op ,bool=false) const override;

  virtual std::string as_string() const override;
};

/*!
  A composite indexstruct is a vector of non-composite indexstructs.
  The components are disjoint, but not necessarily sorted.
  \todo the structs vector should not be a pointer
*/
template<typename I,int d>
class composite_indexstruct : public indexstruct<I,d> {
private:
  std::vector<std::shared_ptr<indexstruct<I,d>>> structs;
public:
  composite_indexstruct() {};
  virtual bool is_composite() const override { return true; };
  virtual std::string type_as_string() const override { return std::string("composite"); };
  void push_back( std::shared_ptr<indexstruct<I,d>> idx );
  const std::vector<std::shared_ptr<indexstruct<I,d>>> &get_structs() const { return structs; };
  virtual std::shared_ptr<indexstruct<I,d>> make_clone() const override;

  /*
   * Statistics
   */
  virtual bool is_empty() const override {
    for (auto s : structs )
      if (!s->is_empty()) return false; return true; };
  virtual coordinate<I,d> first_index() const override;
  // virtual coordinate<I,d> last_index() undefined for composite
  virtual coordinate<I,d> last_actual_index()  const override;
  virtual I volume()  const override;

  virtual bool contains_element( const coordinate<I,d>& idx ) const override;
  virtual bool contains( const std::shared_ptr<indexstruct<I,d>>& idx ) const override;
  virtual I find( coordinate<I,d> idx ) const override ;
  virtual coordinate<I,d> get_ith_element( const I i ) const override;
  virtual std::shared_ptr<indexstruct<I,d>> struct_union( std::shared_ptr<indexstruct<I,d>> ,bool=false ) override;
  virtual std::shared_ptr<indexstruct<I,d>> intersect( std::shared_ptr<indexstruct<I,d>> idx ) override;
  virtual std::shared_ptr<indexstruct<I,d>> convert_to_indexed() const override;
  virtual std::shared_ptr<indexstruct<I,d>> force_simplify(bool=false) const override;
  virtual std::shared_ptr<indexstruct<I,d>> over_simplify() override;
  virtual std::shared_ptr<indexstruct<I,d>> relativize_to( std::shared_ptr<indexstruct<I,d>>,bool=false) override ;
  virtual std::shared_ptr<indexstruct<I,d>> minus( std::shared_ptr<indexstruct<I,d>> idx ) const override;
  //  virtual bool disjoint( indexstruct *idx ) override;
  virtual bool disjoint( std::shared_ptr<indexstruct<I,d>> idx ) override;
  virtual bool equals( const std::shared_ptr<indexstruct<I,d>>& idx ) const override;

  virtual std::shared_ptr<indexstruct<I,d>> operate( const ioperator<I,d> &op ,bool=false) const override;

  virtual std::string as_string() const override;

  /*
   * Iterator functions
   */
  virtual void init_cur() override {
    this->current_iterate = 0; this->last_iterate = volume(); };
  virtual indexstruct<I,d>& end() override {
    this->last_iterate = volume(); return *this; }
// protected:
//   int cur_struct = 0;
// public:
//   void init_cur();
//   composite_indexstruct& begin() override { init_cur(); return *this; };
//   composite_indexstruct& end() override { return *this; };
//   virtual bool operator!=( indexstruct rr ) override;
//   void operator++() override;
//   I operator*() override { return *( *structs.at(cur_struct).get() ); };
};

/****************
 ****************
 **************** ioperator
 ****************
 ****************/

enum class iop_type {INVALID, NONE, SHIFT_REL,SHIFT_ABS, MULT, DIV, CONTDIV, FUNC };

/*!
 An ioperator is a function from Int to Int. This will be used to operate on
 \ref distribution objects. See \ref ioperator::ioperator for a description
 of the allowed operations.
 \todo change the function to int->int
*/
template<typename I, int d>
class ioperator {
protected:
  iop_type type{iop_type::INVALID};
  int mod{0}, baseop{0};
  coordinate<I,d> by;
  std::function< I(I) > func;
public:
  //! This constructor is  needed for the derived classes
  ioperator() {};
  ioperator( std::string op );
  ioperator( std::string op,const coordinate<I,d>& amt );
  //! In the most literal interpretation, we operate an actual function pointer.
  ioperator( I(*f)(I) ) { type = iop_type::FUNC; func = f; };
  ioperator( std::function< I(I) > f ) { type = iop_type::FUNC; func = f; };
  // operate
  coordinate<I,d> operate( const coordinate<I,d>& ) const;
  // operate and truncate
  coordinate<I,d> operate( const coordinate<I,d>&,const coordinate<I,d>& ) const;
  // operate
  coordinate<I,d> inverse_operate( const coordinate<I,d>& ) const;
  // operate and truncate
  coordinate<I,d> inverse_operate( const coordinate<I,d>&,const coordinate<I,d>& ) const;
  bool is_none_op()        const { return type==iop_type::NONE; };
  bool is_shift_op()       const {
    return type==iop_type::SHIFT_REL || type==iop_type::SHIFT_ABS; };
  bool is_shift_to()       const { return type==iop_type::SHIFT_ABS; };
  bool is_right_shift_op() const { return is_shift_op() && (by.at(0)>0); };
  bool is_left_shift_op()  const { return is_shift_op() && (by.at(0)<0); };
  bool is_modulo_op()      const { return is_shift_op() && mod; };
  bool is_bump_op()        const { return is_shift_op() && !mod; };
  bool is_restrict_op()    const { return type==iop_type::MULT; }; // VLE don't like the name. lose
  bool is_mult_op()        const { return type==iop_type::MULT; };
  bool is_base_op()        const { return baseop; };
  bool is_div_op()         const { return type==iop_type::DIV; };
  bool is_contdiv_op()     const { return type==iop_type::CONTDIV; };
  bool is_function_op()    const { return type==iop_type::FUNC; };
  const coordinate<I,d>& amount() const { return by; };
  std::string type_string() const {
    if (type==iop_type::INVALID) return "INVALID";
    if (type==iop_type::NONE) return "NONE";
    if (type==iop_type::SHIFT_REL) return "SHIFT_REL";
    if (type==iop_type::SHIFT_ABS) return "SHIFT_ABS";
    if (type==iop_type::MULT) return "MULT";
    if (type==iop_type::DIV) return "DIV";
    if (type==iop_type::FUNC) return "FUNC";
    return "MISSING"; };  
  std::string as_string() const;
};

template<typename I,int d>
class shift_operator : public ioperator<I,d> {
public:
  shift_operator( coordinate<I,d> by,bool relative=true )
    : ioperator<I,d>() {
    if (relative)
      this->type = iop_type::SHIFT_REL;
    else
      this->type = iop_type::SHIFT_ABS;
    this->by = by;
  };
};

template<typename I,int d>
class mult_operator : public ioperator<I,d> {
public:
  mult_operator( I n )
    : ioperator<I,d>() {
    this->type = iop_type::MULT;
    this->by = n; };
};

template<typename I,int d>
class multi_indexstruct;

/*!
  A sigma operator takes a point and gives a structure;
*/
template<typename I,int d>
class sigma_operator {
protected:
  std::function< std::shared_ptr<indexstruct<I,d>>( coordinate<I,d> ) > func;
  bool lambda_i{false};
  std::function< std::shared_ptr<indexstruct<I,d>>( const indexstruct<I,d>& ) > sfunc;
  bool lambda_s{false};
  ioperator<I,d> point_func;
  bool lambda_p{false};
public:
  //! Default creator
  sigma_operator() {};
  //! Create from function pointer
  sigma_operator( std::shared_ptr<indexstruct<I,d>>(*f)(coordinate<I,d>) ) {
    func = f; lambda_i = true; };
  //! Create from point lambda
  sigma_operator( std::function< std::shared_ptr<indexstruct<I,d>>(coordinate<I,d>) > f ) {
    func = f; lambda_i = true; };
  //! Create from struct lambda
  sigma_operator( std::function< std::shared_ptr<indexstruct<I,d>>(const indexstruct<I,d>&) > f ) {
    sfunc = f; lambda_s = true; };
  //! Create from ioperator
  sigma_operator( std::shared_ptr<ioperator<I,d>> f ) {
    throw("please no sigma from pointer to iop\n");
    point_func = *(f.get()); lambda_p = true; };
  //! Create from ioperator
  sigma_operator( ioperator<I,d> f ) {
    point_func = f; lambda_p = true; };
  //! Create from scalar lambda \todo this should be the pointfunc?
  sigma_operator( std::function< I(I) > f ) {
    throw("You sure this is not the pointfunc?\n");
    // func = [f] ( coordinate<I,d> i ) -> std::shared_ptr<indexstruct<I,d>> {
    //   return std::shared_ptr<indexstruct<I,d>>{new contiguous_indexstruct<I,d>( f(i) )};
    // };
    lambda_i = true;
  };
  //! Does this produce a single point?
  bool is_point_operator() const { return lambda_p; };
  bool is_struct_operator() const { return lambda_s; };
  std::shared_ptr<indexstruct<I,d>> struct_apply( const indexstruct<I,d>& i) const {
    return sfunc(i); };
  const ioperator<I,d> &point_operator() const {
    if (!is_point_operator()) throw(std::string("Is not point operator"));
    return point_func; };
  std::shared_ptr<indexstruct<I,d>> operate( I i ) const;
  std::shared_ptr<indexstruct<I,d>> operate( std::shared_ptr<indexstruct<I,d>> idx ) const;
  std::string as_string() const;
};

/*
 * And here is the PIMPL class!
 */

template<typename I,int d>
class indexstructure {
private:
  std::shared_ptr<indexstruct<I,d>> strct{nullptr};
  indexstruct_status known_status{indexstruct_status::KNOWN};
public:
  //! Default constructor: set to pointer of unknown indexstruct
  indexstructure() {
    strct = std::shared_ptr<indexstruct<I,d>>( new unknown_indexstruct<I,d>() ); };
  //! All other constructors, set to pointer of copy of that indexstruct
  indexstructure( contiguous_indexstruct<I,d> c )
    : indexstructure<I,d>
        ( std::shared_ptr<indexstruct<I,d>>( new contiguous_indexstruct<I,d>(c) ) ) {};
  indexstructure( strided_indexstruct<I,d> c )
    : indexstructure<I,d>
        ( std::shared_ptr<indexstruct<I,d>>( new strided_indexstruct<I,d>(c) ) )    {};
  indexstructure( indexed_indexstruct<I,d> c )
    : indexstructure<I,d>
        ( std::shared_ptr<indexstruct<I,d>>( new indexed_indexstruct<I,d>(c) ) )    {};
  indexstructure( composite_indexstruct<I,d> c )
    : indexstructure<I,d>
        ( std::shared_ptr<indexstruct<I,d>>( new composite_indexstruct<I,d>(c) ) )  {
    fmt::print("composite type=<{}> strct composite: <{}> this composite: <{}>\n",
	       strct->type_as_string(),strct->is_composite(),is_composite());
  };
  //! Constructor from polymorphic pointer to indexstruct: store pointer.
  indexstructure( std::shared_ptr<indexstruct<I,d>> idx ) { strct = idx; };
  
  bool is_known()              const {    return strct->is_known(); };
  virtual bool is_empty( )     const {    return strct->is_empty(); };
  virtual bool is_contiguous() const {    return strct->is_contiguous(); };
  virtual bool is_strided()    const {    return strct->is_strided(); };
  virtual bool is_indexed()    const {    return strct->is_indexed(); };
  virtual bool is_composite()  const {    return strct->is_composite(); };
  virtual std::string type_as_string()       const {
    return strct->type_as_string(); };
  virtual void reserve( I s )              {
    return strct->reserve(s); };
  void report_unimplemented( const char *c ) const {
    return strct->report_unimplemented(c); };
  indexstructure<I,d> make_clone() {
    return indexstructure<I,d>( *this ); };
  const std::vector<std::shared_ptr<indexstruct<I,d>>> get_structs() const;

  /*
   * Multi
   */
  //  void push_back( contiguous_indexstruct<I,d> &&idx );
  void push_back( indexstructure<I,d> idx );

  /*
   * Statistics
   */
  virtual coordinate<I,d> first_index() const {    return strct->first_index(); };
  //! actually last contained index
  virtual coordinate<I,d> last_index()  const {    return strct->last_index(); };
  //! non-inclusive upper bound, mostly to be correct for contiguous
  virtual coordinate<I,d> last_actual_index()  const { return strct->last_actual_index(); };
  I volume()                            const {    return strct->volume(); };
  I outer_volume()                      const {    return strct->outer_volume(); };
  virtual const coordinate<I,d>& stride() const {    return strct->stride(); };
  virtual bool equals( const std::shared_ptr<indexstruct<I,d>>& idx ) const {
    return strct->equals(idx); };
  virtual bool operator==( std::shared_ptr<indexstruct<I,d>> idx ) const {
    return strct->equals(idx); };
  virtual bool equals( const indexstructure& idx ) const {
    return strct->equals(idx.strct); };
  
  virtual I find( coordinate<I,d> idx )  const {    return strct->find(idx); };
  I location_of( std::shared_ptr<indexstruct<I,d>> inner ) const {
    return strct->location_of(inner); };
  I location_of( indexstructure &inner ) const {    return strct->location_of(inner.strct); };
  I location_of( indexstructure &&inner ) {
    return strct->location_of(inner.strct); };
  virtual bool contains_element( const coordinate<I,d>& idx ) const {
    return strct->contains_element(idx); };
  bool contains_element_in_range( const coordinate<I,d>& idx) const;
  virtual bool contains( const std::shared_ptr<indexstruct<I,d>>& idx ) {
    return strct->contains(idx); };
  virtual bool contains( const indexstructure& idx ) const {
    return strct->contains(idx.strct); };
  virtual bool disjoint( std::shared_ptr<indexstruct<I,d>> idx ) {
    return strct->disjoint(idx); };
  virtual bool disjoint( indexstructure &idx ) {
    return strct->disjoint(idx.strct); };
  virtual bool disjoint( indexstructure &&idx ) {
    return strct->disjoint(idx.strct); };
  virtual bool can_incorporate( coordinate<I,d> v ) const {
    return strct->can_incorporate(v); };
  virtual coordinate<I,d> get_ith_element( const I i ) const {
    return strct->get_ith_element(i); };
  virtual bool is_strided_between_indices(I f,I l,I& s,bool trace=false) const {
    return strct->is_strided_between_indices(f,l,s,trace); };

  /*
   * Operations that yield a new indexstruct
   */
  void simplify() { strct = strct->simplify(); };
  void force_simplify(bool trace=false) { strct = strct->force_simplify(trace); };
  void over_simplify() { strct = strct->over_simplify(); };
  void add_element( coordinate<I,d> idx ) { strct = strct->add_element(idx); };
  void add_in_element( coordinate<I,d> idx ) { strct->add_in_element(idx); };
  void translate_by( coordinate<I,d> shift ) { strct = strct->translate_by(shift); };
  bool has_intersect( std::shared_ptr<indexstruct<I,d>> idx ) {
    return strct->has_intersect(idx); };
  //! make strided, or throw if not possible
  void make_strided() { strct = strct->make_strided(); };

  // union and intersection
  void struct_union( indexstructure &idx ,bool=false ) {   strct = strct->struct_union(idx.strct); };
  void struct_union( indexstructure &&idx ,bool=false ) {  strct = strct->struct_union(idx.strct); };
  void split( std::shared_ptr<indexstruct<I,d>> idx ) {    strct = strct->split(idx); };
  void split( indexstructure &idx ) {                      strct = strct->split(idx.strct); };
  void intersect( std::shared_ptr<indexstruct<I,d>> idx ) {strct = strct->intersect(idx); };
  void intersect( indexstructure &idx ) {                  strct = strct->intersect(idx.strct); };

  void minus( std::shared_ptr<indexstruct<I,d>> idx ) {    strct = strct->minus(idx); };
  void minus( indexstructure &idx ) {                      strct = strct->minus(idx.strct); };
  void truncate_left( coordinate<I,d> i ) { strct = strct->truncate_left(i); };
  void truncate_right( coordinate<I,d> i ) { strct = strct->truncate_right(i); };
  void relativize_to( std::shared_ptr<indexstruct<I,d>> idx ) { strct = strct->relativize_to(idx); };
  void relativize_to( indexstructure &idx ) { strct = strct->relativize_to(idx.strct); };
  void convert_to_indexed() { strct = strct->convert_to_indexed(); };

  // operate
  auto operate( const ioperator<I,d> &op ) { return indexstructure(strct->operate(op)); };
  auto operate( const ioperator<I,d> &&op ) { return indexstructure(strct->operate(op)); };
  auto operate( const ioperator<I,d> &op,I i0,I i1) {
    return indexstructure(strct->operate(op,i0,i1)); };

  auto operate( const sigma_operator<I,d> &op ) { return indexstructure(strct->operate(op)); };
  auto operate( const sigma_operator<I,d> &op, I lo,I hi ){
    return indexstructure(strct->operate(op,lo,hi)); };
  auto operate( const ioperator<I,d> &op,std::shared_ptr<indexstruct<I,d>> outer ) {
    return indexstructure(strct->operate(op,outer)); };
  auto operate( const sigma_operator<I,d> &op,std::shared_ptr<indexstruct<I,d>> outer ) {
    return indexstructure(strct->operate(op,outer)); };

#if 0
  // Iterable functions
  virtual indexstruct& begin() { init_cur(); return *this; }
  // the next 4 need to be pure virtual
  virtual void init_cur() {};
  virtual indexstruct& end() override { last_iterate = volume(); return *this; }
  virtual bool operator!=( indexstruct idx ) { //printf("default test\n");
    return 0; };
  virtual I operator*() { return -1; };
  virtual void operator++() { return; };
#endif
  
  // Stuff
  std::string as_string() const;
  virtual void debug_on() {};
  virtual void debug_off() {};
};

template<typename I,int d>
struct fmt::formatter<std::shared_ptr<indexstruct<I,d>>> {
 constexpr
 auto parse(format_parse_context& ctx)
       -> decltype(ctx.begin()) {
   auto it = ctx.begin(),
     end = ctx.end();
   if (it != end && *it != '}')
     throw format_error("invalid format");
   return it;
  }
  template <typename FormatContext>
  auto format
      (const std::shared_ptr<indexstruct<I,d>>& p, FormatContext& ctx)
        -> decltype(ctx.out()) {
    return fmt::format_to(ctx.out(),"{}", p->as_string());
  }
};

template<typename I, int d>
std::ostream &operator<<(std::ostream &os,const std::shared_ptr<indexstruct<I,d>> &s) {
  os << s->as_string();
  return os;
};

template<typename I, int d>
std::ostream &operator<<(std::ostream &os,const std::shared_ptr<unknown_indexstruct<I,d>> &s) {
  os << s->as_string();
  return os;
};

template<typename I, int d>
std::ostream &operator<<(std::ostream &os,const std::shared_ptr<empty_indexstruct<I,d>> &s) {
  os << s->as_string();
  return os;
};

template<typename I, int d>
std::ostream &operator<<(std::ostream &os,const std::shared_ptr<strided_indexstruct<I,d>> &s) {
  os << s->as_string();
  return os;
};

template<typename I, int d>
std::ostream &operator<<(std::ostream &os,const std::shared_ptr<contiguous_indexstruct<I,d>> &s) {
  os << s->as_string();
  return os;
};
template<typename I, int d>

std::ostream &operator<<(std::ostream &os,const std::shared_ptr<indexed_indexstruct<I,d>> &s) {
  os << s->as_string();
  return os;
};

template<typename I, int d>
std::ostream &operator<<(std::ostream &os,const std::shared_ptr<composite_indexstruct<I,d>> &s) {
  os << s->as_string();
  return os;
};


#endif

