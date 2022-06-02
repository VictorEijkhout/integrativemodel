/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2022
 ****
 **** indexstruct and ioperator package headers
 ****
 ****************************************************************/

#ifndef INDEXSTRUCT_H
#define INDEXSTRUCT_H

// C stuff
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
// C++ stuff
#include <functional>
#include <memory>
#include <vector>
// cppformat
#include "fmt/format.h"
using fmt::format_to;
using fmt::to_string;

// imp stuff
#include "utils.h"

// forward definitions
class empty_indexstruct;
class strided_indexstruct;
class contiguous_indexstruct;
class indexed_indexstruct;
class composite_indexstruct;
class ioperator;
class sigma_operator;
// hm.
class parallel_indexstruct;
class domain_coordinate;
class domain_coordinate1d;

/****************
 ****************
 **************** structs
 ****************
 ****************/

//! Size beyond which we detect contiguous in indexed and such
#define SMALLBLOCKSIZE 10

//! Sometimes we don't know what an indexstruct is.
enum class indexstruct_status { KNOWN, UNKNOWN };

class indexstructure;
class indexstruct : public std::enable_shared_from_this<indexstruct> {
protected:
  indexstruct_status known_status{indexstruct_status::KNOWN};
public:
  indexstruct() {};
  bool is_known()             const { return known_status==indexstruct_status::KNOWN; };
  virtual bool is_empty( )     const { return local_size()==0; };
  virtual bool is_contiguous() const { return 0; };
  virtual bool is_strided()    const { return 0; };
  virtual bool is_indexed()    const { return 0; };
  virtual bool is_composite()  const { return 0; };
  virtual std::string type_as_string() const { return std::string("none"); };
  int type_as_int() const;
  virtual void reserve( index_int s ) { return; };
  void report_unimplemented( const char *c ) const;

  /*
   * Statistics
   */
  virtual index_int first_index() const {
    report_unimplemented("first_index"); return 0; };
  virtual index_int last_index()  const{
    report_unimplemented("last_index"); return 0; };
  virtual index_int local_size()  const {
    report_unimplemented("local_size"); return 0; };
  index_int volume() const { return local_size(); };
  index_int outer_size() const { return last_index()-first_index()+1; };
  virtual int stride() const { throw(std::string("Indexstruct has no stride")); };
  virtual bool equals( std::shared_ptr<indexstruct> idx ) const;
  
  virtual index_int find( index_int idx ) const { throw(std::string("Can not be found")); };
  index_int location_of( std::shared_ptr<indexstruct> inner ) const {
    return find(inner->first_index()); };
  //! Test for element containment; this can not be const because of optimizations.
  virtual bool contains_element( index_int idx ) const { return false; };
  bool contains_element_in_range(index_int idx) const;
  virtual bool contains( std::shared_ptr<indexstruct> idx ) const {
    report_unimplemented("contains"); return false; };
  virtual bool disjoint( std::shared_ptr<indexstruct> idx );
  virtual int can_incorporate( index_int v ) { return 0; } //!< Is this a wise default?
  virtual index_int get_ith_element( const index_int i ) const {
    throw(std::string("Get ith: not implemented")); };

  /*
   * Operations that yield a new indexstruct
   */
  virtual std::shared_ptr<indexstruct> make_clone() const {
    throw(std::string("make_clone: Not implemented")); };
  virtual std::shared_ptr<indexstruct> simplify() {
    return std::shared_ptr<indexstruct>( make_clone() ); }; //shared_from_this();
  //! \todo try the shared_from_this again
  virtual std::shared_ptr<indexstruct> force_simplify() const {
    return std::shared_ptr<indexstruct>( make_clone() ); }; //shared_from_this();
  virtual std::shared_ptr<indexstruct> over_simplify() {
    return std::shared_ptr<indexstruct>( make_clone() ); }; //shared_from_this();
  virtual std::shared_ptr<indexstruct> add_element( const index_int idx ) {
    throw(std::string("add_element: Not implemented")); };
  virtual void addin_element( const index_int idx ) {
    throw(std::string("add_element: Not implemented")); };
  virtual std::shared_ptr<indexstruct> translate_by( index_int shift ) {
    report_unimplemented("translate_by"); return nullptr; };
  virtual bool has_intersect( std::shared_ptr<indexstruct> idx ) {
    report_unimplemented("has_intersect"); return false; };

  // union and intersection
  virtual std::shared_ptr<indexstruct> struct_union( std::shared_ptr<indexstruct> idx ) {
    throw(fmt::format("shared struct_union: Not implemented for {}",type_as_string())); };
  virtual std::shared_ptr<indexstruct> struct_union( indexstruct* idx ) {
    return struct_union( idx->make_clone() ); };
  //return struct_union( std::shared_ptr<indexstruct>{idx->make_clone()} ); };
  virtual std::shared_ptr<indexstruct> split( std::shared_ptr<indexstruct> idx ) {
    throw(fmt::format("shared split: Not implemented for {}",type_as_string())); };
  virtual std::shared_ptr<indexstruct> intersect( std::shared_ptr<indexstruct> idx ) {
    report_unimplemented("intersect"); return nullptr; };

  //! \todo make a variant that takes idx by const reference: it is often constructed ad-hoc
  virtual std::shared_ptr<indexstruct> minus( std::shared_ptr<indexstruct> idx ) const {
    report_unimplemented("minus"); return nullptr; };
  virtual std::shared_ptr<indexstruct> truncate_left( index_int i );
  virtual std::shared_ptr<indexstruct> truncate_right( index_int i );
  virtual std::shared_ptr<indexstruct> relativize_to( std::shared_ptr<indexstruct>,bool=false) {
    report_unimplemented("relativize_to"); return nullptr; };
  virtual std::shared_ptr<indexstruct> convert_to_indexed() const {
    report_unimplemented("convert_to_indexed"); return nullptr; };
  virtual int can_merge_with_type( std::shared_ptr<indexstruct> idx) { return 0; };

  // operate
  virtual std::shared_ptr<indexstruct> operate( const ioperator &op ) const {
    throw(fmt::format("ioperate: Not implemented for struct type <<{}>>",type_as_string())); };
  virtual std::shared_ptr<indexstruct> operate( const ioperator &&op ) const {
    throw(fmt::format("ioperate rv: Not implemented for type <<{}>>",type_as_string())); };
  virtual std::shared_ptr<indexstruct> operate( const ioperator&,index_int,index_int) const;

  virtual std::shared_ptr<indexstruct> operate( const sigma_operator &op ) const {
    throw(fmt::format("sigma operate: Not implemented for struct type <<{}>>",type_as_string()));
  };
  virtual std::shared_ptr<indexstruct> operate( const sigma_operator &op, index_int lo,index_int hi ) const;
  virtual std::shared_ptr<indexstruct> operate( const ioperator&,std::shared_ptr<indexstruct> ) const;
  virtual std::shared_ptr<indexstruct> operate
      ( const sigma_operator&,std::shared_ptr<indexstruct> outer ) const;

  // Iterable functions
protected:
  int current_iterate{-1},last_iterate{-2};
  std::function< index_int(index_int) > ith_iterate
    { [] (index_int i) -> index_int { fmt::print("No ith_iterate defined\n"); return 0; } };
public:
  virtual indexstruct& begin() {
    ith_iterate = [&] (int i) -> index_int { return get_ith_element(i); };
    init_cur(); return *this;
  }
  virtual indexstruct& end() { last_iterate = -1; return *this; }
  // the next 4 need to be pure virtual
  virtual void init_cur() { current_iterate = 0; last_iterate = -1; };
  virtual bool operator!=( indexstruct idx ) {
    //fmt::print("Comparing {} to {}\n",current_iterate,idx.last_iterate);
    return current_iterate<idx.last_iterate;
  };
  virtual index_int operator*() { return ith_iterate(current_iterate); };
  virtual void operator++() { current_iterate++; };

  // Stuff
  virtual std::string as_string() const { throw(std::string("as_string: Not implemented")); };
  virtual void debug_on() {};
  virtual void debug_off() {};
};

class unknown_indexstruct : public indexstruct {
public:
  unknown_indexstruct() : indexstruct() {
    known_status = indexstruct_status::UNKNOWN;
  };
  virtual std::string as_string() const override { return std::string("unknown"); };
};

class empty_indexstruct : public indexstruct {
public:
  virtual bool is_empty( ) const override { return true; };
  virtual std::string type_as_string() const override { return std::string("empty"); };
  virtual index_int local_size() const override { return 0; };
  virtual std::shared_ptr<indexstruct> make_clone() const override {
    return std::shared_ptr<indexstruct>{ new empty_indexstruct() }; };
  virtual std::shared_ptr<indexstruct> add_element( const index_int idx ) override;
  virtual index_int first_index() const override {
    throw(std::string("No first index for empty")); };
  virtual index_int last_index()  const override {
    throw(std::string("No last index for empty")); };
  virtual index_int get_ith_element( const index_int i ) const override {
    throw(fmt::format("Can not get ith <<{}>> in empty",i)); };
  virtual std::shared_ptr<indexstruct> translate_by( index_int shift ) override {
    return this->make_clone(); };
  virtual bool has_intersect( std::shared_ptr<indexstruct> idx ) override { return false; };
  virtual std::shared_ptr<indexstruct> relativize_to( std::shared_ptr<indexstruct>,bool=false) override{
    return std::shared_ptr<indexstruct>( new empty_indexstruct() );
  };
  virtual std::shared_ptr<indexstruct> minus( std::shared_ptr<indexstruct> idx ) const override {
    return std::shared_ptr<indexstruct>( make_clone() ); }; //shared_from_this(); };
  virtual std::shared_ptr<indexstruct> truncate_left( index_int i ) override {
    return std::shared_ptr<indexstruct>( make_clone() ); }; //shared_from_this(); };
  virtual std::shared_ptr<indexstruct> truncate_right( index_int i ) override {
    return std::shared_ptr<indexstruct>( make_clone() ); }; //shared_from_this(); };

  //! Operating on empty struct give the struct itself.
  virtual std::shared_ptr<indexstruct> operate( const ioperator &op ) const override {
    return std::shared_ptr<indexstruct>( make_clone() ); };
  virtual std::shared_ptr<indexstruct> operate( const ioperator &&op ) const override {
    return std::shared_ptr<indexstruct>( make_clone() ); };
  virtual std::shared_ptr<indexstruct> operate( const ioperator&,index_int,index_int )
    const override {
    return std::shared_ptr<indexstruct>( make_clone() ); };

  virtual std::shared_ptr<indexstruct> operate( const sigma_operator &op ) const override {
    return std::shared_ptr<indexstruct>( make_clone() ); };
  //! Operating on empty struct give the struct itself.
  virtual std::shared_ptr<indexstruct> operate( const sigma_operator &op, index_int lo,index_int hi )
    const override {
    return std::shared_ptr<indexstruct>( make_clone() ); };

  virtual std::shared_ptr<indexstruct> struct_union( std::shared_ptr<indexstruct> idx ) override;
  virtual int can_merge_with_type( std::shared_ptr<indexstruct> idx) override { return 1; };
  virtual bool equals( std::shared_ptr<indexstruct> idx ) const override {
    return idx->is_empty(); };
  virtual std::string as_string() const override { return std::string("empty"); };

  /*
   * Iterator functions
   */
public:
  virtual void init_cur() override { current_iterate = 0; last_iterate = local_size(); };
  virtual indexstruct& end() override { last_iterate = local_size(); return *this; }
  // virtual empty_indexstruct& begin() override { return *this; };
  // virtual empty_indexstruct& end() override { return *this; };
  // virtual bool operator!=( indexstruct rr ) override { return 0; };
  // virtual void operator++() override { return; };
};

class strided_indexstruct : public indexstruct {
protected:
  index_int first,last, stride_amount{1};
public:
  strided_indexstruct(const index_int f,const index_int l,const int s)
    : first(f),last(l),stride_amount(s) { //cur(f) {
    last -= (last-first)%stride_amount; // make sure last is actually included
  };
  ~strided_indexstruct() {};
  /*
   * Statistics
   */
  index_int first_index() const override { return first; };
  index_int last_index()  const override { return last; };
  index_int local_size()  const override { return (last-first+stride_amount-1)/stride_amount+1; };
  virtual int stride() const override { return stride_amount; };
  virtual bool is_strided() const override { return true; };
  virtual std::string type_as_string() const override { return std::string("strided"); };
  virtual index_int find( index_int idx ) const override;
  virtual bool contains_element( index_int idx ) const override;
  virtual index_int get_ith_element( const index_int i ) const override;
  virtual int can_incorporate( index_int v ) override {
    return v==first-stride_amount || v==last+stride_amount; };
  //  virtual bool contains( indexstruct *idx ) override;
  virtual bool contains( std::shared_ptr<indexstruct> idx ) const override;
  //  virtual bool disjoint( indexstruct *idx ) override;
  virtual bool disjoint( std::shared_ptr<indexstruct> idx ) override;
  virtual bool equals( std::shared_ptr<indexstruct> idx ) const override;

  /*
   * Operations that yield a new indexstruct
   */
  virtual std::shared_ptr<indexstruct> convert_to_indexed() const override;
  virtual std::shared_ptr<indexstruct> make_clone() const override {
    return std::shared_ptr<indexstruct>{ new strided_indexstruct(first,last,stride_amount) };
  };
  virtual std::shared_ptr<indexstruct> add_element( const index_int idx ) override;
  virtual std::shared_ptr<indexstruct> translate_by( index_int shift ) override;
  virtual bool has_intersect( std::shared_ptr<indexstruct> idx ) override;
  virtual std::shared_ptr<indexstruct> intersect( std::shared_ptr<indexstruct> idx ) override;
  virtual std::shared_ptr<indexstruct> minus( std::shared_ptr<indexstruct> idx ) const override;
  virtual std::shared_ptr<indexstruct> relativize_to( std::shared_ptr<indexstruct>,bool=false) override;
  virtual std::shared_ptr<indexstruct> struct_union( std::shared_ptr<indexstruct> idx ) override;
  // virtual int can_merge_with_type(indexstruct *idx) override {
  //   strided_indexstruct *strided = dynamic_cast<strided_indexstruct*>(idx);
  //   if (strided!=nullptr) { // strided & strided
  //     return stride_amount==strided->stride_amount &&
  // 	first%stride_amount==strided->first%strided->stride_amount &&
  // 	strided->first<=last+stride_amount && strided->last>=first-stride_amount;
  //   } else return 0;
  // };
  virtual int can_merge_with_type( std::shared_ptr<indexstruct> idx) override {
    strided_indexstruct *strided = dynamic_cast<strided_indexstruct*>(idx.get());
    if (strided!=nullptr) { // strided & strided
      return stride_amount==strided->stride_amount &&
	first%stride_amount==strided->first%strided->stride_amount &&
	strided->first<=last+stride_amount && strided->last>=first-stride_amount;
    } else return 0;
  };
  virtual std::shared_ptr<indexstruct> split( std::shared_ptr<indexstruct> idx ) override;

  /*
   * Iterator functions
   */
  virtual indexstruct& end() override { last_iterate = local_size(); return *this; }
  virtual void init_cur() override {
    current_iterate = 0; last_iterate = local_size();
    //fmt::print("Setting last iterate to {}\n",last_iterate);
  };
// protected:
//   int cur{0};
// public:
//   void init_cur() { cur = first; };
//   //! \todo can we get this to be const? is that reference needed, and then return copy of this?
//   strided_indexstruct& begin() { init_cur(); return *this; }
//   strided_indexstruct& end() { return *this; }
//   bool operator!=( indexstruct rr ) override { return cur <= last /*rr.last_index()*/; }
//   void operator++() { cur += stride_amount; }
//   index_int operator*() const { return cur; }

  /*
   * Operate
   */
  virtual std::shared_ptr<indexstruct> operate( const ioperator &op ) const override;
  virtual std::shared_ptr<indexstruct> operate( const ioperator &&op ) const override;
  virtual std::shared_ptr<indexstruct> operate( const sigma_operator &op ) const override;

  virtual std::string as_string() const override {
    return fmt::format("strided: [{}-by{}--{}]",first,stride_amount,last); };
};

class contiguous_indexstruct : public strided_indexstruct {
public:
  contiguous_indexstruct(const index_int s,const index_int l) : strided_indexstruct(s,l,1) {};
  contiguous_indexstruct(const index_int f) : contiguous_indexstruct(f,f) {};
  ~contiguous_indexstruct() {};
  virtual bool is_contiguous() const override { return true; };
  virtual std::string type_as_string() const override { return std::string("contiguous"); };
  std::shared_ptr<indexstruct> make_clone() const override {
    return std::shared_ptr<indexstruct>{ new contiguous_indexstruct(first,last) };
  };
  virtual std::string as_string() const override {
    return fmt::format("contiguous: [{}--{}]",first,last); };
};

/*!
  An indexed indexstruct contains a standard vector of indices.
 */
class indexed_indexstruct : public indexstruct {
private:
  std::vector<index_int> indices;
public:
  indexed_indexstruct() {}; //!< Create an empty indexed struct
  indexed_indexstruct( const index_int len,const index_int *idxs );
  indexed_indexstruct( const std::vector<index_int> idxs ) {
    for ( auto i : idxs ) indices.push_back(i);
  };
  indexed_indexstruct( strided_indexstruct *cont ) {
    index_int s = cont->stride();
    for (index_int idx = cont->first_index(); idx<=cont->last_index(); idx+=s)
      indices.push_back(idx); };
  ~indexed_indexstruct() {};
  bool is_indexed() const override { return true; };
  virtual std::shared_ptr<indexstruct> simplify() override;
  virtual std::shared_ptr<indexstruct> force_simplify() const override;
  virtual std::string type_as_string() const override { return std::string("indexed"); };
  virtual void reserve( index_int s ) override { indices.reserve(s); };

  /*
   * Iterator functions
   */
// protected:
//   mutable int cur{0};
// public:
//   indexed_indexstruct& begin() { init_cur(); return *this; };
//   indexed_indexstruct& end() { return *this; };
//   virtual void init_cur() override { cur = 0; };
  virtual void init_cur() override { current_iterate = 0; last_iterate = local_size(); };
  virtual indexstruct& end() override { last_iterate = local_size(); return *this; }
  // indexed_indexstruct& begin_at_value(index_int v) {
  //   if (cur>=indices.size() || indices[cur]>v) cur = 0;
  //   return *this; }
  // bool operator!=( indexstruct rr ) override { return cur<rr.local_size(); };
  // void operator++() override { cur += 1; };
  // index_int operator*() override { return indices[cur]; }
  //  int search_loc() { return cur; };

  /*
   * Statistics
   */
  index_int first_index() const override {
    if (indices.size()==0) throw(std::string("Can not ask first for empty indexed"));
    return indices.at(0); };
  index_int last_index() const override {
    if (indices.size()==0) throw(std::string("Can not ask last for empty indexed"));
    return indices.at(indices.size()-1); };
  index_int local_size() const override { return indices.size(); };
  virtual index_int find( index_int idx ) const override {
    for (int i=0; i<indices.size(); i++)
      if (indices[i]==idx) return i;
    // for (auto loc=this->begin_at_value(idx); loc!=this->end(); ++loc)
    //   if (*loc==idx) return loc.search_loc();
    throw(std::string("Index to find is out of range"));
  };
  virtual bool contains_element( index_int idx ) const override {
    if (indices.size()==0) return false;
    for (int i=0; i<indices.size(); i++)
      if (indices[i]==idx) return true;
    return false;
  };
  virtual index_int get_ith_element( const index_int i ) const override;
  bool is_strided_between_indices(int,int,int&) const;
  //  virtual bool contains( indexstruct *idx ) override;
  virtual bool contains( std::shared_ptr<indexstruct> idx ) const override;
  virtual int can_incorporate( index_int v ) override { return 1; } //!< We can always add an index.
  //  virtual bool disjoint( indexstruct *idx ) override;
  virtual bool disjoint( std::shared_ptr<indexstruct> idx ) override;
  virtual bool equals( std::shared_ptr<indexstruct> idx ) const override;

  /*
   * Operations that yield a new indexstruct
   */
  virtual std::shared_ptr<indexstruct> make_clone() const override {
    return std::shared_ptr<indexstruct>{ new indexed_indexstruct(indices) }; };
  virtual std::shared_ptr<indexstruct> convert_to_indexed() const override {
    return std::shared_ptr<indexstruct>( this->make_clone() ); }; //shared_from_this(); };
  virtual std::shared_ptr<indexstruct> add_element( const index_int idx ) override;
  virtual void addin_element( const index_int idx ) override;
  virtual std::shared_ptr<indexstruct> translate_by( index_int shift ) override;

  virtual std::shared_ptr<indexstruct> minus( std::shared_ptr<indexstruct> idx ) const override;
  virtual std::shared_ptr<indexstruct> relativize_to( std::shared_ptr<indexstruct>,bool=false) override;
  virtual std::shared_ptr<indexstruct> struct_union( std::shared_ptr<indexstruct> idx ) override;
  // virtual int can_merge_with_type(indexstruct *idx) override {
  //   return idx->is_indexed(); };
  virtual int can_merge_with_type( std::shared_ptr<indexstruct> idx) override {
    return idx->is_indexed(); };
  virtual std::shared_ptr<indexstruct> intersect( std::shared_ptr<indexstruct> idx ) override;

  /*
   * Operate
   */
  virtual std::shared_ptr<indexstruct> operate( const ioperator &op ) const override;

  virtual std::string as_string() const override { fmt::memory_buffer w;
    format_to(w.end(),"indexed: {}:[",indices.size());
    for (auto i : indices) format_to(w.end(),"{},",i); format_to(w.end(),"]"); return to_string(w);
  }
};

/*!
  A composite indexstruct is a vector of non-composite indexstructs.
  The components are disjoint, but not necessarily sorted.
  \todo the structs vector should not be a pointer
*/
class composite_indexstruct : public indexstruct {
private:
  std::vector<std::shared_ptr<indexstruct>> structs;
public:
  composite_indexstruct() {};
  virtual bool is_composite() const override { return true; };
  virtual std::string type_as_string() const override { return std::string("composite"); };
  void push_back( std::shared_ptr<indexstruct> idx );
  const std::vector<std::shared_ptr<indexstruct>> &get_structs() const { return structs; };
  virtual std::shared_ptr<indexstruct> make_clone() const override;

  /*
   * Statistics
   */
  virtual bool is_empty() const override {
    for (auto s : structs )
      if (!s->is_empty()) return false; return true; };
  virtual index_int first_index() const override;
  virtual index_int last_index()  const override;
  virtual index_int local_size()  const override;

  virtual bool contains_element( index_int idx ) const override;
  virtual bool contains( std::shared_ptr<indexstruct> idx ) const override;
  virtual index_int find( index_int idx ) const override ;
  virtual index_int get_ith_element( const index_int i ) const override;
  virtual std::shared_ptr<indexstruct> struct_union( std::shared_ptr<indexstruct> ) override;
  virtual std::shared_ptr<indexstruct> intersect( std::shared_ptr<indexstruct> idx ) override;
  virtual std::shared_ptr<indexstruct> convert_to_indexed() const override;
  virtual std::shared_ptr<indexstruct> force_simplify() const override;
  virtual std::shared_ptr<indexstruct> over_simplify() override;
  virtual std::shared_ptr<indexstruct> relativize_to( std::shared_ptr<indexstruct>,bool=false) override ;
  virtual std::shared_ptr<indexstruct> minus( std::shared_ptr<indexstruct> idx ) const override;
  //  virtual bool disjoint( indexstruct *idx ) override;
  virtual bool disjoint( std::shared_ptr<indexstruct> idx ) override;
  virtual bool equals( std::shared_ptr<indexstruct> idx ) const override;

  virtual std::shared_ptr<indexstruct> operate( const ioperator &op ) const override;

  virtual std::string as_string() const override;

  /*
   * Iterator functions
   */
  virtual void init_cur() override { current_iterate = 0; last_iterate = local_size(); };
  virtual indexstruct& end() override { last_iterate = local_size(); return *this; }
// protected:
//   int cur_struct = 0;
// public:
//   void init_cur();
//   composite_indexstruct& begin() override { init_cur(); return *this; };
//   composite_indexstruct& end() override { return *this; };
//   virtual bool operator!=( indexstruct rr ) override;
//   void operator++() override;
//   index_int operator*() override { return *( *structs.at(cur_struct).get() ); };
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
class ioperator {
protected:
  iop_type type{iop_type::INVALID};
  int mod{0}, baseop{0};
  index_int by{0};
  std::function< index_int(index_int) > func;
public:
  //! This constructor is  needed for the derived classes
  ioperator() {};
  ioperator( std::string op );
  ioperator( std::string op,index_int amt );
  //! In the most literal interpretation, we operate an actual function pointer.
  ioperator( index_int(*f)(index_int) ) { type = iop_type::FUNC; func = f; };
  ioperator( std::function< index_int(index_int) > f ) { type = iop_type::FUNC; func = f; };
  index_int operate( index_int ) const;
  index_int operate( index_int, index_int ) const;
  index_int inverse_operate( index_int ) const;
  index_int inverse_operate( index_int, index_int ) const;
  bool is_none_op()        const { return type==iop_type::NONE; };
  bool is_shift_op()       const {
    return type==iop_type::SHIFT_REL || type==iop_type::SHIFT_ABS; };
  bool is_shift_to()       const { return type==iop_type::SHIFT_ABS; };
  bool is_right_shift_op() const { return is_shift_op() && (by>0); };
  bool is_left_shift_op()  const { return is_shift_op() && (by<0); };
  bool is_modulo_op()      const { return is_shift_op() && mod; };
  bool is_bump_op()        const { return is_shift_op() && !mod; };
  bool is_restrict_op()    const { return type==iop_type::MULT; }; // VLE don't like the name. lose
  bool is_mult_op()        const { return type==iop_type::MULT; };
  bool is_base_op()        const { return baseop; };
  bool is_div_op()         const { return type==iop_type::DIV; };
  bool is_contdiv_op()     const { return type==iop_type::CONTDIV; };
  bool is_function_op()    const { return type==iop_type::FUNC; };
  index_int amount() const { return by; };
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
protected:
  int dim{0}; //!< By default we shift in dimension zero.
public:
  int get_dimension() const { return dim; };
};

class shift_operator : public ioperator {
public:
  shift_operator( index_int n,bool relative=true )
    : ioperator() {
    if (relative)
      type = iop_type::SHIFT_REL;
    else
      type = iop_type::SHIFT_ABS;
    by = n; };
  shift_operator( int d,index_int n ) : shift_operator(n) { dim = d; };
};

class mult_operator : public ioperator {
public:
  mult_operator( index_int n )
    : ioperator() {
    type = iop_type::MULT;
    by = n; };
};

class multi_indexstruct;

//! A domain coordinate is a domain point. \todo make templated coordinate class
class domain_coordinate {
private:
  std::vector<index_int> coordinates;
public:
  domain_coordinate() {}; // in case we need a default constructor
  // Create an empty domain_coordinate object of given dimension
  domain_coordinate(int dim);
  //! Create from std vector
  domain_coordinate( std::vector<index_int> ic ) { coordinates = ic; };
  domain_coordinate( std::vector<int> ic ) { int dim = ic.size();
    coordinates = std::vector<index_int>(dim);
    for (int i=0; i<dim; i++)
      coordinates.at(i) = ic.at(i);
  };
  // Copy constructor
  domain_coordinate( domain_coordinate *other );

  // basic stats and manipulation
  int get_dimensionality() const; int get_same_dimensionality(int) const;
  void reserve(int n) { coordinates.reserve(n); };
  index_int &at(int i) { return coordinates.at(i); };
  index_int sub(int i) const { return coordinates.at(i); };
  void set(int d,index_int v) { coordinates.at(d) = v; };
  index_int coord(int d) const;
  //! \todo make by reference
  std::vector<index_int> data() const { return coordinates; };
  index_int volume() const;

  index_int linear_location_in( domain_coordinate,domain_coordinate ) const;
  index_int linear_location_in( domain_coordinate*,domain_coordinate* ) const;
  index_int linear_location_in( domain_coordinate farcorner ) const;
  index_int linear_location_in( domain_coordinate *farcorner ) const;
  index_int linear_location_in( std::shared_ptr<multi_indexstruct> bigstruct ) const;
  index_int linear_location_in( const multi_indexstruct &bigstruct ) const;

  // operators
  // bool operator==( domain_coordinate &other ) const; VLE don't work with denotations
  // bool operator!=( domain_coordinate &other ) const;
  bool operator==( const domain_coordinate &&other ) const;
  bool operator==( const domain_coordinate &other ) const;
  bool operator!=( const domain_coordinate other ) const;
  domain_coordinate operator+(index_int i) const;
  domain_coordinate operator+(const domain_coordinate i) const;
  domain_coordinate operator-(index_int i) const;
  domain_coordinate operator-(const domain_coordinate i) const;
  domain_coordinate operator*(index_int i) const;
  domain_coordinate operator/(index_int i) const;
  domain_coordinate operator%(index_int i) const;
  bool operator<(domain_coordinate other) const;
  bool operator>(domain_coordinate other) const;
  bool operator<=(domain_coordinate other) const;
  bool operator>=(domain_coordinate other) const;
  domain_coordinate *negate();
  const domain_coordinate operate( const ioperator& ) const ;
  domain_coordinate *operate_p( const ioperator& ) const;
  //!\todo can we make this by reference?
  index_int operator[](int id) const {
    if (id<0 || id>=get_dimensionality())
      throw(fmt::format("Wrong dimension {} to get from coordinate <<{}>>",id,as_string()));
    auto cid = coordinates[id]; return cid;
  };
  //! \todo const ref !
  void min_with(const domain_coordinate&); void max_with(const domain_coordinate&);
  //void min_with(domain_coordinate*); void max_with(domain_coordinate*);
  bool equals( domain_coordinate *other );
  bool is_zero();
  std::string as_string() const { fmt::memory_buffer w;
    format_to(w.end(),"["); for ( auto i : coordinates ) format_to(w.end(),"{},",i);
    format_to(w.end(),"]"); return to_string(w); };

  bool is_on_left_face(int d,std::shared_ptr<multi_indexstruct>) const;
  bool is_on_right_face(int d,std::shared_ptr<multi_indexstruct>) const;

  // iterating
protected:
  index_int iterator{-1};
public:
  domain_coordinate& begin() { iterator = 0; return *this; };
  domain_coordinate& end() { return *this; };
  bool operator!=( domain_coordinate ps ) { return iterator<coordinates.size()-1; };
  void operator++() { iterator++; };
  //  index_int operator[](int d) { return coordinates[d]; };
  index_int operator*() const {
    if (iterator<0)
      throw(fmt::format("dereferincing iterator {} in {}",iterator,as_string()));
    index_int v = coordinates[iterator];
    //printf("deref domain coord @%d to %d\n",iterator,v);
    return v;
  };
};

//! \todo this needs to be a singleton class
class domain_coordinate_zero : public domain_coordinate {
public:
  domain_coordinate_zero(int dim) : domain_coordinate(dim) {
    for (int id=0; id<dim; id++) set(id,0.); };
};

class domain_coordinate_allones : public domain_coordinate {
public:
  domain_coordinate_allones(int dim) : domain_coordinate(dim) {
    for (int id=0; id<dim; id++) set(id,1.); };
};

//! Wrap a single index_int into a one-d \ref domain_coordinate.
class domain_coordinate1d : public domain_coordinate {
public:
  domain_coordinate1d( index_int i ) : domain_coordinate(1) {
    set(0,i); };
};

/*!
  A sigma operator takes a point and gives a structure;
*/
class sigma_operator {
protected:
  std::function< std::shared_ptr<indexstruct>(index_int) > func;
  bool lambda_i{false};
  std::function< std::shared_ptr<indexstruct>( const indexstruct& ) > sfunc;
  bool lambda_s{false};
  //  std::shared_ptr<ioperator> point_func{nullptr};
  ioperator point_func;
  bool lambda_p{false};
public:
  //! Default creator
  sigma_operator() {};
  //! Create from function pointer
  sigma_operator( std::shared_ptr<indexstruct>(*f)(index_int) ) {
    func = f; lambda_i = true; };
  //! Create from point lambda
  sigma_operator( std::function< std::shared_ptr<indexstruct>(index_int) > f ) {
    func = f; lambda_i = true; };
  //! Create from struct lambda
  sigma_operator( std::function< std::shared_ptr<indexstruct>(const indexstruct&) > f ) {
    sfunc = f; lambda_s = true; };
  //! Create from ioperator
  sigma_operator( std::shared_ptr<ioperator> f ) {
    fmt::print("please no sigma from pointer to iop\n");
    point_func = *(f.get()); lambda_p = true; };
  //! Create from ioperator
  sigma_operator( ioperator f ) {
    point_func = f; lambda_p = true; };
  //! Create from scalar lambda \todo this should be the pointfunc?
  sigma_operator( std::function< index_int(index_int) > f ) {
    printf("You sure this is not the pointfunc?\n");
    func = [f] ( index_int i ) -> std::shared_ptr<indexstruct> {
      return std::shared_ptr<indexstruct>{new contiguous_indexstruct( f(i) )};
    };
    lambda_i = true;
  };
  //! Does this produce a single point?
  bool is_point_operator() const { return lambda_p; };
  bool is_struct_operator() const { return lambda_s; };
  std::shared_ptr<indexstruct> struct_apply( const indexstruct &i) const { return sfunc(i); };
  const ioperator &point_operator() const {
    if (!is_point_operator()) throw(std::string("Is not point operator"));
    return point_func; };
  std::shared_ptr<indexstruct> operate( index_int i ) const;
  std::shared_ptr<indexstruct> operate( std::shared_ptr<indexstruct> idx ) const;
  std::string as_string() const;
};

class multi_indexstruct;
class multi_sigma_operator_implementation {
protected:
  int dim{-1};
public:
  multi_sigma_operator_implementation(int dim) : dim(dim) {};
  bool get_same_dimensionality(int d) const {
    if (dim!=d) throw(fmt::format("Not the same dimensionality: {}<>{}",dim,d));
    return dim;
  };
  virtual std::shared_ptr<multi_indexstruct> operate
      ( const domain_coordinate &point ) const = 0;
  virtual std::shared_ptr<multi_indexstruct> operate
      ( const multi_indexstruct &idx ) const = 0;
  virtual bool is_single_operator() const { return false; };
  virtual bool is_point_operator() const { return false; };
  virtual bool is_coord_struct_operator() const { return false; };
  virtual sigma_operator get_operator(int id) const {
    throw(fmt::format("Can not get operator in dim {}",id)); };
};

/*!
  Multi-dimensional sigma operator: give a \ref multi_indexstruct from a \ref domain_coordinate.
  Cases:
  - an array of \ref sigma_operator 
  - a single function pointer multi_indexstruct -> multi_indexstruct
  - a function processor_coordinate -> processor_coordinate, 
    which is to be applied all over the domain; in practice to first & last index
    \todo use coordinate& instead of pointer
*/
class multi_sigma_operator {
protected:
  int opdim{-1};
  // pointer to private implementation
  std::shared_ptr<multi_sigma_operator_implementation> operator_implementation{nullptr};
public:
  //! Default constructor
  multi_sigma_operator() {};
  //! Construct with dimension
  multi_sigma_operator( int d ) { opdim = d; };
  int get_dimensionality() const;
  int get_same_dimensionality(int dim) const;

  // constructor from list of sigma operators
  multi_sigma_operator( std::vector<ioperator> ops );
  multi_sigma_operator( std::vector<sigma_operator> ops );
  multi_sigma_operator( sigma_operator op )
    : multi_sigma_operator( std::vector<sigma_operator>{op} ) {};

  // constructor from coordinate operator
  multi_sigma_operator
  ( int dim, std::function< domain_coordinate(const domain_coordinate&) > f );

  // constructor from coordinate-to-struct operator
  multi_sigma_operator
  ( int dim,std::function< std::shared_ptr<multi_indexstruct>(const domain_coordinate&) > f );
  //( int dim,std::shared_ptr<multi_indexstruct>(*)(const domain_coordinate&) );

  // constructor from struct to struct
  multi_sigma_operator
  ( int dim,std::function< std::shared_ptr<multi_indexstruct>(const multi_indexstruct&) > f );
  //( int dim,std::shared_ptr<multi_indexstruct>(*)(const multi_indexstruct&) );
    
  // What type are we?
  bool is_single_operator() const {
    return operator_implementation->is_single_operator(); };
  bool is_point_operator() const {
    return operator_implementation->is_point_operator(); };
  bool is_coord_struct_operator() const {
    return operator_implementation->is_coord_struct_operator(); };
  // operating is implementation dependent
  std::shared_ptr<multi_indexstruct> operate( const domain_coordinate &point ) const {
    return operator_implementation->operate(point); };
  std::shared_ptr<multi_indexstruct> operate( const multi_indexstruct &idx ) const {
    return operator_implementation->operate(idx); };
  std::shared_ptr<multi_indexstruct> operate( std::shared_ptr<const multi_indexstruct> idx ) const {
    return operate( *(idx.get()) ); };
  std::shared_ptr<multi_indexstruct> operate( std::shared_ptr<multi_indexstruct> idx ) const {
    return operate( *(idx.get()) ); };
  // get one dimension
  sigma_operator get_operator(int id) const {
    return operator_implementation->get_operator(id); };

protected:
  iop_type type{iop_type::INVALID};
public:
  bool is_shift_op()       const {
    return type==iop_type::SHIFT_REL || type==iop_type::SHIFT_ABS; };
  //! For functionally defined operators, the user can set the type.
  void set_is_shift_op() { type = iop_type::SHIFT_ABS; };
};

// vector of sigma operators
class multi_sigma_operator_impl_vector : public multi_sigma_operator_implementation {
protected:
  std::vector<sigma_operator> operators;
public:
  multi_sigma_operator_impl_vector( std::vector<sigma_operator> ops );
  multi_sigma_operator_impl_vector( std::vector<ioperator> ops );
  // implementation of pure virtual functions
  virtual std::shared_ptr<multi_indexstruct> operate
      ( const domain_coordinate &point ) const override;
  virtual std::shared_ptr<multi_indexstruct> operate
      ( const multi_indexstruct &idx ) const override;
  virtual bool is_point_operator() const override { return true; };
  virtual sigma_operator get_operator(int id) const override { return operators.at(id); };
};

/*!
  Implementation of the multi_sigma_operator_implementation 
  through coordinate operator
 */
class multi_sigma_operator_impl_coord : public multi_sigma_operator_implementation {
protected:
  std::function< domain_coordinate(const domain_coordinate&) >    coord_oper{nullptr};
public:
  multi_sigma_operator_impl_coord
      ( int dim,std::function<domain_coordinate(const domain_coordinate&)> f )
    : multi_sigma_operator_implementation(dim),coord_oper(f) {
  };
  virtual std::shared_ptr<multi_indexstruct> operate
      ( const domain_coordinate &point ) const override;
  virtual std::shared_ptr<multi_indexstruct> operate
      ( const multi_indexstruct &idx ) const override;
  virtual bool is_point_operator() const override { return true; };
};

// coordinate-to-struct operator
class multi_sigma_operator_impl_sigma : public multi_sigma_operator_implementation {
protected:
  std::function< std::shared_ptr<multi_indexstruct>(const domain_coordinate&) >
                                                                  sigma_oper{nullptr};
public:
  multi_sigma_operator_impl_sigma
  ( int dim, std::function< std::shared_ptr<multi_indexstruct>(const domain_coordinate&) > f )
    : multi_sigma_operator_implementation(dim),sigma_oper(f) {
  };
  virtual std::shared_ptr<multi_indexstruct> operate
      ( const domain_coordinate &point ) const override;
  virtual std::shared_ptr<multi_indexstruct> operate
      ( const multi_indexstruct &idx ) const override;
  virtual bool is_coord_struct_operator() const override { return true; };
};

// struct to struct
class multi_sigma_operator_impl_struct : public multi_sigma_operator_implementation {
protected:
  std::function
    < std::shared_ptr<multi_indexstruct>(const multi_indexstruct&) >struct_oper{nullptr};
public:
  multi_sigma_operator_impl_struct
  ( int dim,
    std::function< std::shared_ptr<multi_indexstruct>(const multi_indexstruct&) > f )
    : multi_sigma_operator_implementation(dim) {
    struct_oper = f ; };
  virtual std::shared_ptr<multi_indexstruct> operate
      ( const domain_coordinate &point ) const override;
  virtual std::shared_ptr<multi_indexstruct> operate
      ( const multi_indexstruct &idx ) const override;
  virtual bool is_single_operator() const override { return true; };
};

class multi_ioperator;
/*!
  We have limited support for multi-dimensional indexstructs.
  \todo lose the dim parameter
*/
class multi_indexstruct : public std::enable_shared_from_this<multi_indexstruct> {
protected:
  int dim{-1};
  std::vector< std::shared_ptr<indexstruct> > components;
public:
  std::vector< std::shared_ptr<multi_indexstruct> > multi;
public:
  multi_indexstruct() {}; 
  multi_indexstruct(int);
  multi_indexstruct(std::shared_ptr<indexstruct>);
  multi_indexstruct(domain_coordinate);
  multi_indexstruct(domain_coordinate,domain_coordinate);
  multi_indexstruct( std::vector< std::shared_ptr<indexstruct> > );
  multi_indexstruct( std::vector<index_int> sizes );

  bool is_uninitialized() const { return dim<0; };
  bool is_multi() const { return multi.size()>0; };
  int multi_size() const { return multi.size(); };
  std::shared_ptr<multi_indexstruct> make_clone() const;
  std::shared_ptr<multi_indexstruct> relativize_to( std::shared_ptr<multi_indexstruct> );
  bool is_strided() const;
  int type_as_int() const;
  std::string as_string() const;
      
  /*
   * Access
   */
  int get_dimensionality() const { return dim; }; int get_same_dimensionality(int) const;
  bool is_known() const; bool is_empty() const; bool is_contiguous() const;
  std::string type_as_string() const;
protected:
  void set_needs_recomputing() { stored_volume = -1;
    stored_first_index = nullptr; stored_last_index = nullptr;
  };
  domain_coordinate stored_local_size;

protected:
  //! This has to be a pointer otherwise we get infinite inclusion
  mutable std::shared_ptr<multi_indexstruct> stored_enclosing_structure{nullptr};
public:
  //  void compute_enclosing_structure();
  const multi_indexstruct &enclosing_structure_r() const;
  const std::shared_ptr<multi_indexstruct> enclosing_structure() const;

protected:
  mutable index_int stored_volume{-1};
public:
  index_int volume() const;

protected:
  mutable std::shared_ptr<domain_coordinate> stored_first_index{nullptr};
  mutable std::shared_ptr<domain_coordinate> stored_last_index{nullptr};
public:
  //  void compute_first_index(); void compute_last_index();
  const domain_coordinate &first_index_r() const;
  const domain_coordinate &last_index_r() const;
  const domain_coordinate &local_size_r() const;
  domain_coordinate *stride() const;
  // 1d cases
  index_int first_index(int d) const { return components.at(d)->first_index(); };
  index_int last_index(int d)  const { return components.at(d)->last_index(); };
  index_int local_size(int d)  const { return components.at(d)->local_size(); };
  std::vector<domain_coordinate> get_corners() const;

  /*
   * Data manipulation
   */
  void set_component( int, std::shared_ptr<indexstruct> );
  std::shared_ptr<indexstruct> get_component(int) const;
  //  bool contains_element( const domain_coordinate *i );
  bool contains_element( const domain_coordinate &i ) const;
  bool contains_element( const domain_coordinate &&i ) const;
  bool contains( std::shared_ptr<multi_indexstruct> ) const;
  bool contains( const multi_indexstruct& ) const;
  bool equals( std::shared_ptr<multi_indexstruct> ) const;
  bool operator==( const multi_indexstruct& ) const;

  index_int linear_location_of( std::shared_ptr<multi_indexstruct> ) const;
  index_int linear_location_in( std::shared_ptr<multi_indexstruct> ) const;
  index_int linear_location_in( const multi_indexstruct& ) const;
  index_int linear_location_of( domain_coordinate * ) const;
  domain_coordinate *location_of( std::shared_ptr<multi_indexstruct> inner ) const;
  index_int linearfind( index_int i );
  domain_coordinate linear_offsets(std::shared_ptr<multi_indexstruct> ) const;

  /*
   * Operations
   */
  std::shared_ptr<multi_indexstruct> operate( const ioperator &op ) const;
  std::shared_ptr<multi_indexstruct> operate( const ioperator &&op ) const;
  // VLE lose!
  std::shared_ptr<multi_indexstruct> operate( multi_ioperator* ) const;
  std::shared_ptr<multi_indexstruct> operate
      ( multi_ioperator*,const multi_indexstruct& ) const;
  std::shared_ptr<multi_indexstruct> operate
      ( multi_ioperator*,std::shared_ptr<multi_indexstruct> ) const;
  // end lose
  std::shared_ptr<multi_indexstruct> operate( const multi_sigma_operator& ) const;
  std::shared_ptr<multi_indexstruct> operate
      ( const ioperator&,std::shared_ptr<multi_indexstruct> ) const;
  std::shared_ptr<multi_indexstruct> operate
      ( const ioperator&,const multi_indexstruct& ) const;
  void translate_by(int d,index_int amt);

  std::shared_ptr<multi_indexstruct> struct_union( std::shared_ptr<multi_indexstruct>);
  bool can_union_in_place(std::shared_ptr<multi_indexstruct> other,int &diff) const;
  std::shared_ptr<multi_indexstruct> struct_union_in_place
      (std::shared_ptr<multi_indexstruct> other,int diff);
  std::shared_ptr<multi_indexstruct> minus( std::shared_ptr<multi_indexstruct>,bool=false ) const;
  std::shared_ptr<multi_indexstruct> split_along_dim(int,std::shared_ptr<indexstruct>) const;
  std::shared_ptr<multi_indexstruct> intersect( std::shared_ptr<multi_indexstruct> );
  std::shared_ptr<multi_indexstruct> intersect( const multi_indexstruct& );
  std::shared_ptr<multi_indexstruct> force_simplify(bool=false) const;

  /*
   * Ranging
   */
protected:
  index_int cur_linear{0};
  domain_coordinate cur_coord;
public:
  multi_indexstruct &begin();
  multi_indexstruct &end();
  void operator++();
  bool operator!=( multi_indexstruct& );
  bool operator==( multi_indexstruct& );
  domain_coordinate &operator*();
};

//! A multi_indexstruct with all components unknown
class unknown_multi_indexstruct : public multi_indexstruct {
public:
  unknown_multi_indexstruct( int dim ) : multi_indexstruct(dim) {
    for (int id=0; id<dim; id++)
      set_component( id, std::shared_ptr<indexstruct>( new unknown_indexstruct() ) );
  };
};

/*!
  A empty multi indexstruct is mostly an initializer shortcut. 
  For now there are no specific methods for this class.
  \todo write a unit test for being empty
*/
class empty_multi_indexstruct : public multi_indexstruct {
public:
  empty_multi_indexstruct( int dim ) : multi_indexstruct(dim) {
    for (int id=0; id<dim; id++)
      set_component( id, std::shared_ptr<indexstruct>( new empty_indexstruct() ) );
  };
};

/*!
  A contiguous multi indexstruct is mostly an initializer shortcut. 
  For now there are no specific methods for this class.
  \todo write a unit test for being contiguous
*/
class contiguous_multi_indexstruct : public multi_indexstruct {
public:
  contiguous_multi_indexstruct( const domain_coordinate first ) :
    contiguous_multi_indexstruct(first,first) {};
  contiguous_multi_indexstruct
      ( const domain_coordinate first,const domain_coordinate last ) :
    multi_indexstruct(first.get_dimensionality()) {
    int dim = first.get_same_dimensionality(last.get_dimensionality());
    for (int id=0; id<dim; id++)
      set_component
	( id, std::shared_ptr<indexstruct>
	  ( new contiguous_indexstruct(first.coord(id),last.coord(id)) ) );
  };
};

/*!
  A strided multi indexstruct is mostly an initializer shortcut. 
  For now there are no specific methods for this class.
  \todo write a unit test for being strided
*/
class strided_multi_indexstruct : public multi_indexstruct {
public:
  strided_multi_indexstruct( domain_coordinate first,domain_coordinate last,int stride ) :
    multi_indexstruct(first.get_dimensionality()) {
    int dim = first.get_same_dimensionality( last.get_dimensionality() );
    for (int id=0; id<dim; id++)
      set_component
	( id, std::shared_ptr<indexstruct>
	  ( new strided_indexstruct(first.coord(id),last.coord(id),stride) ) );
  };
};

/*! A vector of \ref ioperator objects.
 */
class multi_ioperator {
protected:
  std::vector<ioperator /*std::shared_ptr<ioperator>*/ > operators;
  bool operator_based{false};
  std::function< domain_coordinate*(domain_coordinate*) > pointf; bool point_based{false};
public:
  //! Create the operator vector and set with no-ops
  multi_ioperator( int d ) {
    if (d<=0)
      throw(fmt::format("multi ioperator dimension {} s/b >=1",d));
    for (int dm=0; dm<d; dm++)
      operators.push_back( ioperator("none") );
    //operators.push_back( std::shared_ptr<ioperator>(new ioperator("none")) );
    operator_based = true;
  };
  //! Create from same ioperator in all dimensions
  multi_ioperator( int dim,ioperator &op ) : multi_ioperator(dim) {
    for (int idim=0; idim<dim; idim++)
      set_operator(idim,op);
    operator_based = true;
  };
  multi_ioperator( int dim,ioperator &&op ) : multi_ioperator(dim) {
    for (int idim=0; idim<dim; idim++)
      set_operator(idim,op);
    operator_based = true;
  };
  //! Create from a single ioperator: one-d case shortcut.
  multi_ioperator( ioperator &op ) : multi_ioperator(1,op) {};
  multi_ioperator( ioperator &&op ) : multi_ioperator(1,op) {};
  // Create from explicit function
  multi_ioperator( std::function< domain_coordinate*(domain_coordinate*) > f ) {
    pointf = f; point_based = true; };
  
protected:
  iop_type type{iop_type::INVALID};
public:
  bool is_shift_op()       const {
    return type==iop_type::SHIFT_REL || type==iop_type::SHIFT_ABS; };
  //! For functionally defined operators, the user can set the type.
  void set_is_shift_op() { type = iop_type::SHIFT_ABS; };

  int get_dimensionality() const { return operators.size(); };
  int get_same_dimensionality( int d ) const {
    int dim = get_dimensionality();
    if (d!=dim) throw(fmt::format("multi_iop differing dims: {} & {}",dim,d));
    return dim; };
  const ioperator &get_operator(int id) const { return operators.at(id); };
  //void set_operator(int id,std::shared_ptr<ioperator> op);
  void set_operator(int id,ioperator &op);
  void set_operator(int id,ioperator &&op);
  bool is_modulo_op(); bool is_shift_op(); bool is_restrict_op();

  // operate on coordinate & multi_indexstruct
  domain_coordinate *operate( domain_coordinate *c );
  std::shared_ptr<multi_indexstruct> operate( std::shared_ptr<multi_indexstruct> idx ) {
    return idx->operate(this); };

  std::string as_string() {
    fmt::memory_buffer w;
    format_to(w.end(),"["); int id=0; const char *sep="";
    for ( auto iop : operators ) {
      format_to(w.end(),"{}op{}:{}",sep,id++,iop.as_string());
      sep = ", ";
    }
    format_to(w.end(),"]");
    return to_string(w);
  };
};

class multi_shift_operator : public multi_ioperator {
public:
  multi_shift_operator(int d) : multi_ioperator(d) {};
  multi_shift_operator( std::vector<index_int> shifts )
    : multi_ioperator(shifts.size()) {
    for (int id=0; id<shifts.size(); id++) {
      set_operator(id,shift_operator(shifts[id]));
    }
  };
  multi_shift_operator( domain_coordinate d )
    : multi_shift_operator( d.data() ) {}
};

/*! A stencil is an array of \ref multi_shift_operator objects.
  This is mostly for notational convenience.
  \todo make this an array of std::shared_ptr's; has to wait for \ref signature_function::add_sigma_oper to accept shared pointers.
*/
class stencil_operator {
  std::vector<multi_shift_operator*> ops;
  int dim{0};
public:
  stencil_operator(int d) { dim = d; };
  std::vector<multi_shift_operator*> &get_operators() { return ops; };
  void add(int i,int j) {
    if (dim!=2)
      throw(fmt::format("Stencil operator dimensionality defined as {}",dim));
    ops.push_back( new multi_shift_operator(std::vector<index_int>{i,j}) );
  };
  void add(int i,int j,int k) {
    if (dim!=3)
      throw(fmt::format("Stencil operator dimensionality defined as {}",dim));
    ops.push_back( new multi_shift_operator(std::vector<index_int>{i,j,k}) );
  };
  void add( const std::vector<index_int> &offset ) {
    if(dim!=offset.size())
      throw(fmt::format("Offset dimensionality {} incompatible with {}",offset.size(),dim));
    ops.push_back( new multi_shift_operator(offset) );
  };
  void add( domain_coordinate offset ) {
    if(dim!=offset.get_dimensionality())
      throw(fmt::format("DC Offset dimensionality {} incompatible with {}",
			offset.get_dimensionality(),dim));
    ops.push_back( new multi_shift_operator(offset) );
  };
  int get_dimensionality() const { return dim; };
};

/*
 * And here is the PIMPL class!
 */

class indexstructure {
private:
  std::shared_ptr<indexstruct> strct{nullptr};
  indexstruct_status known_status{indexstruct_status::KNOWN};
public:
  indexstructure() {
    strct = std::shared_ptr<indexstruct>( new unknown_indexstruct() );};
  //! \todo get rid of this, and onl accept references
  indexstructure( std::shared_ptr<indexstruct> idx ) {
    strct = idx->make_clone();
    //fmt::print("from shared ptr: idx={}\n",strct->as_string());
  };
  indexstructure( contiguous_indexstruct &c )
    : indexstructure( std::shared_ptr<indexstruct>( new contiguous_indexstruct(c) ) ) {};
  indexstructure( contiguous_indexstruct &&c )
    : indexstructure( std::shared_ptr<indexstruct>( new contiguous_indexstruct(c) ) ) {};
  indexstructure( strided_indexstruct &&c )
    : indexstructure( std::shared_ptr<indexstruct>( new strided_indexstruct(c) ) )    {};
  indexstructure( indexed_indexstruct &&c )
    : indexstructure( std::shared_ptr<indexstruct>( new indexed_indexstruct(c) ) )    {};
  indexstructure( composite_indexstruct &&c )
    : indexstructure( std::shared_ptr<indexstruct>( new composite_indexstruct(c) ) )  {
    fmt::print("composite type=<{}> strct composite: <{}> this composite: <{}>\n",
	       strct->type_as_string(),strct->is_composite(),is_composite());
  };
  
  bool is_known()             const {
    return strct->is_known(); };
  virtual bool is_empty( )     const {
    return strct->is_empty(); };
  virtual bool is_contiguous() const {
    //fmt::print("test contiguous {}\n",strct->as_string());
    return strct->is_contiguous(); };
  virtual bool is_strided()    const {
    return strct->is_strided(); };
  virtual bool is_indexed()    const {
    return strct->is_indexed(); };
  virtual bool is_composite()  const {
    return strct->is_composite(); };
  virtual std::string type_as_string()       const {
    return strct->type_as_string(); };
  virtual void reserve( index_int s )              {
    return strct->reserve(s); };
  void report_unimplemented( const char *c ) const {
    return strct->report_unimplemented(c); };

  /*
   * Multi
   */
  void push_back( contiguous_indexstruct &&idx );

  /*
   * Statistics
   */
  virtual index_int first_index() const {
    return strct->first_index(); };
  virtual index_int last_index()  const{
    return strct->last_index(); };
  virtual index_int local_size()  const {
    return strct->local_size(); };
  index_int volume() const {
    return strct->volume(); };
  index_int outer_size() const {
    return strct->outer_size(); };
  virtual int stride() const {
    return strct->stride(); };
  virtual bool equals( std::shared_ptr<indexstruct> idx ) const {
    return strct->equals(idx); };
  virtual bool operator==( std::shared_ptr<indexstruct> idx ) const {
    return strct->equals(idx); };
  virtual bool equals( indexstructure &idx ) const {
    return strct->equals(idx.strct); };
  
  virtual index_int find( index_int idx ) {
    return strct->find(idx); };
  index_int location_of( std::shared_ptr<indexstruct> inner ) {
    return strct->location_of(inner); };
  index_int location_of( indexstructure &inner ) {
    return strct->location_of(inner.strct); };
  index_int location_of( indexstructure &&inner ) {
    return strct->location_of(inner.strct); };
  virtual bool contains_element( index_int idx ) const {
    return strct->contains_element(idx); };
  bool contains_element_in_range(index_int idx) const;
  virtual bool contains( std::shared_ptr<indexstruct> idx ) {
    return strct->contains(idx); };
  virtual bool contains( indexstructure &idx ) const {
    return strct->contains(idx.strct); };
  virtual bool disjoint( std::shared_ptr<indexstruct> idx ) {
    return strct->disjoint(idx); };
  virtual bool disjoint( indexstructure &idx ) {
    return strct->disjoint(idx.strct); };
  virtual bool disjoint( indexstructure &&idx ) {
    return strct->disjoint(idx.strct); };
  virtual int can_incorporate( index_int v ) {
    return strct->can_incorporate(v); };
  virtual index_int get_ith_element( const index_int i ) const {
    return strct->get_ith_element(i); };

  /*
   * Operations that yield a new indexstruct
   */
  void simplify() { strct = strct->simplify(); };
  void force_simplify() { strct = strct->force_simplify(); };
  void over_simplify() { strct = strct->over_simplify(); };
  void add_element( const index_int idx ) { strct = strct->add_element(idx); };
  void addin_element( const index_int idx ) { strct->addin_element(idx); };
  void translate_by( index_int shift ) { strct = strct->translate_by(shift); };
  bool has_intersect( std::shared_ptr<indexstruct> idx ) {
    return strct->has_intersect(idx); };

  // union and intersection
  auto struct_union( indexstructure &idx ) {
    return indexstructure(strct->struct_union(idx.strct)); };
  auto struct_union( indexstructure &&idx ) {
    return indexstructure(strct->struct_union(idx.strct)); };
  void split( std::shared_ptr<indexstruct> idx ) { strct = strct->split(idx); };
  indexstructure split( indexstructure &idx ) {
    return indexstructure(strct->split(idx.strct)); };
  indexstructure intersect( std::shared_ptr<indexstruct> idx ) {
    return indexstructure( strct->intersect(idx) ); };
  indexstructure intersect( indexstructure &idx ) {
    return indexstructure( strct->intersect(idx.strct) ); };

  indexstructure minus( std::shared_ptr<indexstruct> idx ) {
    return indexstructure(strct->minus(idx)); };
  indexstructure minus( indexstructure &idx ) {
    return indexstructure(strct->minus(idx.strct)); };
  void truncate_left( index_int i ) { strct = strct->truncate_left(i); };
  void truncate_right( index_int i ) { strct = strct->truncate_right(i); };
  void relativize_to( std::shared_ptr<indexstruct> idx ) { strct = strct->relativize_to(idx); };
  void relativize_to( indexstructure &idx ) { strct = strct->relativize_to(idx.strct); };
  void convert_to_indexed() { strct = strct->convert_to_indexed(); };

  // operate
  auto operate( const ioperator &op ) { return indexstructure(strct->operate(op)); };
  auto operate( const ioperator &&op ) { return indexstructure(strct->operate(op)); };
  auto operate( const ioperator &op,index_int i0,index_int i1) {
    return indexstructure(strct->operate(op,i0,i1)); };

  auto operate( const sigma_operator &op ) { return indexstructure(strct->operate(op)); };
  auto operate( const sigma_operator &op, index_int lo,index_int hi ){
    return indexstructure(strct->operate(op,lo,hi)); };
  auto operate( const ioperator &op,std::shared_ptr<indexstruct> outer ) {
    return indexstructure(strct->operate(op,outer)); };
  auto operate( const sigma_operator &op,std::shared_ptr<indexstruct> outer ) {
    return indexstructure(strct->operate(op,outer)); };

#if 0
  // Iterable functions
  virtual indexstruct& begin() { init_cur(); return *this; }
  // the next 4 need to be pure virtual
  virtual void init_cur() {};
  virtual indexstruct& end() override { last_iterate = local_size(); return *this; }
  virtual bool operator!=( indexstruct idx ) { //printf("default test\n");
    return 0; };
  virtual index_int operator*() { return -1; };
  virtual void operator++() { return; };
#endif
  
  // Stuff
  std::string as_string() const;
  virtual void debug_on() {};
  virtual void debug_off() {};
};

class multi_indexstructure {
private:
  std::shared_ptr<multi_indexstruct> strct{nullptr};
public:
  multi_indexstructure() {};
  multi_indexstructure(int dim) {
    strct = std::shared_ptr<multi_indexstruct>( new unknown_multi_indexstruct(dim) );
  };
};
  // VLE why do we have an explicit copy constructor?
  // indexstructure( const indexstructure &idx )
  //   : indexstructure( idx.strct->make_clone() ) {};
  // indexstructure( const indexstructure &&idx )
  //   : indexstructure( idx.strct->make_clone() ) {};
  // //! if we omit this one, all assignments complain about `use of deleted operator='
  // indexstructure operator=(const indexstructure &idx ) const {
  //   fmt::print("indexstructure assignment operator is broken\n");
  //   auto c = dynamic_cast<contiguous_indexstruct*>( idx.strct.get() );
  //   if (c!=nullptr) {
  //     fmt::print("indexstructure operator= with contiguous\n");
  //     return indexstructure( contiguous_indexstruct(*c) );
  //   } else
  //     throw(std::string("Operator= not for this type"));
  // };

#endif
