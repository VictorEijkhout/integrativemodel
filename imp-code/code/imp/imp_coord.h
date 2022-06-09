#pragma once

#include <array>
#include "imp_env.h"

template<class I,int d>
class ioperator;

template<typename I,int d>
std::array<I,d> endpoint(I s);

//! Processor coordinates on a pretend grid. We order them by rows.
//! \todo write method to range over this
template<class I,int d>
class coordinate {
protected :
  std::array<I,d> coordinates;
public :
  coordinate();
  coordinate(I span);
  coordinate( std::array<I,d> );
  coordinate( environment& );

  // basic manipulation
  constexpr int dimensionality() const { return d; }
  I &at(int i);
  const I &at(int i) const;
  I span() const;
  bool before( const coordinate<I,d>& ) const;
  I linear( const coordinate<I,d>& ) const;
  //  bool operator>( const coordinate<I,d>& ) const;

  // // linearization
  // int linearize( const coordinate&) const; // linear number wrt cube layout
  // int linearize( const decomposition& ) const; // linear number wrt cube

  // equality operation
  coordinate negate();
  bool is_zero();
  // coordinate operate( const ioperator<I,d> &op );
  // coordinate operate( const ioperator<I,d> &&op );
  coordinate<I,d> operate( const ioperator<I,d> &op );
  coordinate<I,d> operate( const ioperator<I,d> &&op );
  std::string as_string() const;

#if 0
  bool is_on_left_face(const decomposition&) const;
  bool is_on_right_face(const decomposition&) const;
  bool is_on_face( const decomposition& ) const;
  bool is_on_face( const std::shared_ptr<object> ) const;
  bool is_on_face( const object& ) const;
  bool is_null() const;
  coordinate left_face_proc(int d,coordinate &&farcorner) const ;
  coordinate right_face_proc(int d,coordinate &&farcorner) const ;
  coordinate left_face_proc(int d,const coordinate &farcorner) const ;
  coordinate right_face_proc(int d,const coordinate &farcorner) const ;
#endif
  
  // operators
  //  coordinate<I,d> operator*(index_int i);

  // iterating
protected:
  int iterator{-1};
public:
  coordinate<I,d>& begin() { iterator = 0; return *this; };
  coordinate<I,d>& end() { return *this; };
  bool operator!=( coordinate<I,d> ps ) { return iterator<coordinates.size()-1; };
  void operator++() { iterator++; };
  int operator*() const {
    if (iterator<0)
      throw(fmt::format("deref negative iterator {} in {}",iterator,as_string()));
    int v = coordinates[iterator];
    //printf("deref coord @%d to %d\n",iterator,v);
    return v; };
};

