// -*- c++ -*-
/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2023
 ****
 **** imp_coord.h: Header file for coordinate stuff
 ****
 ****************************************************************/

#pragma once

#include <array>
#include <vector>
#include "imp_env.h"
#include "fmt/format.h"

// forward definition
template<typename I,int d>
class ioperator;

/*
 * Some free functions
 */
template<typename I,int d>
std::array<I,d> endpoint(I s);
template<typename I>
std::vector<I> split_points(I,I,int);
template<typename I>
std::vector<I> split_points(I,int);

//! Processor coordinates on a pretend grid. We order them by rows.
//! \todo write method to range over this
template<typename I,int d>
class coordinate {
public :
  coordinate();
  coordinate( std::array<I,d> );
  //! Simplified case for 1D
  coordinate( I i ) requires (d==1) : coordinate( std::array<I,1>{i} ) {};
  //! Copy constructor from other int type
  template<typename J>
  coordinate( coordinate<J,d> other ) {
    for (int id=0; id<d; id++)
      coordinates.at(id) = static_cast<I>( other.data().at(id) );
  };
  
  // data
protected :
  std::array<I,d> coordinates;
public:
  auto &data() { return coordinates; };
  const auto& data() const { return coordinates; };
  coordinate<I,d>& set( I val );
  
  // basic manipulation
  constexpr int dimensionality() const { return d; }
  constexpr bool same_dimensionality(int dd) const { return d==dd; }
  I &at(int i); const I &at(int i) const;
  I &operator[](int i); const I &operator[](int i) const;
  I span() const;
  bool before( const coordinate<I,d>& ) const;

  // linearization
  I linear_location_of( const coordinate<I,d>& ) const;
  I linear_location_in( const coordinate<I,d>& ) const;
  coordinate<I,d> location_of_linear( I s ) const;

  // equality operation
  coordinate negate();
  bool is_zero();
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
  coordinate<I,d> operator+( const coordinate<I,d>& ) const;
  coordinate<I,d> operator+( I ) const;

  coordinate<I,d> operator-( const coordinate<I,d>& ) const;
  coordinate<I,d> operator-( I ) const;
  coordinate<I,d> operator-( ) const;
  void operator-=( const coordinate<I,d>& );

  coordinate<I,d> operator%( const coordinate<I,d>& ) const;
  coordinate<I,d> operator%( I ) const;

  coordinate<I,d> operator*( const coordinate<I,d>& ) const;
  coordinate<I,d> operator*( I ) const;

  coordinate<I,d> operator/( const coordinate<I,d>& ) const;
  coordinate<I,d> operator/( I ) const;

// #include <compare>
//   auto operator<=>( coordinate<I,d> ) const {
//     return std::accumulate( coordinates.begin(),coordinates.end(),
// 		       []( I x,I y) { return x<=>y; } );
//   };
  bool operator==( const coordinate<I,d>& other ) const;
  bool operator!=( const coordinate<I,d>& other ) const;

  bool operator<( const coordinate<I,d> ) const;
  bool operator<=( const coordinate<I,d> other ) const;
  bool operator>( const coordinate<I,d> ) const;
  bool operator>=( const coordinate<I,d> ) const;

  bool operator>>( const coordinate<I,d> ) const;
  bool operator==( I ) const;

  // iterating
protected:
  int iterator{-1};
public:
  coordinate<I,d>& begin() { iterator = 0; return *this; };
  coordinate<I,d>& end() { return *this; };
  //  bool operator!=( coordinate<I,d> ps ) { return iterator<coordinates.size()-1; };
  void operator++() { iterator++; };
  int operator*() const {
    if (iterator<0)
      throw(fmt::format("deref negative iterator {} in {}",iterator,as_string()));
    int v = coordinates[iterator];
    //printf("deref coord @%d to %d\n",iterator,v);
    return v; };
};

template<typename I,int d>
coordinate<I,d> constant_coordinate( I v ) {
  coordinate<I,d> r;
  for (int id=0; id<d; id++)
    r[id] = v;
  return r;
};

template<typename I,int d>
coordinate<I,d> coordmax( const coordinate<I,d>& current,const coordinate<I,d>& other );
template<typename I,int d>
coordinate<I,d> coordmin( const coordinate<I,d>& current,const coordinate<I,d>& other );
template<typename I,int d>
coordinate<I,d> coordmod( const coordinate<I,d>& current,const coordinate<I,d>& other );
template<typename I,int d>
void require_sorted( std::vector<coordinate<I,d>> idxs );

template<typename I,int d>
struct fmt::formatter<coordinate<I,d>>;
template<typename I, int d>
std::ostream &operator<<(std::ostream &os,const coordinate<I,d> &c);
template<typename I, int d>
std::ostream &operator<<(std::ostream &os,const std::shared_ptr<coordinate<I,d>> &c);

/*
 * Coordinate sets
 */

/*! A coordinate_set is a vector of coordinates
  with some obvious methods. This is used in \see decomposition
*/
template<typename I,int d>
class coordinate_set {
private:
  std::vector< coordinate<I,d> > set;
public:
  bool contains( const coordinate<I,d> &p ) const;
  void add( const coordinate<I,d>& p );
  int size() { return set.size(); };
private:
  int cur{0};
public:
  coordinate_set &begin() { cur = 0; return *this; };
  coordinate_set &end() { return *this; };
  bool operator!=( coordinate_set & s ) { return cur<set.size(); };
  void operator++() { cur += 1; };
  coordinate<I,d> &operator*() { return set.at(cur); };
};

