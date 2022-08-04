#pragma once

#include <array>
#include <vector>
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
  auto &data() { return coordinates; };
  const auto& data() const { return coordinates; };
  
  // basic manipulation
  constexpr int dimensionality() const { return d; }
  I &at(int i); const I &at(int i) const;
  I &operator[](int i); const I &operator[](int i) const;
  I span() const;
  bool before( const coordinate<I,d>& ) const;
  //  I linear( const coordinate<I,d>& ) const;

  // linearization
  I linear_location_of( const coordinate<I,d>& ) const;
  I linear_location_in( const coordinate<I,d>& ) const;

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
  coordinate<I,d> operator%( I ) const;
  coordinate<I,d> operator*( I ) const;
  coordinate<I,d> operator/( I ) const;
// #include <compare>
//   auto operator<=>( coordinate<I,d> ) const {
//     return std::accumulate( coordinates.begin(),coordinates.end(),
// 		       []( I x,I y) { return x<=>y; } );
//   };
  bool operator==( const coordinate<I,d>& other ) const;
  bool operator!=( const coordinate<I,d>& other ) const;
  bool operator<=( coordinate<I,d> other ) const;
  bool operator<( coordinate<I,d> ) const;
  bool operator>( coordinate<I,d> ) const;
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

template<class I,int d>
coordinate<I,d> coordmax( coordinate<I,d> current,coordinate<I,d> other );
template<class I,int d>
coordinate<I,d> coordmin( coordinate<I,d> current,coordinate<I,d> other );
template<typename I,int d>
void require_sorted( std::vector<coordinate<I,d>> idxs );

template<typename I,int d>
struct fmt::formatter<coordinate<I,d>> {
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
    (const coordinate<I,d>& p, FormatContext& ctx)
        -> decltype(ctx.out()) {
    return format_to(ctx.out(),"{}", p.as_string());
  }
};

template<typename I, int d>
std::ostream &operator<<(std::ostream &os,const std::shared_ptr<coordinate<I,d>> &c) {
  os << c->as_string();
  return os;
};

/*
 * Coordinate sets
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

