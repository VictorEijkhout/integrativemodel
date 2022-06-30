#include "imp_coord.h"

#include <array>
using std::array;
#include <string>
using std::string;
#include <vector>
using std::vector;
#include <sstream>
using std::stringstream;

#include <cassert>
#include <fmt/format.h>

template<typename I,int d>
array<I,d> endpoint(I s) {
  array<I,d> endpoint; constexpr I last=d-1;
  for (int id=0; id<d; id++)
    endpoint.at(id) = 1;
  endpoint[0] = s;
  if (d==1) return endpoint;
  if (d==2) {
    for ( I div = sqrt(s),rem = s/div; ; div++,rem=s/div) {
      //print("try {} x {} = {}\n",div,rem,s);
      endpoint[0] = div; endpoint[1] = rem;
      if (div*rem==s) break;
    }
    return endpoint;
  }
  for (I prime=2; ; ) {
    auto s = endpoint.front();
    if (s==prime or s==endpoint.back()) break;
    if (s%prime==0) {
      //cout << "prime divisor: " << prime << "\n";
      s /= prime; endpoint.front() = s;
      // move everything greater equal this prime one back
      auto loc = endpoint.end(); loc--;
      auto nothead = endpoint.begin(); nothead++;
      while (loc!=nothead) {
	auto compare = loc; compare--;
	if (*loc==1 and *compare>1)
	  *loc = *compare;
	loc--;
      }
      *loc = prime;
    } else {
      // next prime
      I saved_prime = prime;
      for (I next_prime=prime+1; ; next_prime++) {
	bool isprime=true;
	for (I f=2; f<next_prime/2; f++) {
	  if (next_prime%f==0) { isprime = false; break; }
	}
	if (isprime) { prime = next_prime; break; }
      } // end of next prime loop
      // cout << "next prime: " << prime << "\n";
      assert(prime>saved_prime);
      assert(prime<=s);
    } // end else
  }
  return endpoint;
};

/*
 * Coordinates
 * constructors
 */
//snippet pcoorddim
template<typename I,int d>
coordinate<I,d>::coordinate() {
  for (int id=0; id<d; id++)
    coordinates.at(id) = -1;
};
template<typename I,int d>
coordinate<I,d>::coordinate( I s )
  : coordinate( endpoint<I,d>(s) ) {
};
template<typename I,int d>
coordinate<I,d>::coordinate( std::array<I,d> c)
  : coordinates( c ) {
};
template<typename I,int d>
coordinate<I,d>::coordinate( environment& e )
  : coordinate( e.nprocs() ) {
};
//snippet end

/*
 * Access
 */
template<typename I,int d>
I coordinate<I,d>::span() const {
  I res = 1;
  for ( auto i : coordinates )
    res *= i;
  return res;
  // return accumulate
  //   ( coordinates.begin(),coordinates.end()
  //     ,multiplies<I>(),static_cast<I>(1) );
};

template<typename I,int d>
I& coordinate<I,d>::at(int i) {
  return coordinates.at(i);
};
template<typename I,int d>
const I& coordinate<I,d>::at(int i) const {
  return coordinates.at(i);
};

template<typename I,int d>
I& coordinate<I,d>::operator[](int i) {
  return coordinates[i];
};
template<typename I,int d>
const I& coordinate<I,d>::operator[](int i) const {
  return coordinates[i];
};

/*
 * Operators
 */
// plus
template<typename I,int d>
coordinate<I,d> coordinate<I,d>::operator+( coordinate<I,d> other ) const {
  auto r(*this);
  for ( int id=0; id<d; id++ )
    r.coordinates[id] += other.coordinates[id];
  return r;
};
template<typename I,int d>
coordinate<I,d> coordinate<I,d>::operator+( I other ) const {
  auto r(*this);
  for ( int id=0; id<d; id++ )
    r.coordinates[id] += other;
  return r;
};

// minus
template<typename I,int d>
coordinate<I,d> coordinate<I,d>::operator-( coordinate<I,d> other ) const {
  auto r(*this);
  for ( int id=0; id<d; id++ )
    r.coordinates[id] -= other.coordinates[id];
  return r;
};
template<typename I,int d>
coordinate<I,d> coordinate<I,d>::operator-( I other ) const {
  auto r(*this);
  for ( int id=0; id<d; id++ )
    r.coordinates[id] -= other;
  return r;
};
template<typename I,int d>
coordinate<I,d> coordinate<I,d>::operator-() const {
  auto r(*this);
  for ( int id=0; id<d; id++ )
    r.coordinates[id] = -r.coordinates[id];
  return r;
};
template<typename I,int d>
void coordinate<I,d>::operator-=( coordinate<I,d> other ) {
  for ( int id=0; id<d; id++ )
    coordinates[id] -= other.coordinates[id];
};

// mult
template<typename I,int d>
coordinate<I,d> coordinate<I,d>::operator*( I f ) const {
  auto r(*this);
  for ( int id=0; id<d; id++ )
    r.coordinates[id] *= f;
  return r;
};
template<typename I,int d>
coordinate<I,d> coordinate<I,d>::operator/( I f ) const {
  auto r(*this);
  for ( int id=0; id<d; id++ )
    r.coordinates[id] /= f;
  return r;
};
template<typename I,int d>
coordinate<I,d> coordinate<I,d>::operator%( I other ) const {
  auto r(*this);
  for ( int id=0; id<d; id++ )
    r.coordinates[id] %= other;
  return r;
};
// equals and other comparisons
template<typename I,int d>
bool coordinate<I,d>::operator==( coordinate<I,d> other ) const {
  bool r{true};
  for ( int id=0; id<d; id++ )
    r = r and coordinates[id]==other.coordinates[id];
  return r;
};
template<typename I,int d>
bool coordinate<I,d>::operator!=( coordinate<I,d> other ) const {
  bool r{true};
  for ( int id=0; id<d; id++ )
    r = r or coordinates[id]!=other.coordinates[id];
  return r;
};
template<typename I,int d>
bool coordinate<I,d>::operator==( I other ) const {
  bool r{true};
  for ( int id=0; id<d; id++ )
    r = r and coordinates[id]==other;
  return r;
};
template<typename I,int d>
bool coordinate<I,d>::operator<=( coordinate<I,d> other ) const {
  bool r{true};
  for ( int id=0; id<d; id++ )
    r = r and coordinates[id]<=other.coordinates[id];
  return r;
};
template<typename I,int d>
bool coordinate<I,d>::operator<( coordinate<I,d> other ) const {
  bool r{true};
  for ( int id=0; id<d; id++ )
    r = r and coordinates[id]<other.coordinates[id];
  return r;
};
template<typename I,int d>
bool coordinate<I,d>::operator>( coordinate<I,d> other ) const {
  bool r{true};
  for ( int id=0; id<d; id++ )
    r = r and coordinates[id]>other.coordinates[id];
  return r;
};

// stuff
template<typename I,int d>
I coordinate<I,d>::linear( const coordinate<I,d>& layout ) const {
  int s = coordinates.at(0);
  for (int id=1; id<d; id++) {
    auto layout_dim = layout.coordinates[id];
    s = s*layout_dim + coordinates[id];
  }
  return s;
};
template<typename I,int d>
coordinate<I,d> coordmax( coordinate<I,d> current,coordinate<I,d> other ) {
  auto r(current);
  for ( int id=0; id<d; id++ ) {
    auto cmp = other.data()[id];
    if (cmp>r.data()[id]) r.data()[id] = cmp;
  }
  return r;
};
template<typename I,int d>
coordinate<I,d>  coordmin( coordinate<I,d> current,coordinate<I,d> other ) {
  auto r(current);
  for ( int id=0; id<d; id++ ) {
    auto cmp = other.data()[id];
    if (cmp<r.data()[id]) r.data()[id] = cmp;
  }
  return r;
};

template<typename I,int d>
void require_sorted( vector<coordinate<I,d>> idxs ) {
  auto v = idxs.front();
  for ( auto vv : idxs ) {
    if (v==vv or v<vv)
      v = vv;
    else
      throw("Indices need to be sorted");
  }    
};

template<typename I,int d>
bool coordinate<I,d>::before( const coordinate<I,d>& other ) const {
  return other.span()>=span();
};

/*
 * String-ifying
 */

template<typename I,int d>
string coordinate<I,d>::as_string() const {
  stringstream ss;
  ss << "<";
  auto sser = [&ss,need_comma=false] ( auto c ) mutable {
    if (need_comma) ss << ",";
    ss << c;
    need_comma = true; };
  for ( auto c : coordinates )
    sser(c); // ss << c << ",";
  ss << ">";
  return ss.str();
};

// template<typename I,int d>
// //template <>
// struct fmt::formatter<coordinate<I,d>> {
//  constexpr
//  auto parse(format_parse_context& ctx)
//        -> decltype(ctx.begin()) {
//    auto it = ctx.begin(),
//      end = ctx.end();
//    if (it != end && *it != '}')
//      throw format_error("invalid format");
//    return it;
//   }
//   template <typename FormatContext>
//   auto format
//     (const coordinate<I,d>& p, FormatContext& ctx)
//         -> decltype(ctx.out()) {
//     return format_to(ctx.out(),"{}", p.as_string());
//   }
// };

/*
 * Specializations
 */

template array<int,1> endpoint<int,1>(int);
template array<int,2> endpoint<int,2>(int);
template array<int,3> endpoint<int,3>(int);
template array<index_int,1> endpoint<index_int,1>(index_int);
template array<index_int,2> endpoint<index_int,2>(index_int);
template array<index_int,3> endpoint<index_int,3>(index_int);

template void require_sorted( vector<coordinate<int,1>> idxs );
template void require_sorted( vector<coordinate<int,2>> idxs );
template void require_sorted( vector<coordinate<index_int,1>> idxs );
template void require_sorted( vector<coordinate<index_int,2>> idxs );

template class coordinate<int,1>;
template class coordinate<int,2>;
template class coordinate<int,3>;
template class coordinate<index_int,1>;
template class coordinate<index_int,2>;
template class coordinate<index_int,3>;

template struct fmt::formatter<coordinate<int,1>>;
template struct fmt::formatter<coordinate<int,2>>;
template struct fmt::formatter<coordinate<index_int,1>>;
template struct fmt::formatter<coordinate<index_int,2>>;

template coordinate<int,1> coordmax<int,1>( coordinate<int,1>,coordinate<int,1> );
template coordinate<int,2> coordmax<int,2>( coordinate<int,2>,coordinate<int,2> );
template coordinate<index_int,1> coordmax<index_int,1>( coordinate<index_int,1>,coordinate<index_int,1> );
template coordinate<index_int,2> coordmax<index_int,2>( coordinate<index_int,2>,coordinate<index_int,2> );
template coordinate<int,1> coordmin<int,1>( coordinate<int,1>,coordinate<int,1> );
template coordinate<int,2> coordmin<int,2>( coordinate<int,2>,coordinate<int,2> );
template coordinate<index_int,1> coordmin<index_int,1>( coordinate<index_int,1>,coordinate<index_int,1> );
template coordinate<index_int,2> coordmin<index_int,2>( coordinate<index_int,2>,coordinate<index_int,2> );
