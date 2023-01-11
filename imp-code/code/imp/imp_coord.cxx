#include "imp_coord.h"

#include <array>
using std::array;
#include <string>
using std::string;
#include <vector>
using std::vector;
#include <sstream>
using std::ostream, std::stringstream;

#include <memory>
using std::shared_ptr, std::make_shared;

#include <cassert>
#include <fmt/format.h>
using fmt::print,fmt::format;

/*! Construct non-inclusive upper bound on
  a brick of size `s', the input.

  Probably only works for d=1 and d=2
*/
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
//! Make a coordinate with all components invalid
//snippet pcoorddim
template<typename I,int d>
coordinate<I,d>::coordinate() {
  for (int id=0; id<d; id++)
    coordinates.at(id) = -1;
};
//! Make a coordinate from a given span
template<typename I,int d>
coordinate<I,d>::coordinate( I s )
  : coordinate( endpoint<I,d>(s) ) {
};
template<typename I,int d>
coordinate<I,d>::coordinate( std::array<I,d> c)
  : coordinates( c ) {
};
//! Make a coordinate with one point per process
template<typename I,int d>
coordinate<I,d>::coordinate( const environment& e )
  : coordinate( e.nprocs() ) {
};
//snippet end

/*
 * Access
 */
/*! Span is another word for volume,
  if the coordinate is the non-inclusive upper bound
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
coordinate<I,d> coordinate<I,d>::operator+( const coordinate<I,d>& other ) const {
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
coordinate<I,d> coordinate<I,d>::operator-( const coordinate<I,d>& other ) const {
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
void coordinate<I,d>::operator-=( const coordinate<I,d>& other ) {
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
bool coordinate<I,d>::operator==( const coordinate<I,d>& other ) const {
  // print("test equal\n");
  bool r{true};
  for ( int id=0; id<d; id++ )
    r = r and coordinates[id]==other.coordinates[id];
  return r;
};
template<typename I,int d>
bool coordinate<I,d>::operator==( I other ) const {
  bool r{true};
  for ( int id=0; id<d; id++ )
    r = r and coordinates[id]==other;
  return r;
};

// template<typename I,int d>
// bool coordinate<I,d>::operator==( coordinate<I,d>&& other ) const {
//   return *this==std::move(other);
// };

template<typename I,int d>
bool coordinate<I,d>::operator!=( const coordinate<I,d>& other ) const {
  //  print("test neq\n");
  return not (*this==other);
  // bool r{ coordinates[0]!=other.coordinates[0] };
  // print("not eq {}",r);
  // for ( int id=1; id<d; id++ ) {
  //   r = r or coordinates[id]!=other.coordinates[id];
  //   print("{}",r);
  // } print("\n");
  // return r;
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

/*
 * Linearization
 */

/*! Linear location of this coordinate in a span */
template<typename I,int d>
I coordinate<I,d>::linear_location_in( const coordinate<I,d>& layout ) const {
  int id=0;
  if ( coordinates[id]<0 or coordinates[id]>=layout.coordinates[id] )
    throw( format("coordinate {} not contained in: {}",as_string(),layout.as_string()) );
  int s = coordinates.at(id);

  for (int id=1; id<d; id++) {
    auto layout_dim = layout.coordinates[id];
    if ( coordinates[id]<0 or coordinates[id]>=layout.coordinates[id] )
      throw( format("coordinate {} not contained in: {}",as_string(),layout.as_string()) );
    s = s*layout_dim + coordinates[id];
  }
  return s;
};
/*! Linear location in this span of another coordinate */
template<typename I,int d>
I coordinate<I,d>::linear_location_of( const coordinate<I,d>& inside ) const {
  return inside.linear_location_in( *this );
};
/*! Coordinate from linear location */
template<typename I,int d>
coordinate<I,d> coordinate<I,d>::location_of_linear( I s ) const {
  coordinate<I,d> loc;
  for (int id=0; id<d; id++) {
    I trail_block{1};
    for (int idd=id+1; idd<d; idd++)
      trail_block *= data()[idd];
    loc[id] = s / trail_block;
    s = s % trail_block;
  }
  return loc;
};

// stuff
/*! Pointwise max of two coordinates */
template<typename I,int d>
coordinate<I,d> coordmax( coordinate<I,d> current,coordinate<I,d> other ) {
  auto r(current);
  for ( int id=0; id<d; id++ ) {
    auto cmp = other.data()[id];
    if (cmp>r.data()[id]) r.data()[id] = cmp;
  }
  return r;
};
/*! Pointwise min of two coordinates */
template<typename I,int d>
coordinate<I,d>  coordmin( coordinate<I,d> current,coordinate<I,d> other ) {
  auto r(current);
  for ( int id=0; id<d; id++ ) {
    auto cmp = other.data()[id];
    if (cmp<r.data()[id]) r.data()[id] = cmp;
  }
  return r;
};

/*! Require a vector of coordinates to be sorted increasing,
  otherwise throw an exception
*/
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

/*! Test whether this coordinate comes lexicographically before another */
template<typename I,int d>
bool coordinate<I,d>::before( const coordinate<I,d>& other ) const {
  return other.span()>=span();
};

/*
 * String-ifying
 */

/*! Render a coordinate as string */
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

/*! Test if a coordinate_set contains some coordinate */
template<typename I,int d>
bool coordinate_set<I,d>::contains( const coordinate<I,d> &p ) const {
  for ( auto pp : set )
    if (p==pp)
      return true;
  return false;
};

/*! Add a coordinate to a coordinate set */
template<typename I,int d>
void coordinate_set<I,d>::add( const coordinate<I,d>& p ) {
  if (set.size()>0 &&
      p.dimensionality()!=set.at(0).dimensionality())
    throw(fmt::format("Can not add vector of dim {}: previous {}",
		      p.dimensionality(),set.at(0).dimensionality()));
  if (!contains(p))
    set.push_back(p);
};

/*
 * Formatters
 */
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
ostream &operator<<(ostream &os,const coordinate<I,d> &c) {
  os << c.as_string();
  return os;
};
template<typename I, int d>
ostream &operator<<(ostream &os,const shared_ptr<coordinate<I,d>> &c) {
  os << c->as_string();
  return os;
};

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
template void require_sorted( vector<coordinate<int,3>> idxs );

template void require_sorted( vector<coordinate<index_int,1>> idxs );
template void require_sorted( vector<coordinate<index_int,2>> idxs );
template void require_sorted( vector<coordinate<index_int,3>> idxs );

template class coordinate<int,1>;
template class coordinate<int,2>;
template class coordinate<int,3>;
template class coordinate<index_int,1>;
template class coordinate<index_int,2>;
template class coordinate<index_int,3>;

template class coordinate_set<int,1>;
template class coordinate_set<int,2>;
template class coordinate_set<int,3>;
template class coordinate_set<index_int,1>;
template class coordinate_set<index_int,2>;
template class coordinate_set<index_int,3>;

template struct fmt::formatter<coordinate<int,1>>;
template struct fmt::formatter<coordinate<int,2>>;
template struct fmt::formatter<coordinate<int,3>>;

template struct fmt::formatter<coordinate<index_int,1>>;
template struct fmt::formatter<coordinate<index_int,2>>;
template struct fmt::formatter<coordinate<index_int,3>>;

template coordinate<int,1> coordmax<int,1>( coordinate<int,1>,coordinate<int,1> );
template coordinate<int,2> coordmax<int,2>( coordinate<int,2>,coordinate<int,2> );
template coordinate<int,3> coordmax<int,3>( coordinate<int,3>,coordinate<int,3> );

template coordinate<index_int,1> coordmax<index_int,1>( coordinate<index_int,1>,coordinate<index_int,1> );
template coordinate<index_int,2> coordmax<index_int,2>( coordinate<index_int,2>,coordinate<index_int,2> );
template coordinate<index_int,3> coordmax<index_int,3>( coordinate<index_int,3>,coordinate<index_int,3> );

template coordinate<int,1> coordmin<int,1>( coordinate<int,1>,coordinate<int,1> );
template coordinate<int,2> coordmin<int,2>( coordinate<int,2>,coordinate<int,2> );
template coordinate<int,3> coordmin<int,3>( coordinate<int,3>,coordinate<int,3> );

template coordinate<index_int,1> coordmin<index_int,1>( coordinate<index_int,1>,coordinate<index_int,1> );
template coordinate<index_int,2> coordmin<index_int,2>( coordinate<index_int,2>,coordinate<index_int,2> );
template coordinate<index_int,3> coordmin<index_int,3>( coordinate<index_int,3>,coordinate<index_int,3> );

template ostream &operator<<(ostream &os,const coordinate<int,1> &c);
template ostream &operator<<(ostream &os,const coordinate<int,2> &c);
template ostream &operator<<(ostream &os,const coordinate<int,3> &c);

template ostream &operator<<(ostream &os,const coordinate<index_int,1> &c);
template ostream &operator<<(ostream &os,const coordinate<index_int,2> &c);
template ostream &operator<<(ostream &os,const coordinate<index_int,3> &c);
