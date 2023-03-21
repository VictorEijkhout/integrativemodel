#include <array>
#include <numeric>
#include <iostream>
using namespace std;

template<int d>
class foo {
private:
  array<float,d> dat;
public:
  foo( array<float,d> dat ) {
    cout << accumulate(dat.begin(),dat.end(),0.f) << '\n';
  };
  foo( float dat ) requires (d==1) {};
};

int main() {
  foo<1> f1( {1.f} );
  foo<2> f2( {1.f,1.f} );
  //    foo<1> f1( {1.f,1.f} );
  foo<1> f3( 1.f );
  foo<2> f4( 1.f );
  return 0;
}
