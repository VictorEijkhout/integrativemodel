#include <iostream>
#include <vector>
#include <array>
using namespace std;

template<int d>
class vec { 
private: vector<int> a;
public: 
  vec( vector<int> a ) : a(a) {};
  vec( int x ) : vec( vector<int>(d,x) ) {};
};
  
template<int d>
class arr { 
public: array<int,d> a;
public: 
  arr( array<int,d> a ) : a(a) {};
  // arr( int x ) : arr( array<int,d>(d,x) ) {};
  arr( int x ) : arr( to_array<int,d>(d,x) ) {};
};

int main() {
  arr<2> a(1);
  cout << a.a.at(1) << '\n';
  return 0;
};
