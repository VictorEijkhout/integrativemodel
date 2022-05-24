#include <iostream>

class bag {
private:
  int lo,hi;
public:
  bag(int lo,int hi) { this->lo = lo; this->hi = hi; };

private:
  int cur;
public:
  bag &begin() { cur = lo; return *this; };
  bag &end() { cur = hi+1; return *this; };
  bool operator!=( bag &other ) {
    std::cout << lo << "," << hi << "," << cur << " vs " <<
      other.lo << "," << other.hi << "," << other.cur << std::endl;
    return lo!=other.lo || hi!=other.hi || cur!=other.cur; };
  void operator++() { cur++; };
  int operator*() { return cur; };
};

int main() {
  
  bag grocery(7,11);
  for ( auto item : grocery )
    std::cout << item << std::endl;
  
  return 0;
}
