#include <iostream>
using std::cout;
using std::endl;

class bag {
private:
  int first,last;
public:
  bag(int first,int last) : first(first),last(last) {};
  int seek{0};
  bag begin() const {
    bag head(*this); head.seek = first; return head;
  };
  bag end() const {
    bag limen(*this); return limen;
  };
  bool operator!=( const bag &test ) const {
    return seek<=test.last;
  };
  void operator++() { seek++; };
  int operator*() { return seek; };
  bool has(int tst) const {
    for (auto seek : *this )
      if (seek==tst) return true;
    return false;
  };
};

int main() {

  bag digits(0,9);

  bool find3{false};
  for ( auto seek : digits )
    find3 = find3 || (seek==3);
  cout << "found 3: " << find3 << endl;

  bool find15{false};
  for ( auto seek : digits )
    find15 = find15 || (seek==15);
  cout << "found 15: " << find15 << endl;

  cout << "f3: " << digits.has(3) << endl;
  cout << "f15: " << digits.has(15) << endl;

  return 0;
}
