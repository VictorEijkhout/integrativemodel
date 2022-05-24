#include <iostream>
using std::cout;
using std::endl;

#include <utility>
using std::move;

class thing {
private:
  int i;
public:
  thing(int i) : i(i) {
    cout << "construct with i=" << i << endl;
  };
  thing( const thing &t ) : i(t.i+1) { 
    cout << "copy with i=" << i << endl;
  };
  thing another() { return thing(i+1); };
};

void f(const thing &t) {
  // code
};
void f(const thing &&t) {
  const thing &tt = t;
  f(tt);
  // same code
};

int main() {
  
  cout << "\nConstructor one" << endl;
  thing one(1);

  cout << "\nFunction one" << endl;
  f(one);

  cout << "\nCopy one -> two" << endl;
  thing two(one);

  cout << "\nFunction two rvalue" << endl;
  f(two.another());

  return 0;
}
