#include <iostream>
using std::cout;
using std::endl;

template <typename T>
class MemoAble {
private:
  mutable T data;
public:
  MemoAble(T data) : data(data) {};
  const T& Data(bool inc=false) const {
    if (inc) data++;
    return data;
  };
};

int main() {
  MemoAble<float> x(1.3);
  cout << x.Data() << endl;
  cout << x.Data(true) << endl;
  cout << x.Data() << endl;
  return 0;
}
