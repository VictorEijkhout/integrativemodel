#include <iostream>
using std::cout;
using std::endl;

class has {
private:
  int data;
public:
  has(int data) : data(data) {
    cout << "made with: " << data << endl;
  };
};

int main() {
  has five(5);
  return 0;
}
