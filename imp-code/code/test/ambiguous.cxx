#include <memory>
using std::shared_ptr;
#include <functional>
using std::function;

class out {
};
class in1 {
};
class in2 {
};

class hasop {
public:
  hasop( function< shared_ptr<out>(const in1& i) > op ) {
  };
  hasop( function< shared_ptr<out>(const in2& i) > op ) {
  };
};

shared_ptr<out> f(const in1& i) {}

int main() {
  hasop hasf( &f );
  return 0;
}
