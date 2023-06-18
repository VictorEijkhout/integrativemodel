#include <iostream>
using std::cout;

#define CATCH_CONFIG_RUNNER
#include "catch2/catch_all.hpp"

void fthrow() { throw("foo"); };

TEST_CASE( "throw" ) {
  REQUIRE_THROWS( fthrow() );
};

int main(int argc,char **argv) {

  int result;
  try {
    result = Catch::Session().run( argc, argv );
  } catch (std::string c) {
    cout << "Unittesting aborted: " << c << '\n';
  } catch (...) {
    cout << "Unittesting aborted.\n";
  }
  return result;
}
