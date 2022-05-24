#include <memory>
#include <stdio.h>

class idx : public std::enable_shared_from_this<idx> {
public:
  int v{0};
public:
  idx(int i ) { v=i; };
  std::shared_ptr<idx> simplify() {
    if (v%2==0)
      return std::shared_ptr<idx>( shared_from_this() ); /* wrong */
    //return std::shared_ptr<idx>( new idx(*this) ); /* right */
    else
      return std::shared_ptr<idx>( new idx(v-1) );
  };
};

int main() {
  std::shared_ptr<idx>
    i1 = std::shared_ptr<idx>( new idx(1) ),
    i2 = i1->simplify();

  printf("should be even: %d\n",i2->v);

  std::shared_ptr<idx>
    i3 = std::shared_ptr<idx>( new idx(6) ),
    i4 = i3->simplify();

  printf("should be even: %d\n",i4->v);

  return 0;
}
