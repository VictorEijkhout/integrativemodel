#include <stdlib.h>
#include <stdio.h>

#include <functional>

class decomposition {
public:
  decomposition() {}; decomposition( decomposition *d ) {};
};

class object;
class distribution : public decomposition {
public:
  distribution(decomposition *d) : decomposition(d) {};
  distribution(distribution *d) : distribution(dynamic_cast<decomposition*>(d)) {
    get_visibility = d->get_visibility; };

  std::function< int(int p) > get_visibility {
    [this] (int p) -> int { printf("local get visibility in {}\n"); throw(1); } };
  std::function < object*() > new_object;
};

class object : public distribution {
public:
  object( distribution *d ) : distribution(d) {};
};

class mpi_distribution : virtual public distribution {
public:
  mpi_distribution( decomposition *d ) : distribution(d) { printf("mpi dist\n");
    set_dist_factory(); set_numa();
  };
  mpi_distribution( distribution *d ) : distribution(d) { printf("mpi dist copy\n"); };

  void set_numa() { printf("set numa\n");
    get_visibility = [this] (int p) -> int { printf("mpi visibility\n"); return 1; };
  };
  void set_dist_factory();
};

class mpi_object : public object {
public:
  mpi_object( distribution *d ) : object(d) { printf("mpi object1\n"); };
};

void mpi_distribution::set_dist_factory() { printf("installating mpi_dist factory\n");
  // factory from new objects
  new_object = [this] (void) -> object*
    { printf("Factory new mpi object from <<{}>>\n");
      return new mpi_object(this); };
};

int main() {
  decomposition *c = new decomposition();
  distribution *d = new mpi_distribution(c);
  object *o = new mpi_object(d);
  printf("test one\n");
  o->get_visibility(0);
  printf(".. concluded\n");
  o = d->new_object();
  printf("test two\n");
  o->get_visibility(0);
  printf(".. concluded\n");

  return 0;
}
