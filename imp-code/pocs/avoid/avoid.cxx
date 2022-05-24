#include <iostream>
#include <vector>

class task {
private:
  int proc{-1},step{-1};
  std::vector
public:
  task(){};
  task(int p,int s) : proc(p),int(s) {};
};

class taskgraph {
private:
  std::vector< std::shared_ptr<task> > tasks;
public:
  taskgraph() {};
  void add( task t ) { tasks.push_back(t); };
};
  
int main() {
  int blocking{5};
  int nodes{3},over{10};
  int nprocs = nodes*over;

  /*
   * Build a task graph over `blocking' steps
   */
  taskgraph graph;
  for (int is=0; is<blocking; is++) {
    for (int ip=0; ip<nprocs; ip++) {
      task t(ip,is);
      if (is>0)
      graph.add(t);
    }
  }

  return 0;
}
