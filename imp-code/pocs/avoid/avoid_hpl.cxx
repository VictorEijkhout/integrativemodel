#include "fmt/format.h"
using fmt::format;
using fmt::print;

#include <fstream>
#include <iostream>
#include <memory>
using std::shared_ptr;
#include <string>
using std::string;
#include <vector>

#include "tasklib.hpp"

int main( int argc,char **argv) {
  int steps{1}, blocking{1}, nlocal{0}, latency{100};
  int nodes{3}, over{10}, cores{-1};
  bool dot{false}; int verbose{0};

  print("================\n");
  int ret = set_options(argc,argv,nlocal,latency,blocking,cores,nodes,over,steps,dot,verbose);
  if (ret!=0) return 1;
  if (steps>1) {
    print("HPL steps parameter is meaningless\n"); return 1; }
  if (blocking>1) {
    print("HPL blocking parameter is meaningless\n"); return 1; }

  print("Running {} for {} steps, with blocking={}",argv[0],steps,blocking);
  if (cores>0)
    print(", on {} cores",cores);
  print(", local domain {} pts, latency {} ops",nlocal,latency);
  print("\n");

  /*
   * Initial disjoint distribution
   */
  distribution blocked(nodes,over,3);
  print("================\n\n");

  /*
   * Build a task graph over `blocking' steps;
   * both global graphs and one per node.
   */
  parallel gauss(blocked.nnodes(),blocked);

  try {
    gauss.make_3d(blocked,blocking,verbose);
  } catch (string c) {
    print("Error building graph: {}",c); return -1;
  } catch (std::out_of_range o) {
    print("Error building graph\n"); throw(o);
  }

  print("Global graph has {} nodes\n",gauss.global_size());

  try {
    gauss.graph_building(verbose);
    gauss.graph_leveling(cores,verbose);
    gauss.graph_dotting(blocked,dot);
    gauss.graph_execution(steps,nlocal,latency,verbose);
  } catch ( string e ) {
    print("Abort: {}\n",e);
  } catch ( int e ) {
    print("Abort with code: {}\n",e);
  }

  return 0;
}
