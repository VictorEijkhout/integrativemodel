#include <cmath>

#include <functional>
using std::function;

#include <iostream>
#include <iomanip>
using std::cout;
using std::endl;
using std::setw;

#include <string>
using std::string;

#include <vector>
using std::vector;

#include <Eigen/Dense>
using Eigen::MatrixXd;
using Eigen::VectorXd;

void report_timing( vector<double> timings ) {
  int np = timings.size();
  if (np==0) return;
  auto s = 0*timings.at(0), tmin = timings.at(0), tmax = timings.at(0);
  for (int i=0; i<np; i++) {
    auto t = timings.at(i);
    if (t>tmax) tmax = t; if (t<tmin) tmin = t;
    s += timings.at(i);
  }
  s /= np;
  cout << "Max timing: " << tmax << ", min: " << tmin 
       << "; avg over " << np << ": " << s << endl;
};

class grid {
private:
  vector<double> point_work;
  vector<int> partition_points;
  MatrixXd adjacency;
public:
  grid(int N,int p) { // p segments
    point_work = vector<double>(N);
    partition_points = vector<int>(p+1); // every partition has a left point; one extra
    for (int i=0; i<p; i++)
      partition_points[i] = i*N*(1./p);
    partition_points[p] = N;
    adjacency = MatrixXd::Constant(p,p-1,0.);
    for (int ip=0; ip<p-1; ip++) // contribution from the right
      adjacency(ip,ip) = -1.;
    for (int ip=1; ip<p; ip++) // contribution from the left
      adjacency(ip,ip-1) = 1.;
  };
  vector<double> compute_timing() {
    int npartitions = partition_points.size()-1;
    vector<double> timing(npartitions);

    for (int ipart=0; ipart<npartitions; ipart++) {
      int istart = partition_points.at(ipart), iend = partition_points.at(ipart+1);
      double total=0;
      for (int point=istart; point<iend; point++)
	total += point_work.at(point);
      timing.at(ipart) = total;
    }
    return timing;
  };
  void balance( vector<double> times ) {
    int nsegments = partition_points.size()-1, ninterior = nsegments-1;
    if (times.size()!=nsegments) {
      cout << "#times=" << times.size() << ", #segments=" << nsegments << endl;
      throw(1); }

    double avg_time = 0; for ( auto t : times ) avg_time += t; avg_time /= nsegments;
    VectorXd imbalance = VectorXd::Constant(nsegments,-avg_time);
    for (int it=0; it<times.size(); it++)
      imbalance[it] += times[it];
    cout << "Imbalance:\n" << imbalance.transpose() << endl;
    auto normal_matrix = adjacency.transpose() * adjacency;
    //cout << "Normal matrix:\n" << normal_matrix << endl;
    auto ata_fact = normal_matrix.ldlt();
    auto scale_balance = adjacency.transpose() * imbalance;
    auto loadmove = ata_fact.solve( scale_balance );
    cout << "Move amounts:\n" << loadmove.transpose() << endl ;

    if (loadmove.rows()!=ninterior) {
      cout << "loadmove " << loadmove.rows() << " vs interior " << ninterior << endl;
      throw(1); }
    vector<int> point_motion(ninterior);
    cout << "Move points: ";
    for (int ipoint=1; ipoint<=ninterior; ipoint++) {
      double flow = loadmove(ipoint-1,0) / 2;
      { int iseg=ipoint-1; // iseg<=ipoint; iseg++) { // left and right of the point
	auto seg_size = partition_points.at(iseg+1)-partition_points.at(iseg);
	auto load_per_point = times.at(iseg) / seg_size;
	auto points_to_move = flow / load_per_point; cout << points_to_move << " ";
	point_motion.at(iseg) = points_to_move;
      }
    } cout << endl;
    for (int ipoint=1; ipoint<partition_points.size()-1; ipoint++)
      partition_points.at(ipoint) =
	std::min( partition_points.at(partition_points.size()-1),
		  std::max( partition_points.at(ipoint)+point_motion[ipoint-1],
			    partition_points.at(ipoint-1)+2 )
		  );
  };
  void report_quantity( std::function< void(int i) > draw3, string fill ) {
    int nsegments = partition_points.size()-1;
    for (int ip=0; ip<nsegments; ip++) {
      auto segment_size = partition_points.at(ip+1)-partition_points.at(ip);
      cout << "|";
      for (int pt=0; pt<std::max((segment_size-1)/2-2,1); pt++) cout << fill;
      draw3(ip);
      for (int pt=0; pt<std::max((segment_size-1)/2-1,1); pt++) cout << fill;
      if ((segment_size-1)%2==1) cout << fill;
    }
    cout << "|" << endl;
  };
  void report_partition() {
    report_quantity( [](int i)->void { cout << "..."; },"." );
    report_quantity( [this] (int ip) -> void {
	auto segment_size = partition_points.at(ip+1)-partition_points.at(ip);
	cout << setw(3) << segment_size; }, " " );
  };
  void report_partition( vector<double> times ) {
    report_quantity( [times] (int ip) -> void {
	cout << setw(3) << static_cast<int>(times.at(ip)); }, " " );
    report_partition();
  };
  void set_moving_load( int step,int laststep ) {
    int gsize = point_work.size();
    double center = gsize * step / (1.*laststep);
    cout << "Load centered at " << center << endl;
    for ( int i=0; i<gsize; i++) {
      double imst = (i-center)/10, work = sqrt(gsize) * exp( -imst*imst );
      point_work.at(i) = 1 + work;
    }
  };
};

#define STEPS 50
#define PROCS 10

int main() {
  grid static_grid(100,PROCS), adaptive_grid(100,PROCS);
  adaptive_grid.report_partition();
  for (int istep=0; istep<STEPS; istep++) {
    cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << endl;
    cout << "Step " << istep << endl;

    static_grid.set_moving_load(istep,STEPS);
    if (istep==STEPS/2) {
      cout << "Static grid:" << endl;
      auto t = static_grid.compute_timing();
      report_timing(t);
    }

    adaptive_grid.set_moving_load(istep,STEPS);
    {
      auto t = adaptive_grid.compute_timing();
      adaptive_grid.report_partition(t);
      report_timing(t);
      adaptive_grid.balance(t);
    }

    cout << endl;
  }
  return 0;
}
