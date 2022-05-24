/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-9
 ****
 **** balance_functions.cxx : implementations of the load balancing support functions
 ****
 ****************************************************************/

#include <iostream>
using std::cout;
using std::endl;
#include <iomanip>
using std::setw;

#include "imp_base.h"
using std::shared_ptr;
using std::string;
using std::vector;

// weird shit just for this template
#include <Eigen/Dense>
using Eigen::MatrixXd;
using Eigen::VectorXd;
#include <thread>
#include <chrono>

// prototypes for this file
#include "balance_functions.h"

//snippet transform_average
shared_ptr<distribution> transform_by_average
    (shared_ptr<distribution> unbalance,double *stats_data) {
  if (unbalance->get_dimensionality()!=1)
    throw(string("Can only average in 1D"));
  if (!unbalance->is_known_globally())
    throw(fmt::format
	  ("Can not transform-average <<{}>>: needs globally known",unbalance->get_name()));

  auto decomp = unbalance->get_decomposition();
  parallel_structure astruct(decomp);
  int nprocs = decomp.domains_volume();
  for (int p=0; p<nprocs; p++) {
    auto me = unbalance->coordinate_from_linear(p);
    double
      cleft = 1./3, cmid = 1./3, cright = 1./3,
      work_left,work_right,work_mid = stats_data[p];
    index_int size_left=0,size_right=0,
      size_me = unbalance->volume(me);

    if (p==0) {
      size_left = 0; work_left = 0;
      cleft = 0; cmid = 1./2; cright = 1./2;
    } else {
      size_left = unbalance->volume( me-1 );
      work_left = stats_data[p-1];
    }

    if (p==nprocs-1) {
      size_right = 0; work_right = 0;
      cright = 0; cmid = 1./2; cleft = 1./2;    
    } else {
      size_right = unbalance->volume( me+1 );
      work_right = stats_data[p+1];
    }

    index_int new_size = ( cleft * work_left * size_left + cmid * work_mid * size_me
			   + cright * work_right *size_right ) / 3.;
    // fmt::print("{} New size: {},{},{} -> {}\n",
    // 	       me.as_string(),size_left,size_me,size_right,new_size);

    auto idx = shared_ptr<indexstruct>( new contiguous_indexstruct(1,new_size) );
    auto old_pstruct = unbalance->get_processor_structure(me);
    auto new_pstruct = shared_ptr<multi_indexstruct>
      ( new multi_indexstruct( std::vector<shared_ptr<indexstruct>>{ idx } ) );
    astruct.set_processor_structure(me,new_pstruct);
  }
  astruct.set_is_known_globally();
  return unbalance->new_distribution_from_structure(astruct);
}
//snippet end

shared_ptr<distribution> transform_by_diffusion
    (shared_ptr<distribution> unbalance,shared_ptr<object> stats_data,
     MatrixXd adjacency,
     bool trace) {
  auto decomp = unbalance->get_decomposition();

  if (trace) fmt::print("Starting distribution: {}\n",unbalance->as_string());
  auto partition_points = unbalance->partitioning_points();
  int p = partition_points.size()-1; auto N = partition_points.at(p);
  int nsegments = partition_points.size()-1, ninterior = nsegments-1;

  auto times = stats_data->get_raw_data();
  auto ntimes = stats_data->volume(unbalance->proc_coord(/*decomp*/));
  if (ntimes!=nsegments)
    throw(fmt::format("#times={} <> #segments={}",ntimes,nsegments));

  double avg_time = 0;
  for (int it=0; it<ntimes; it++ ) avg_time += times[it];
  avg_time /= nsegments;
  VectorXd
    imbalance = VectorXd::Constant(nsegments,-avg_time),
    loadmove;
  for (int it=0; it<ntimes; it++)
    imbalance[it] += times[it];
  if (adjacency.rows()>=adjacency.cols()) {
    auto normal_matrix = adjacency.transpose() * adjacency;
    auto ata_fact = normal_matrix.ldlt();
    auto scale_balance = adjacency.transpose() * imbalance;
    loadmove = ata_fact.solve( scale_balance );
  } else {
    auto normal_matrix = adjacency * adjacency.transpose();
    auto ata_fact = normal_matrix.ldlt();
    auto balance_solve = ata_fact.solve( imbalance );
    loadmove = adjacency.transpose() * balance_solve;
  }

  if (loadmove.rows()!=ninterior)
    throw(fmt::format("loadmove {} vs interior {}",loadmove.rows(),ninterior));
  vector<int> point_motion(ninterior);
  //cout << "Move points: ";
  for (int ipoint=1; ipoint<=ninterior; ipoint++) {
    double flow = loadmove(ipoint-1,0) / 2;
    { int iseg=ipoint-1; // iseg<=ipoint; iseg++) { // left and right of the point
      auto seg_size = partition_points.at(iseg+1)-partition_points.at(iseg);
      auto load_per_point = times[iseg] / seg_size;
      auto points_to_move = flow / load_per_point;
      //cout << points_to_move << " ";
      point_motion.at(iseg) = points_to_move;
    }
  } //cout << endl;
  for (int ipoint=1; ipoint<partition_points.size()-1; ipoint++)
    partition_points.at(ipoint) =
      std::min( partition_points.at(partition_points.size()-1),
		std::max( partition_points.at(ipoint)+point_motion[ipoint-1],
			  partition_points.at(ipoint-1)+2 )
		);

  vector<index_int> localsizes(p);
  for (int ip=0; ip<p; ip++)
    localsizes.at(ip) = partition_points.at(ip+1)-partition_points.at(ip);
  parallel_structure shifted(decomp);
  shifted.create_from_local_sizes(localsizes);
  if (trace) fmt::print("New struct {}\n",shifted.as_string());
  return unbalance->new_distribution_from_structure(shifted);
}

MatrixXd AdjacencyMatrix1D(int procs) {
  MatrixXd adjacency = MatrixXd::Constant(procs,procs-1,0.);
  for (int ip=0; ip<procs-1; ip++) // contribution from the right
    adjacency(ip,ip) = -1.;
  for (int ip=1; ip<procs; ip++) // contribution from the left
    adjacency(ip,ip-1) = 1.;
  return adjacency;
}

#define index2d(i,j,m,n) (i)*(m)+(j)
MatrixXd AdjacencyMatrix2D(int procs) {
  int cols;
  for (cols=sqrt(procs); cols>1 && procs%cols!=0; cols--) ;
  if (cols==1) { fmt::print("Processors can not be gridded\n"); throw(1); }
  int rows = procs/cols;
  MatrixXd adjacency = MatrixXd::Constant(procs,rows*(cols-1)+cols*(rows-1),0.);
  // in each row, between columns
  for (int r=0; r<rows-1; r++) {
    for (int c=0; c<cols-1; c++)
      adjacency( index2d(r,c,rows,cols) ) = -1.; // from the right
    for (int c=1; c<cols; c++)
      adjacency( index2d(r,c,rows,cols-1) ) = 1.; // from the right
  }
  // in each column, between rows
  for (int c=1; c<cols; c++) {
    for (int r=0; r<rows-1; r++)
      adjacency( index2d(r,c,rows,cols) ) = -1.; // from the right
    for (int r=1; r<rows; r++)
      adjacency( index2d(r,c,rows,cols-1) ) = 1.; // from the right
  }
  return adjacency;
}

void work_moving_weight( kernel_function_args , int globalstep, int laststep ) {
  auto invector = invectors.at(0);
  const auto indistro = invector->get_distribution(),
    outdistro = outvector->get_distribution();
  int
    dim = p.get_same_dimensionality(outdistro->get_dimensionality()),
    k = outdistro->get_orthogonal_dimension();
  if (k>1)
    throw(fmt::format("Moving weight not implemented for k>1: got {}",k));
  auto outdata = outvector->get_data(p);
  
  // description of the indices on which we work
  auto pstruct = outdistro->get_processor_structure(p);
  domain_coordinate
    pfirst = pstruct->first_index_r(), plast = pstruct->last_index_r();
  
  // placement in the global data structures
  auto out_nstruct = outdistro->get_numa_structure();
  //auto out_gstruct = outdistro->get_global_structure();
  domain_coordinate
    out_nsize = out_nstruct->local_size_r(),
    out_offsets = outdistro->offset_vector();
  auto
    out_gsize = outdistro->global_volume();
  
  if (dim==1) {
    double center = globalstep*out_gsize/(1.*laststep), sum=0;
    for (index_int i=pfirst[0]; i<=plast[0]; i++) {
      index_int I = INDEX1D(i,out_offsets,out_nsize);
      //fmt::print("[{}] copy global index {}@{}:{}\n",p->as_string(),i,I,indata[I]);
      double
	imst = (i-center)/10,
	w = 1 + sqrt(out_gsize) * exp( -imst*imst );
      outdata.at(I) = w; sum += w;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(sum)));
  } else
    throw(fmt::format("Moving weight not implemented for d={}",dim));
}

void report_quantity
    ( vector<index_int> partition_points,
      std::function< void(index_int i) > draw3,
      string fill ) {
  int nsegments = partition_points.size()-1;
  for (int ip=0; ip<nsegments; ip++) {
    auto segment_size = static_cast<int>(partition_points.at(ip+1)-partition_points.at(ip));
    cout << "|";
    for (int pt=0; pt<std::max((segment_size-1)/2-2,1); pt++) cout << fill;
    draw3(ip);
    for (int pt=0; pt<std::max((segment_size-1)/2-1,1); pt++) cout << fill;
    if ((segment_size-1)%2==1) cout << fill;
  }
  cout << "|" << endl;
};

void report_partition( vector<index_int> partition_points, double *times ) {
  int np = partition_points.size()-1;
  index_int gsize = partition_points.at(np);
  if (gsize<132) {
    report_quantity
      ( partition_points,
	[times] (int ip) -> void {
	cout << setw(3) << static_cast<int>(times[ip]); }, " " );
    report_quantity
      ( partition_points,
	[](int i)->void { cout << "..."; },"." );
    report_quantity
      ( partition_points,
	[partition_points] (int ip) -> void {
	auto segment_size = partition_points.at(ip+1)-partition_points.at(ip);
	cout << setw(3) << segment_size; }, " " );
  } else {
    for (int p=0; p<np; p++)
      fmt::print("P={} w={} s={}-{}\n",
		 p,times[p],partition_points.at(p),partition_points.at(p+1));
  }
  double mintime{times[0]},maxtime{0},avgtime{0};
  for (int it=0; it<np; it++) {
    avgtime += times[it];
    if (times[it]>maxtime) maxtime = times[it];
    if (times[it]<mintime) mintime = times[it];
  }
  avgtime /= np;
  cout << "Maximum processor time: " << maxtime << endl;
  cout << "Minimum processor time: " << mintime << endl;
  cout << "Average processor time: " << avgtime << endl; 
};
