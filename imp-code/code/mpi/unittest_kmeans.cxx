/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2020
 ****
 **** Unit tests for the MPI product backend of IMP
 **** based on the CATCH framework (https://github.com/philsquared/Catch)
 ****
 **** kmeans clustering
 ****
 ****************************************************************/

#include <stdlib.h>
#include <stdio.h>

#include "catch.hpp"

#include "mpi_base.h"
#include "mpi_ops.h"
#include "mpi_static_vars.h"
#include "kmeans_functions.h"

using std::make_shared;
using std::shared_ptr;

using std::string;
using std::vector;

using fmt::format;
using fmt::memory_buffer;
using fmt::format_to;
using fmt::to_string;

// specific parameters for kmeans
int nsteps, k, globalsize;
two_object_struct two_objects;

void dummy_centers_1d( shared_ptr<object> centers,int icase ) {
  auto comm = centers->get_communicator(); //static_cast<communicator*>(centers.get());
  int ntids = comm.nprocs(), mytid = comm.procid();
  int
    k = centers->global_volume(); //get_orthogonal_dimension();
  printf("ortho: %d\n",k);
  data_pointer center_data;
  try { center_data = centers->get_data(mycoord);
  } catch (...) { throw(string("could not get data in initial centers")); }

  switch (icase) {
  case 0 : // centers 1D, equally spaced
  case 1 : // centers 1D, equally spaced
    for (int ik=0; ik<k; ik++) {
      center_data.at(ik) = ( (double)ik+1 )/(k+1);
      //printf("[%d] set %d as %e\n",mytid,ik,center_data.at(ik));
    }
    break;
  case 2 : // centers 1D, random
    srand((int)(mytid*(double)RAND_MAX/ntids));
    for (int ik=0; ik<k; ik++) {
      float randomfraction = (rand() / (double)RAND_MAX);
      center_data.at(ik) = randomfraction;
    }
    break;
  }
}

//! Set all coordinates on this processor to the normalized processor number
void coordinates_on_center_1d( kernel_function_args )
{
  auto outdata = outvector->get_data(p);

  auto outdistro = outvector->get_distribution();
  index_int
    tar0 = outdistro->location_of_first_index( outdistro,p ),
    len = outvector->volume(p),
    gsize = outvector->global_volume();

  int k = outvector->get_orthogonal_dimension();

  int tar = tar0;
  for (index_int i=0; i<len; i++) {
    index_int iset = outvector->first_index_r(p).coord(0)+1;
    double v = ( (double)iset )/gsize;
    //printf("[%d] set l:%d g:%d to %e\n",p->coord(0),i,tar0+i,v);
    for (int kk=0; kk<k; kk++)
      outdata.at(tar++) = v;
  }
};

void coordinates_linear( kernel_function_args )
{
  auto outdata = outvector->get_data(p);

  int k = outvector->get_orthogonal_dimension();

  // description of the indices on which we work
  auto pstruct = outvector->get_processor_structure(p);
  domain_coordinate
    pfirst = pstruct->first_index_r(), plast = pstruct->last_index_r();

  // various structures
  auto out_nstruct = outvector->get_numa_structure();
  auto out_gstruct = outvector->get_enclosing_structure();
  domain_coordinate
    out_nsize = out_nstruct->local_size_r(),
    out_gsize = out_gstruct->local_size_r(),
    out_offsets = out_nstruct->first_index_r() - out_gstruct->first_index_r();

  if (outvector->get_dimensionality()==1) {
    for (index_int i=pfirst[0]; i<=plast[0]; i++) {
      index_int
	I = INDEX1D(i,out_offsets,out_nsize),
	Ig = COORD1D(i,out_gsize);
      // fmt::print("[{}] index {} location {} is global {}\n",
      // 		 p->as_string(),i,I,Ig);
      for (int kk=0; kk<k; kk++)
	outdata.at(I*k+kk) = (double)Ig;
    }
  } else
    throw(string("kmeans only works on linear coordinates"));

};

TEST_CASE( "work with predictable centers","[1]" ) {
  return;
  INFO( "mytid=" << mytid );

  for (int icase=0; icase<=2; icase++) {
    int dim,ncluster,globalsize;

    INFO( "case=" << icase );

    // Cases
    // 0: one dimensional, put the initial points on the centers
    // 1: same as zero, but using collectives
    // 2: one dimensional, random points
    switch (icase) {
    case 0 :
    case 1 :
    case 2 :
      ncluster = env->get_architecture()->nprocs(); dim = 1; globalsize = ncluster;
      break;
    default: printf("No case %d implemented\n",icase); throw(5);
    }
    INFO( "number of clusters: " << ncluster );
    
    /* 
     * All the declarations; dependent on problem parameters
     */

    // centers are replicated, size = ncluster, ortho = dim
    auto kreplicated = shared_ptr<distribution>( new mpi_replicated_distribution(decomp,dim,ncluster) );
    auto centers = shared_ptr<object>( new mpi_object( kreplicated ) );
    CHECK( centers->get_orthogonal_dimension()==1 );
    CHECK( centers->global_volume()==dim*ncluster );
    
    // coordinates are N x dim with the N distributed
    auto twoblocked = shared_ptr<distribution>( make_shared<mpi_block_distribution>
						   (decomp,dim,-1,globalsize) );
    auto coordinates = shared_ptr<object>( new mpi_object( twoblocked ) );

    // distances are Nxk, with the N distributed
    //snippet kmeansdistance
    auto kblocked = shared_ptr<distribution>( make_shared<mpi_block_distribution>
						 (decomp,ncluster,-1,globalsize) );
    auto distances = shared_ptr<object>( new mpi_object( kblocked ) );
    //snippet end

    // grouping should be N integers, just use reals
    //snippet kmeansgroup
    auto blocked = shared_ptr<distribution>( make_shared<mpi_block_distribution>
						(decomp,-1,globalsize) );
    auto grouping = shared_ptr<object>( new mpi_object( blocked ) );
    //snippet end

    // VLE only for tracing?
    //snippet kmeansmindist
    auto min_distance = shared_ptr<object>( new mpi_object( blocked ) );
    //snippet end

    // masked coordinates is a Nx2k array with only 1 nonzero coordinate for each i<N
    //snippet kmeansmask
    auto k2blocked = shared_ptr<distribution>( make_shared<mpi_block_distribution>
						  (decomp,ncluster*dim,-1,globalsize) );
    auto masked_coordinates = shared_ptr<object>( new mpi_object( k2blocked ) );
    //snippet end
    
    // sum masked coordinates
    //snippet kmeansnewcenters
    auto klocal = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,ncluster*dim,1,-1) );
    auto partial_sums = shared_ptr<object>( new mpi_object( klocal ) );
    auto compute_new_centers1 =
      shared_ptr<kernel>( new mpi_kernel( masked_coordinates,partial_sums ) );
    //snippet end
    
    /*
     * set initial centers
     */
    switch (icase) {
    case 0 :
    case 1 :
    case 2 :
      dummy_centers_1d( centers,icase ); // not in the queue
      break;
    default: printf("No case %d implemented\n",icase); throw(5);
    }

    // test correct dummy centers
    CHECK( centers->volume(mycoord)==ncluster );
    auto centerdata = centers->get_data(mycoord);
    //    REQUIRE( centerdata!=nullptr );
    for (int icluster=0; icluster<ncluster; icluster++) {
      INFO( "checking cluster " << icluster );
      double c = centerdata.at(icluster);
      switch (icase) {
      case 0 :
      case 1 :
	CHECK( c == Approx( (double)(icluster+1)/(ncluster+1) ) );
	break;
      case 2 :
	CHECK(  ( (c>=0.) && (c<=1.) ) );
	break;
      }
    }

    /*
     * set initial coordinates
     */

    //snippet kmeansinitcoord
    auto set_random_coordinates = shared_ptr<kernel>( new mpi_kernel( coordinates ) );
    //snippet end

    set_random_coordinates->set_name("set random coordinates");
    switch (icase) {
    case 0 :
    case 1 :
    case 2 :
      set_random_coordinates->set_localexecutefn( &coordinates_on_center_1d );
      break;
    default: printf("No case %d implemented\n",icase); throw(5);
    }

    // check that the local coordinates are in place
    REQUIRE_NOTHROW( set_random_coordinates->analyze_dependencies() );
    REQUIRE_NOTHROW( set_random_coordinates->execute() );
    {
      INFO( "test initial coordinates" );
      auto centerdata = coordinates->get_data(mycoord);
      double c = centerdata.at(0); 
      switch (icase) {
      case 0 :
      case 1 :
    	CHECK( c == Approx( (double)(mytid+1)/(globalsize+1) ) );
    	break;
      case 2 :
	CHECK(  ( (c>=0.) && (c<=1.) ) );
	break;
      default: printf("No case %d implemented\n",icase); throw(5);
      }
    }
    return;

    /* 
     * distance calculation: outer product of 
     * my coordinates x all centers
     */
    REQUIRE( coordinates!=nullptr );
    REQUIRE( distances!=nullptr );
    REQUIRE( centers!=nullptr );
    shared_ptr<kernel> calculate_distances;
    switch (icase) {
    case 0 :
      //snippet kmeansdistcalc
      calculate_distances = shared_ptr<kernel>( new mpi_kernel( coordinates,distances ) );
      calculate_distances->set_last_dependency().set_explicit_beta_distribution(coordinates->get_distribution());
      calculate_distances->add_in_object( centers);
      calculate_distances->set_last_dependency().set_explicit_beta_distribution( centers->get_distribution() );
      calculate_distances->set_localexecutefn( &distance_calculation);
      //snippet end
      break ;
    case 1 :
    case 2 :
      //snippet kmeansdistkernel
      printf("the outer product kernel still uses the context. Wrong!\n");
      calculate_distances = shared_ptr<kernel>( new mpi_outerproduct_kernel
						      ( coordinates,distances,centers,&distance_calculation ) );
      //snippet end
      break;
    }
    calculate_distances->set_name("calculate distances");

    REQUIRE_NOTHROW( calculate_distances->analyze_dependencies() );
    { // check that this is a local operation
      vector<shared_ptr<task>> tasks;
      REQUIRE_NOTHROW( tasks = calculate_distances->get_tasks() );
      for (auto t : tasks ) {
	CHECK( t->get_receive_messages().size()==1 );
	CHECK( t->get_send_messages().size()==1 );
      }
    }

    // that gives a zero distance to the mytid'th center
    calculate_distances->execute(); // REQUIRE_NOTHROW( calculate_distances->execute() );

    {
      auto dist = distances->get_data(mycoord);
      switch (icase) {
      case 0 :
      case 1 :
	for (int icluster=0; icluster<ncluster; icluster++) {
	  INFO( "icluster=" << icluster );
	  if (icluster==mytid) {
	    CHECK( dist.at(icluster) == Approx( 0.0 ) );
	  } else {
	    CHECK( dist.at(icluster) != Approx( 0.0 ) );
	  }
	}
	break;
      case 2 :
	break; // initial coordinates are not in any relation to the centers
      default: printf("No case %d implemented\n",icase); throw(5);
      }
    }

    /*
     * set initial grouping (i'th point in i'th group) and compute distance
     */
    shared_ptr<kernel> find_nearest_center;
    switch (icase) {
    case 0:
      //snippet kmeansnearcalc
      find_nearest_center = shared_ptr<kernel>( new mpi_kernel( distances,grouping ) );
      find_nearest_center->set_localexecutefn( &group_calculation );
      find_nearest_center->set_explicit_beta_distribution( blocked );
      //find_nearest_center->set_last_dependency().set_type_local();
      //snippet end
      break;
    case 1 :
    case 2 :
      //snippet kmeansnearkernel
      find_nearest_center = shared_ptr<kernel>( new mpi_outerproduct_kernel
						      ( distances,grouping,NULL,group_calculation) );
      //snippet end
      break;
    }
    find_nearest_center->set_name("find nearest center");

    {
      auto group = grouping->get_data(mycoord);
      switch (icase) {
      case 0 : 
      case 1 :
      case 2 :
	group.at(0) = (double) mytid;
	break;
      default: printf("No case %d implemented\n",icase); throw(6);
      }
    }

    // compute new grouping
    REQUIRE_NOTHROW( find_nearest_center->execute() );
    {
      auto group = grouping->get_data(mycoord);
      switch (icase) {
      case 0 :
      case 1 :
	CHECK( group.at(0)== Approx( (double)mytid ) );
	break;
      case 2 :
	break; // no idea what the proper group is
      default: printf("No case %d implemented\n",icase); throw(6);
      }
    }

    /*
     * mask coordinates
     */
    shared_ptr<kernel> group_coordinates;
    switch (icase) {
    case 0 :
      //snippet kmeansgroupcalc
      group_coordinates = shared_ptr<kernel>( new mpi_kernel( coordinates,masked_coordinates ) );
      group_coordinates->set_localexecutefn( &coordinate_masking );
      group_coordinates->add_in_object(grouping);
      group_coordinates->set_last_dependency().set_explicit_beta_distribution(grouping->get_distribution());
      group_coordinates->set_last_dependency().set_type_local();
      //snippet end
      break;
    case 1 :
    case 2 :
      //snippet kmeansgroupkernel
      group_coordinates = shared_ptr<kernel>( new mpi_outerproduct_kernel
						    ( coordinates,masked_coordinates,grouping,coordinate_masking ) );
      //snippet end
      break;
    }
    group_coordinates->set_name("group coordinates");
    group_coordinates->execute();

    {
      auto group = masked_coordinates->get_data(mycoord);
      CHECK( masked_coordinates->volume(mycoord)==1 ); // hm
      CHECK( masked_coordinates->get_orthogonal_dimension()==ncluster );
      switch (icase) {
      case 0 :
      case 1 :
	for (int icluster=0; icluster<ncluster; icluster++) {
	  if (icluster!=mytid) {
	    CHECK( group.at(icluster) == Approx(0) );
	  } else {
	    CHECK( group.at(icluster) == Approx( (double)(mytid+1)/(globalsize+1) ) );
	  }
	}
	break;
      case 2 :
	break;
      default: printf("No case %d implemented\n",icase); throw(6);
      }
    }

    // locally sum the masked coordinates
    //snippet kmeansnewcenterscalc
    compute_new_centers1->set_name("partial sum calculation");
    compute_new_centers1->set_localexecutefn( &center_calculation_partial );
    compute_new_centers1->set_last_dependency().set_type_local();
    //snippet end
    CHECK_NOTHROW( compute_new_centers1->execute() );

    { 
      auto centerdata = coordinates->get_data(mycoord);
      auto newcenter = partial_sums->get_data(mycoord);
      switch (icase) {
      case 0 : // the partial sums should be equal to the old 
      case 1 :
	// for (int k=0; k<ncluster*dim; k++)
	//   CHECK( centerdata.at(k)==Approx(newcenter[k]) );
	break;
      case 2 :
	break;
      default : throw("Unimplemented\n"); 
      }
    }

    //delete centers;
    //delete kreplicated;
  } // end of case loop
};

TEST_CASE( "distance to centers","[10]" ) {
  index_int localsize, globalsize;
  int dim, ncluster = ntids;

  SECTION( "1d" ) { dim = 1; localsize = 1; }
  SECTION( "2d" ) { dim = 2; localsize = 1; }
  globalsize = localsize*ntids;
  INFO( "dim=" << dim << ", points per process=" << localsize );

  auto dblocked = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,dim,-1,globalsize) );
  auto coordinates = shared_ptr<object>( new mpi_object( dblocked ) );

  auto set_random_coordinates = shared_ptr<kernel>( new mpi_kernel( coordinates ) );
  set_random_coordinates->set_name("set random coordinates");
  set_random_coordinates->set_localexecutefn( &coordinates_on_center_1d );
  REQUIRE_NOTHROW( set_random_coordinates->analyze_dependencies() );
  REQUIRE_NOTHROW( set_random_coordinates->execute() );
  {
    data_pointer coordinate_data; REQUIRE_NOTHROW( coordinate_data = coordinates->get_data(mycoord) );
    double c = coordinate_data.at(0); 
    CHECK( c == Approx( (double)(mytid*localsize+1)/globalsize ) );
  }
  
  auto kreplicated = shared_ptr<distribution>( new mpi_replicated_distribution(decomp,dim,ncluster) );
  auto centers = shared_ptr<object>( new mpi_object( kreplicated ) );
  CHECK( centers->volume(mycoord)==ncluster );
  CHECK( centers->get_orthogonal_dimension()==dim );
  {
    REQUIRE_NOTHROW( centers->allocate() );
    data_pointer centerdata ; REQUIRE_NOTHROW( centerdata = centers->get_data(mycoord) );
    int iloc = 0;
    for (int tid=0; tid<ntids; tid++) {
      for (int idim=0; idim<dim; idim++)
	centerdata.at(iloc++) = (double)(tid*localsize+1)/globalsize;
    }
  }

  auto kblocked = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,ncluster,-1,globalsize) );
  auto distances = shared_ptr<object>( new mpi_object( kblocked ) );
  CHECK( distances->volume(mycoord)==localsize );
  CHECK( distances->get_orthogonal_dimension()==ncluster );

  auto calculate_distances = shared_ptr<kernel>( new mpi_kernel( coordinates,distances ) );
  calculate_distances->set_explicit_beta_distribution(coordinates->get_distribution());
  calculate_distances->add_in_object( centers);
  calculate_distances->set_explicit_beta_distribution( centers->get_distribution() );
  calculate_distances->set_localexecutefn( &distance_calculation);

  REQUIRE_NOTHROW( calculate_distances->analyze_dependencies() );
  REQUIRE_NOTHROW( calculate_distances->execute() );
  {
    data_pointer distance_data; REQUIRE_NOTHROW( distance_data = distances->get_data(mycoord) );
    for (int icluster=0; icluster<ncluster; icluster++) {
      double dist = distance_data.at(icluster); 
      INFO( "[" << mytid << "] cluster " << icluster << ", distance=" << dist );
      if (icluster==mytid)
	CHECK( dist==Approx(0.) );
      else
	CHECK( dist>=(.999/ntids) );
    }
  }
}

TEST_CASE( "find nearest center","[11]" ) {
  index_int localsize, globalsize;
  int dim, ncluster = ntids;

  SECTION( "1d" ) { dim = 1; localsize = 1; }
  SECTION( "2d" ) { dim = 2; localsize = 1; }
  globalsize = localsize*ntids;
  INFO( format("dim={}, points per process={}, ncluster={}",dim,localsize,ncluster ) );

  auto kblocked = shared_ptr<distribution>
    ( make_shared<mpi_block_distribution>(decomp,ncluster,-1,globalsize) );
  auto distances = shared_ptr<object>( new mpi_object( kblocked ) );
  CHECK( distances->volume(mycoord)==localsize );
  CHECK( distances->get_orthogonal_dimension()==ncluster );

  // set dummy distances
  {
    data_pointer distance_data;
    REQUIRE_NOTHROW( distance_data = distances->get_data(mycoord) );
    for (int icluster=0; icluster<ncluster; icluster++) {
      // distance to me is zero, everyone else a little more.
      if (icluster==mytid)
	distance_data.at(icluster) = 0.;
      else
	distance_data.at(icluster) = 1./ntids;
    }
  }

  auto blocked = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,-1,globalsize) );
  auto grouping = shared_ptr<object>( new mpi_object( blocked ) );
  shared_ptr<kernel> find_nearest_center;
  REQUIRE_NOTHROW
    ( find_nearest_center = shared_ptr<kernel>( new mpi_kernel( distances,grouping ) ) );
  REQUIRE_NOTHROW( find_nearest_center->set_last_dependency().set_explicit_beta_distribution( distances ) );
  find_nearest_center->set_name("find nearest center");
  REQUIRE_NOTHROW( find_nearest_center->set_localexecutefn( &group_calculation ) );
  //  REQUIRE_NOTHROW( find_nearest_center->set_explicit_beta_distribution( kblocked ) );
  REQUIRE_NOTHROW( find_nearest_center->analyze_dependencies() );
  shared_ptr<object> beta;
  REQUIRE_NOTHROW( beta = find_nearest_center->last_dependency().get_beta_object() );
  REQUIRE( beta->get_distribution()->get_orthogonal_dimension()==ncluster );
  REQUIRE_NOTHROW( find_nearest_center->execute() );

  {
    auto groups = grouping->get_data(mycoord);
    CHECK( grouping->volume(mycoord)==localsize );
    for (int i=0; i<localsize; i++) {
      CHECK( groups.at(i)==Approx(mytid) );
    }
  }
}

TEST_CASE( "group coordinates","[12]" ) {

  index_int localsize, globalsize;
  int dim, ncluster = ntids;

  SECTION( "1d" ) { dim = 1; localsize = 1; }
  SECTION( "2d" ) { dim = 2; localsize = 1; }
  globalsize = localsize*ntids;
  INFO( "dim=" << dim << ", points per process=" << localsize );

  algorithm *kmeans = new mpi_algorithm( decomp );
  kmeans->set_name(format("K-means clustering in {} dimensions",dim));

  auto dblocked = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,dim,-1,globalsize) );
  auto coordinates = shared_ptr<object>( new mpi_object( dblocked ) );

  auto set_random_coordinates = shared_ptr<kernel>( new mpi_origin_kernel( coordinates ) );
  set_random_coordinates->set_name("set random coordinates");
  set_random_coordinates->set_localexecutefn( &coordinates_on_center_1d );
  REQUIRE_NOTHROW( kmeans->add_kernel( set_random_coordinates ) );

  // make grouping array
  auto blocked = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,-1,globalsize) );
  auto grouping = shared_ptr<object>( new mpi_object( blocked ) );
  REQUIRE_NOTHROW( grouping->allocate() );
  {
    auto groups = grouping->get_data(mycoord);
    CHECK( grouping->volume(mycoord)==localsize );
    for (int i=0; i<localsize; i++) {
      REQUIRE_NOTHROW( groups.at(i)=(double)mytid );
    }
  }
  REQUIRE_NOTHROW( kmeans->add_kernel( shared_ptr<kernel>( new mpi_origin_kernel(grouping) ) ) );

  // masked coordinates is a N-by-kx(dim+1) array
  // where "dim+1" stands for "flag plus coordinate":
  // a negative flag is not reduced over
  auto kdblocked = shared_ptr<distribution>( make_shared<mpi_block_distribution>
						(decomp,ncluster*(dim+1),-1,globalsize) );
  shared_ptr<object> masked_coordinates;
  REQUIRE_NOTHROW( masked_coordinates= shared_ptr<object>( new mpi_object( kdblocked ) ) );
  shared_ptr<kernel> group_coordinates;
  REQUIRE_NOTHROW( group_coordinates = shared_ptr<kernel>( new mpi_kernel( coordinates,masked_coordinates ) ) );
  group_coordinates->add_sigma_operator( ioperator("no_op")  );
  group_coordinates->set_name("group coordinates");
  group_coordinates->set_localexecutefn( &coordinate_masking );
  REQUIRE_NOTHROW( group_coordinates->add_in_object(grouping) );
  REQUIRE_NOTHROW( group_coordinates->set_explicit_beta_distribution(grouping->get_distribution()) );
  kmeans->add_kernel( group_coordinates );

  REQUIRE_NOTHROW( kmeans->analyze_dependencies() );
  REQUIRE_NOTHROW( kmeans->execute() );

  {
    data_pointer coord_data, mask_data;
    REQUIRE_NOTHROW( coord_data = coordinates->get_data(mycoord) );
    REQUIRE_NOTHROW( mask_data = masked_coordinates->get_data(mycoord) );
    CHECK( masked_coordinates->volume(mycoord)==localsize );
    CHECK( masked_coordinates->get_orthogonal_dimension()==ncluster*(dim+1) );
    INFO( "inspecting mask on proc " << mytid );
    for (int ic=0; ic<ncluster; ic++) {
      INFO( "cluster " << ic );
      // first the mask parameter
      if (ic==mytid) {
	CHECK( mask_data.at(ic*(dim+1))==Approx(1.) );
      } else {
	CHECK( mask_data.at(ic*(dim+1))==Approx(-1.) );
      }
      // then the coordinates; duplicated for each cluster.
      for (int id=0; id<dim; id++)
	CHECK( coord_data.at(id)==Approx(mask_data.at(ic*(dim+1)+id+1)) );
    }
  }
}

TEST_CASE( "compute new centers","[13]" ) {

  int dim, ncluster = ntids, localsize = 1, globalsize = ntids*localsize;
  SECTION( "1D" ) { dim = 1; }
  INFO( "dim=" << dim );

  // masked coordinates is a N-by-kx(dim+1) array
  // where "dim+1" stands for "flag plus coordinate":
  // a negative flag is not reduced over
  auto kdblocked = shared_ptr<distribution>( make_shared<mpi_block_distribution>
						(decomp,ncluster*(dim+1),localsize,-1) );
  shared_ptr<object> masked_coordinates;
  REQUIRE_NOTHROW( masked_coordinates= shared_ptr<object>( new mpi_object( kdblocked ) ) );
  masked_coordinates->allocate();
  {
    data_pointer mask_data;
    REQUIRE_NOTHROW( mask_data = masked_coordinates->get_data(mycoord) );
    CHECK( masked_coordinates->volume(mycoord)==localsize );
    index_int k = ncluster*(dim+1);
    CHECK( masked_coordinates->get_orthogonal_dimension()==k );
    INFO( "inspecting mask on proc " << mytid );
    for (int ilocal=0; ilocal<localsize; ilocal++) {
      // VLE uh oh. too much editing. Is this mask_data correct?
      //auto point = mask_data->begin(); point.advance( ilocal*k );
      auto point_data = mask_data.data()+ilocal*k;
      // for each point we have a coordinate+mask per cluster
      // the coordinates are all the same ?! check with [12]
      // that's fine for low numbers of clusters.
      for (int ic=0; ic<ncluster; ic++) {
	INFO( "cluster " << ic );
	for (int id=0; id<dim; id++) {
	  //point.at( ic*(dim+1)+id ) = 1./(mytid+1); // ???? 1./(ic+1);
	  point_data[ic*(dim+1)+id] = 1./(mytid+1);
	}
	// if the cluster corresponds to this processor: select
	if (ic==mytid) {
	  //point.at( ic*(dim+1)+dim ) = 1.;
	  point_data[ic*(dim+1)+dim] = 1.;
	} else {
	  //point.at( ic*(dim+1)+dim ) = -1.;
	  point_data[ic*(dim+1)+dim] = -1.;
	}
      }
    }
  }
  auto make_masked = shared_ptr<kernel>( new mpi_origin_kernel(masked_coordinates) );
  make_masked->analyze_dependencies();
  make_masked->execute();

  /*
   * Reduct!
   */
  auto kreplicated = shared_ptr<distribution>( new mpi_replicated_distribution(decomp,dim,ncluster) );
  auto centers = shared_ptr<object>( new mpi_object( kreplicated ) );

  auto new_centers = shared_ptr<kernel>( new mpi_kernel(masked_coordinates,centers) );
  if (dim==1) {
    new_centers->set_localexecutefn( &masked_reduction_1d );
  } else throw(string("masked reduction only in 1d"));
  new_centers->set_explicit_beta_distribution(kdblocked);
  REQUIRE_NOTHROW( new_centers->analyze_dependencies() );
  REQUIRE_NOTHROW( new_centers->execute() );

  {
    data_pointer center_data;
    REQUIRE_NOTHROW( center_data = centers->get_data(mycoord) );
    CHECK( centers->volume(mycoord)==ncluster );
    CHECK( centers->get_orthogonal_dimension()==dim );
    for (int ic=0; ic<ncluster; ic++) {
      INFO( "cluster " << ic );
      for (int id=0; id<dim; id++) {
	center_data.at( ic*dim + id ) = 1./(mytid+1);
      }
    }
  }
}

TEST_CASE( "distance to centers, general","[20]" ) {
  index_int localsize, globalsize;
  int dim, clusters_per_proc = 2, ncluster = clusters_per_proc * ntids;

  SECTION( "1d" ) { dim = 1; localsize = 5 * clusters_per_proc; }
  SECTION( "2d" ) { dim = 2; localsize = 5 * clusters_per_proc; }
  globalsize = localsize*ntids;
  INFO( format
	("[{}] dim={}, clusters per processor={} for a total of: {}; points per process={}",
	 mycoord.as_string(),dim,clusters_per_proc,ncluster,localsize) );
  auto dblocked = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,dim,localsize,-1) );
  auto coordinates = shared_ptr<object>( new mpi_object( dblocked ) );
  CHECK( coordinates->global_volume()==globalsize );

  /*
   * Set initial coordinates to the point indices, in every dimension:
   * (f,f,f) (f+1,f+1,f+1) ....
   */
  auto set_linear_coordinates = shared_ptr<kernel>( new mpi_kernel( coordinates ) );
  set_linear_coordinates->set_name("set random coordinates");
  set_linear_coordinates->set_localexecutefn( &coordinates_linear );
  REQUIRE_NOTHROW( set_linear_coordinates->analyze_dependencies() );
  REQUIRE_NOTHROW( set_linear_coordinates->execute() );
  {
    data_pointer coordinate_data;
    REQUIRE_NOTHROW( coordinate_data = coordinates->get_data(mycoord) );
    index_int linear_first;
    REQUIRE_NOTHROW
      ( linear_first = coordinates->first_index_r(mycoord)
	.linear_location_in( coordinates->get_enclosing_structure() ) );
    int d = 0;
    for (int c=0; c<localsize; c++) {
      for (int id=0; id<dim; id++) {
	INFO( format("local coordinate: {}, dim: {}, linear: {}",c,id,d) );
	CHECK( coordinate_data.at(d++) == Approx( (double)linear_first +c  ) );
      }
    }
  }

  // nclusters is the #centers, each of size dim, and replicated
  auto kreplicated = shared_ptr<distribution>( new mpi_replicated_distribution(decomp,dim,ncluster) );
  auto centers = shared_ptr<object>( new mpi_object( kreplicated ) );
  CHECK( centers->volume(mycoord)==ncluster );
  CHECK( centers->get_orthogonal_dimension()==dim );
  // equally spaced cluster centers over the points
  memory_buffer w; format_to(w,"Cluster centers:");
  {
    REQUIRE_NOTHROW( centers->allocate() );
    data_pointer centerdata ; REQUIRE_NOTHROW( centerdata = centers->get_data(mycoord) );
    int iloc = 0;
    for (int icluster=0; icluster<ncluster; icluster++) {
      double cluster_center = icluster * globalsize / (double)ncluster;
      format_to(w," {},",cluster_center);
      for (int idim=0; idim<dim; idim++)
	centerdata.at(iloc++) = cluster_center;
    }
  }
  INFO( to_string(w) );

  // for each point the distance to the cluster centers
  auto kblocked = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,ncluster,-1,globalsize) );
  auto distances = shared_ptr<object>( new mpi_object( kblocked ) );
  CHECK( distances->volume(mycoord)==localsize );
  CHECK( distances->get_orthogonal_dimension()==ncluster );

  auto calculate_distances = shared_ptr<kernel>( new mpi_kernel( coordinates,distances ) );
  calculate_distances->set_last_dependency().set_explicit_beta_distribution(coordinates->get_distribution());
  calculate_distances->add_in_object( centers);
  calculate_distances->set_last_dependency().set_explicit_beta_distribution( centers->get_distribution() );
  calculate_distances->set_localexecutefn( &distance_calculation);

  REQUIRE_NOTHROW( calculate_distances->analyze_dependencies() );
  REQUIRE_NOTHROW( calculate_distances->execute() );

  {
    data_pointer distance_data;
    REQUIRE_NOTHROW( distance_data = distances->get_data(mycoord) );
    for (index_int ipoint=0; ipoint<localsize; ipoint++) {
      for (int icluster=0; icluster<ncluster; icluster++) {
	double
	  point_loc = ipoint+mytid*localsize,
	  cluster_loc = icluster*globalsize/(double)ncluster,
	  step_dist = std::abs( point_loc - cluster_loc );
	INFO( format("point {} at {} to cluster {} at {} has step dist {}",
			  ipoint,point_loc,icluster,cluster_loc,step_dist));
	double dist = 0;
	for (int id=0; id<dim; id++)
	  dist += step_dist*step_dist;
	dist = sqrt(dist);
	double d = distance_data.at(INDEXpointclusterdist(ipoint,icluster,localsize,ncluster));
	CHECK( d==Approx(dist) );
      }
    }
  }
}

TEST_CASE( "Find nearest center, general","[30]" ) {
  index_int localsize, globalsize;
  int dim, clusters_per_proc = 2, ncluster = clusters_per_proc * ntids;

  SECTION( "1d" ) { dim = 1; localsize = 5 * clusters_per_proc; }
  SECTION( "2d" ) { dim = 2; localsize = 5 * clusters_per_proc; }
  globalsize = localsize*ntids;
  INFO( format
	("[{}] dim={}, clusters per processor={} for a total of {} clusters; points per process={}",
	 mycoord.as_string(),dim,clusters_per_proc,ncluster,localsize) );

  // for each point the distance to the cluster centers
  auto kblocked = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,ncluster,-1,globalsize) );
  auto distances = shared_ptr<object>( new mpi_object( kblocked ) );

  { // copied from above
    data_pointer distance_data;
    REQUIRE_NOTHROW( distance_data = distances->get_data(mycoord) );
    for (index_int ipoint=0; ipoint<localsize; ipoint++) {
      memory_buffer w;
      index_int gpoint = ipoint+mytid*localsize;
      double point_loc = gpoint;
      format_to(w,"point g={}, cluster dists:",gpoint);
      for (int icluster=0; icluster<ncluster; icluster++) {
	double
	  cluster_loc = icluster*globalsize/(double)ncluster,
	  step_dist = std::abs( point_loc - cluster_loc );
	format_to(w," {}:{}->{}",icluster,cluster_loc,step_dist);
	double dist = 0;
	for (int id=0; id<dim; id++)
	  dist += step_dist*step_dist;
	dist = sqrt(dist);
	REQUIRE_NOTHROW
	  ( distance_data.at(INDEXpointclusterdist(ipoint,icluster,localsize,ncluster)) = dist );
      }
      fmt::print("{}\n",to_string(w));
    }
  }

  // make grouping array
  auto blocked = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,-1,globalsize) );
  auto grouping = shared_ptr<object>( new mpi_object( blocked ) );
  REQUIRE_NOTHROW( grouping->allocate() );

  auto find_nearest_center = shared_ptr<kernel>( new mpi_kernel( distances,grouping ) );
  find_nearest_center->set_name("find nearest center");
  find_nearest_center->set_localexecutefn( &group_calculation );
  find_nearest_center->set_explicit_beta_distribution( blocked );

  find_nearest_center->analyze_dependencies();
  find_nearest_center->execute();

  {
    data_pointer group_data;
    REQUIRE_NOTHROW( group_data = grouping->get_data(mycoord) );
    CHECK( grouping->volume(mycoord)==localsize );
    CHECK( group_data.size()==localsize );
    for (index_int i=0; i<localsize; i++) {
      INFO( format("Proc {}, local point {}, global {}",
			mycoord.as_string(),i,i+mytid*localsize) );
      REQUIRE_NOTHROW( group_data.at(i)>=0 );
      CHECK( group_data.at(i)>=0 );
      CHECK( group_data.at(i)<ncluster );
      CHECK( group_data.at(i)>=mytid*clusters_per_proc );
    }
  }
}

TEST_CASE( "group coordinates, general","[40]" ) {
  index_int localsize, globalsize;
  int dim, clusters_per_proc = 2, ncluster = clusters_per_proc * ntids;


  SECTION( "1d" ) { dim = 1; }
  SECTION( "2d" ) { dim = 2; }

  localsize = 5 * clusters_per_proc;
  globalsize = localsize*ntids;
  INFO( format("{} dim={} global size {}",mycoord.as_string(),dim,globalsize) );

  shared_ptr<distribution> blocked;
  REQUIRE_NOTHROW( blocked = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,-1,globalsize) ) );
  shared_ptr<object> grouping;
  REQUIRE_NOTHROW( grouping = shared_ptr<object>( new mpi_object( blocked ) ) );

  shared_ptr<distribution> twoblocked;
  REQUIRE_NOTHROW( twoblocked = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,dim,-1,globalsize) ) );
  shared_ptr<object> coordinates;
  REQUIRE_NOTHROW( coordinates = shared_ptr<object>( new mpi_object( twoblocked ) ) );
  CHECK( coordinates->get_orthogonal_dimension()==dim );

  domain_coordinate myfirst;
  REQUIRE_NOTHROW( myfirst = blocked->get_processor_structure(mycoord)->first_index_r() );
  CHECK( myfirst.get_dimensionality()==1 );

  /*
   * Set initial coordinates to the point indices, in every dimension:
   * (f,f,f) (f+1,f+1,f+1) ....
   * as in [20] above
   */
  auto set_linear_coordinates = shared_ptr<kernel>( new mpi_kernel( coordinates ) );
  set_linear_coordinates->set_name("set random coordinates");
  set_linear_coordinates->set_localexecutefn( &coordinates_linear );
  REQUIRE_NOTHROW( set_linear_coordinates->analyze_dependencies() );
  REQUIRE_NOTHROW( set_linear_coordinates->execute() );
  { // check
    auto coordinate_data = coordinates->get_data(mycoord);
    for (index_int i=0; i<localsize; i++) {
      for (int id=0; id<dim; id++) {
	CHECK( coordinate_data.at( i*dim+id )==Approx(myfirst[0]+i) );
      }
    }
  }

  // masked coordinates are ncluster times orthogonal version of block
  // but we need an extra location to indicate inclusion
  // so that we can implement a masked MPI reduce.....
  shared_ptr<distribution> kdblocked;
  REQUIRE_NOTHROW
    ( kdblocked = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,ncluster*(dim+1),-1,globalsize) ) );
  shared_ptr<object> masked_coordinates;
  REQUIRE_NOTHROW( masked_coordinates = shared_ptr<object>( new mpi_object( kdblocked ) ) );

  { // set the group of each data point to the first local cluster
    // sort of as in [30] above
    data_pointer group_data;
    REQUIRE_NOTHROW( group_data = grouping->get_data(mycoord) );
    CHECK( grouping->volume(mycoord)==localsize );
    for (index_int i=0; i<localsize; i++) {
      group_data.at(i) = mytid*clusters_per_proc;
    }
  }
  
  auto group_coordinates =
    shared_ptr<kernel>( new mpi_kernel( coordinates,masked_coordinates ) );

  group_coordinates->set_name("group coordinates");
  group_coordinates->set_localexecutefn( &coordinate_masking );
  group_coordinates->add_sigma_operator( ioperator("no_op")  );
  group_coordinates->add_in_object(grouping);
  group_coordinates->set_explicit_beta_distribution(grouping->get_distribution());

  REQUIRE_NOTHROW( group_coordinates->analyze_dependencies() );
  REQUIRE_NOTHROW( group_coordinates->execute() );

  {
    data_pointer masked_data;
    REQUIRE_NOTHROW( masked_data = masked_coordinates->get_data(mycoord) );
    CHECK( masked_coordinates->volume(mycoord)==localsize );
    CHECK( masked_coordinates->get_orthogonal_dimension()==ncluster*(dim+1) );

    data_pointer group_data;
    REQUIRE_NOTHROW( group_data = grouping->get_data(mycoord) );

    data_pointer coordinate_data;
    REQUIRE_NOTHROW( coordinate_data = coordinates->get_data(mycoord) );

    for (index_int i=0; i<localsize; i++) {
      auto masked_coordinate = masked_data.begin()+i*ncluster*(dim+1);

      // what group does this point belong to?
      int group = (int)( group_data.at(i) );
      INFO( format("local point {} is in group {}",i,group) );

      // compare the point to all clusters
      for (int ic=0; ic<ncluster; ic++) {
	INFO( format("cluster {} out of {}",ic,ncluster) );
	int maskv = masked_coordinate[ ic*(dim+1) ];
	// go through all the dimensions of a point
	if (ic==group)
	  CHECK( maskv==Approx(+1) );
	else
	  CHECK( maskv==Approx(-1) );
	for (int id=0; id<dim; id++) {
	  auto
	    mxi = masked_coordinate[ id+1 ], // first position is mask, so id+1
	    xi = coordinate_data.at( i*dim+id );
	  INFO( format
	  	("point belongs to cluster: @idim={} value={}, should be f+i={}+{}={}",
	  	 id,mxi,
		 myfirst[0],i,xi) );
	  // if the point belongs to this cluster: nonzero
	  CHECK( mxi==Approx(xi) );
	}
	// test the mask value?
      }
    }
  }
}

// defined in mpi_kmeans_kernel.cxx
void add_if_mask1( void *indata, void * outdata,int *len,MPI_Datatype *type );

TEST_CASE( "custom MPI reduction, one point per proc","[50]" ) {

  double coordinate_and_mask[2], result_and_mask[2], result;
  coordinate_and_mask[0] = -1;
  coordinate_and_mask[1] = mytid;

  SECTION( "pick first point" ) { result = 0;
    if (mytid==result)
      coordinate_and_mask[0] = +1;
  }

  SECTION( "pick last point" ) { result = ntids-1;
    if (mytid==result)
      coordinate_and_mask[0] = +1;
  }

  SECTION( "pick second and last point" ) { result = ntids;
    if (ntids<3) {
      printf("[50] needs 3 procs for one test\n"); return; }
    if (mytid==1 || mytid==ntids-1)
      coordinate_and_mask[0] = +1;
  }

  MPI_Datatype dim1_type;
  MPI_Type_contiguous(2,MPI_DOUBLE,&dim1_type);
  MPI_Type_commit(&dim1_type);

  MPI_Op masked_add;
  MPI_Op_create(add_if_mask1,1,&masked_add);

  MPI_Allreduce
    (coordinate_and_mask,result_and_mask,
     1,dim1_type,masked_add,MPI_COMM_WORLD);

  CHECK( result_and_mask[0]==Approx(1) );
  CHECK( result_and_mask[1]==Approx(result) );
  
  MPI_Type_free(&dim1_type);
  MPI_Op_free(&masked_add);
}

TEST_CASE( "reduce to new center","[51]" ) {

  index_int localsize, globalsize;
  int dim, clusters_per_proc = 2, ncluster = clusters_per_proc * ntids;


  SECTION( "1d" ) { dim = 1; }
  SECTION( "2d" ) { dim = 2; }

  localsize = 5 * clusters_per_proc;
  globalsize = localsize*ntids;
  INFO( format("{} dim={} global size {}",mycoord.as_string(),dim,globalsize) );

  shared_ptr<distribution> blocked;
  REQUIRE_NOTHROW( blocked = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,-1,globalsize) ) );
  shared_ptr<object> grouping;
  REQUIRE_NOTHROW( grouping = shared_ptr<object>( new mpi_object( blocked ) ) );

  shared_ptr<distribution> twoblocked;
  REQUIRE_NOTHROW( twoblocked = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,dim,-1,globalsize) ) );
  shared_ptr<object> coordinates;
  REQUIRE_NOTHROW( coordinates = shared_ptr<object>( new mpi_object( twoblocked ) ) );
  CHECK( coordinates->get_orthogonal_dimension()==dim );

  // masked coordinates are ncluster times orthogonal version of block
  // but we need an extra location to indicate inclusion
  // so that we can implement a masked MPI reduce.....
  shared_ptr<distribution> kdblocked;
  REQUIRE_NOTHROW
    ( kdblocked = shared_ptr<distribution>( make_shared<mpi_block_distribution>(decomp,ncluster*(dim+1),-1,globalsize) ) );
  shared_ptr<object> masked_coordinates;
  REQUIRE_NOTHROW( masked_coordinates = shared_ptr<object>( new mpi_object( kdblocked ) ) );

  domain_coordinate myfirst;
  REQUIRE_NOTHROW( myfirst = blocked->get_processor_structure(mycoord)->first_index_r() );
  CHECK( myfirst.get_dimensionality()==1 );

  /*
   * set masked_data as it was at the end of [40]
   */
  data_pointer masked_data;
  REQUIRE_NOTHROW( masked_data = masked_coordinates->get_data(mycoord) );
  CHECK( masked_coordinates->volume(mycoord)==localsize );
  CHECK( masked_coordinates->get_orthogonal_dimension()==ncluster*(dim+1) );

  data_pointer group_data;
  REQUIRE_NOTHROW( group_data = grouping->get_data(mycoord) );

  data_pointer coordinate_data;
  REQUIRE_NOTHROW( coordinate_data = coordinates->get_data(mycoord) );

  for (index_int i=0; i<localsize; i++) {
    auto masked_coordinate = masked_data.begin()+i*ncluster*(dim+1);

    // what group does this point belong to?
    int group = (int)( group_data.at(i) );
    INFO( format("local point {} is in group {}",i,group) );

    // compare the point to all clusters
    for (int ic=0; ic<ncluster; ic++) {
      INFO( format("cluster {} out of {}",ic,ncluster) );
      // go through all the dimensions of a point
      if (ic==group)
	masked_coordinate[ ic*(dim+1) ] = +1;
      else
	masked_coordinate[ ic*(dim+1) ] = -1;
      for (int id=0; id<dim; id++) {
	masked_coordinate[ id+1 ] = coordinate_data.at( i*dim+id );
      }
    }
  }

  /*
   * Reduce based on mask
   */
  auto kreplicated = shared_ptr<distribution>( new mpi_replicated_distribution(decomp,dim,ncluster) );
  auto new_centers = shared_ptr<object>( new mpi_object( kreplicated ) );
  auto reduce_with_mask =
    shared_ptr<kernel>( new mpi_kernel(masked_coordinates,new_centers) );
  REQUIRE_NOTHROW( reduce_with_mask->set_explicit_beta_distribution(masked_coordinates->get_distribution()) );
  REQUIRE_NOTHROW( reduce_with_mask->set_localexecutefn(&masked_reduction_1d) );
  REQUIRE_NOTHROW( reduce_with_mask->analyze_dependencies() );
  REQUIRE_NOTHROW( reduce_with_mask->execute() );

}
