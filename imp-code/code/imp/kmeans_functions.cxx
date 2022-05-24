/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-9
 ****
 **** kmeans_functions.cxx : implementations of the kmeans support functions
 ****
 ****************************************************************/

#include <stdlib.h>
#include <stdio.h>

#include "imp_base.h"
#include "kmeans_functions.h"

#include "utils.h"

using std::shared_ptr;
using std::vector;

using fmt::format;
using fmt::print;
using fmt::memory_buffer;

/****
 **** Replicated initial centers
 ****/
void set_initial_centers( std::shared_ptr<object> mpi_centers,processor_coordinate &p ) {
  int k = mpi_centers->get_distribution()->get_orthogonal_dimension();
  data_pointer center_data;
  try { center_data = mpi_centers->get_data(p);
  } catch (int x) {printf("could not get data in initial centers\n"); throw(1); }
  for (int ik=0; ik<k; ik++) {
    center_data.at(2*ik) = ( (double)ik )/k;
    center_data.at(2*ik+1) = ( (double)ik )/k;
  }
}

/****
 **** Kernel for initial data generation
 ****/
void generate_random_coordinates( kernel_function_args ) {
  int dim = p.get_same_dimensionality(outvector->get_dimensionality());

  const auto outdistro = outvector->get_distribution();

#include "impfunc_struct_index.cxx"

  // initialize random
  //srand(p.coord(0));

  for (index_int i=pfirst[0]; i<plast[0]; i++) {
    for (int id=0; id<dim; id++) {
      outdata.at(dim*INDEX1D(i,offsets,nsize)+id) = rand() / (double)RAND_MAX;
    }
  }
};

/****
 **** Kernels for update step
 ****/
void distance_to_center( kernel_function_args,void *ctx ) {
  auto coordinates = invectors.at(0);

  two_object_struct *two_objects = (two_object_struct*) ctx;
  auto
    grouping = invectors.at(1), centers = invectors.at(2);
  //two_objects->one, *centers = two_objects->two;
  index_int
    dim = coordinates->get_distribution()->get_orthogonal_dimension(),
    nclusters = centers->get_distribution()->get_orthogonal_dimension(),
    my_cluster;
  auto ownership = grouping->get_data(p);
  my_cluster = static_cast<int>( ownership.at(0) );
  if (my_cluster<0 || my_cluster>=nclusters) 
    throw(format("Invalid cluster number found: {}",my_cluster));
}

/*!
  The invector[0] is coordinates, ortho=#space dimensions.
  The invector[1] is clusters, local size=#clusters, ortho=#space dimen.
  The outvector has distances to all cluster centers: 
  - same local size as invector[0],
  - ortho = ncluster
 */
void distance_calculation( kernel_function_args ) {
  if (invectors.size()<2)
    throw(std::string("Missing centers inobject"));
  auto invector = invectors.at(0);
  int dimension = invector->get_distribution()->get_orthogonal_dimension();
  auto coordinates = invector->get_data(p); 
  int ncoordinates = invector->get_distribution()->volume(p);
  if (ncoordinates!=outvector->get_distribution()->volume(p))
    throw(format("Distance calculatin in/out need to be compatible"));

  auto distances = outvector->get_data(p);
  // {
  //   index_int first=outvector->get_distribution()->first_index(p).coord(0),last=outvector->get_distribution()->last_index(p).coord(0),
  //     localsize = last-first+1;
  //   if (localsize!=ncoordinates)
  //     throw(format("distance localsize {} != coord local size {}",localsize,ncoordinates));
  // }

  auto centers = invectors.at(1);
  auto center_coordinates = centers->get_data(p);
  index_int nclusters = centers->get_distribution()->volume(p);
  if (nclusters!=outvector->get_distribution()->get_orthogonal_dimension())
    throw(format("ncluster={} is not distance ortho={}",
		      nclusters,outvector->get_distribution()->get_orthogonal_dimension()));
  
  for (int ipoint=0; ipoint<ncoordinates; ipoint++) {
    for (int ikluster=0; ikluster<nclusters; ikluster++) {
      double dist = 0;
      for (int id=0; id<dimension; id++) {
	double
	  pp = coordinates.at( ipoint*dimension+id ),
	  cc = center_coordinates.at( ikluster*dimension+id ),
	  dd = pp-cc;
	dist += dd*dd;
	// printf("[%d] dimension %d: point %d @ %e, cluster %d %e\n",
	//        p.coord(0),id, ipoint,pp, ikluster,cc);
      }
      dist = sqrt(dist);
      index_int array_loc = ipoint*nclusters+ikluster;
      distances.at( array_loc ) = dist;
      // print
      // 	("{} point {} at {}, cluster {} at {}, distance computed={} at array loc {}\n",
      // 	 p.as_string(),ipoint,pp,ikluster,cc,dist,array_loc);
    }
  }

  return;
};

void group_calculation( kernel_function_args ) {
  auto invector = invectors.at(0);
  data_pointer distances;
  try { distances = invector->get_data(p); 
  } catch (int x) {printf("Could not get distance coordinate data\n"); throw(1); }
  index_int ncluster = invector->get_distribution()->get_orthogonal_dimension();

  auto groups = outvector->get_data(p);
  index_int
    first = outvector->get_distribution()->first_index_r(p).coord(0),
    last  = outvector->get_distribution()->last_index_r(p).coord(0),
    localsize = last-first+1;
  
  // print("[{}] calculating groups {}--{} for {} clusters\n",
  // 	     p.coord(0),last,first,ncluster);
  for (int i=0; i<localsize; i++) {
    int kmin = 0;
    double mindist = distances.at( INDEXpointclusterdist(i,kmin,localsize,ncluster) );
    for (int ik=1; ik<ncluster; ik++) {
      double otherdist = distances.at( INDEXpointclusterdist(i,ik,localsize,ncluster) );
      //      printf("i=%d, at ik=%d compare %f / %f\n",i,ik,otherdist,mindist);
      if (otherdist<mindist) {
	kmin = ik; mindist = otherdist;
      }
    }
    groups.at(i) = (double)kmin;
  }

  return;
}

/*!
  Replicate the coordinates for each cluster, with a +/- 1 mask to indicate membership
  \todo pass in the number of clusters as parameter
 */
void coordinate_masking( kernel_function_args ) {
  // invec: [ coord object, grouping ], outvector: masked object
  auto invector = invectors.at(0),
    grouping = invectors.at(1); // groups: ncluster integers, replicated
  int dim = invector->get_distribution()->get_orthogonal_dimension();
  data_pointer coordinates, group, selected;
  try {
    coordinates = invector->get_data(p);
    group = grouping->get_data(p);
    selected = outvector->get_data(p);
  } catch (...) {
    throw(format("Could not get data on {}\n",p.as_string()));
  }

  //  int ncluster = grouping->get_distribution()->get_orthogonal_dimension();

  index_int first=outvector->get_distribution()->first_index_r(p).coord(0),
    last=outvector->get_distribution()->last_index_r(p).coord(0),
    localsize = last-first+1;
  index_int kd = outvector->get_distribution()->get_orthogonal_dimension();
  int ncluster = kd/(dim+1);

  memory_buffer w;
  for (index_int ipoint=0; ipoint<localsize; ipoint++) {
    double
      *coordinate = coordinates.data() + ipoint*dim,
      *masked = selected.data() + ipoint*ncluster*(dim+1);
    int igroup = (int)(group.at(ipoint)); // what is the group of this coordinate?
    int outp = 0; // positioning pointer in the output
    format_to(w,"{} local {} belongs to {}; ",p.as_string(),ipoint,igroup);
    for (int icluster=0; icluster<ncluster; icluster++) {
      // set mask location to +1 or -1
      format_to(w,"cluster {} : ",icluster);
      if (icluster==igroup) {
	format_to(w,"yes, ");
	masked[ outp++ ] = +1.;
      } else {
	format_to(w,"no, ");
	masked[ outp++ ] = -1;
      }
      // copy coordinate data
      for (int idim=0; idim<dim; idim++)
	masked[ outp++ ] = coordinate[ idim ];
    }
    if (ipoint==0)
      print( "{}\n",to_string(w) ); 
  }
}

// compute locally the partial sums
void center_calculation_partial( kernel_function_args ) {
  auto invector = invectors.at(0);
  //  invec: masked_coordinates, outvector: partial_sums
  data_pointer select_coordinates; // distributed 2k x N
  try { select_coordinates = invector->get_data(p); 
  } catch (int x) {printf("Could not get selected coordinate data\n");}
  index_int
    localsize = invector->get_distribution()->volume(p),
    k2 = invector->get_distribution()->get_orthogonal_dimension(); // #groups * dimension

  if (outvector->get_distribution()->volume(p)!=1)
    throw(format("partial sums should be 1-distributed, not {}",
		      outvector->get_distribution()->local_size_r(p).as_string()));
	  
  {
    index_int k2check = outvector->get_distribution()->get_orthogonal_dimension();
    if (k2check!=k2)
      throw(format("outvector k s/b {}, not {}",k2,k2check));
  }

  data_pointer centers_partial;
  try { centers_partial = outvector->get_data(p);
  } catch (int x) {printf("Could not get centers partial data\n");}
  index_int kcheck = outvector->get_distribution()->get_orthogonal_dimension();
  if (kcheck!=k2)
    throw(format("partial sum ortho is {}, not {}",kcheck,k2));
  
  for (index_int ik=0; ik<k2; ik++) {
    double s=0;
    for (index_int i=0; i<localsize; i++)
      s += select_coordinates.at( ik+i*k2 );
    centers_partial.at(ik) = s;
  }

  return;
}

/****
 **** Local norm calculation
 ****/
void local_norm_function( kernel_function_args,void *ctx )
{
  auto invector = invectors.at(0);
  index_int
    first=invector->get_distribution()->first_index_r(p).coord(0),
    last=invector->get_distribution()->last_index_r(p).coord(0);
  auto outdata = outvector->get_data(p);
  auto indata = invector->get_data(p);

  kmeans_vec_sum(outdata,indata,first,last);

};

void kmeans_gen_local(data_pointer outdata,index_int first,index_int last) {

  for (index_int i=first; i<last; i++) {
    outdata.at(i-first) = (double)i;
  }
}

void avg_local
    (data_pointer indata,data_pointer outdata,
     index_int first,index_int last,double *nops) {
  int leftshift=1;
  // initialization
  for (index_int i=0; i<last-first; i++)
    outdata.at(i) = 0.;
  // shift 0
  for (index_int i=first; i<last; i++) {
    index_int i_out=i-first,i_in=i_out+leftshift;
    outdata.at(i_out) += indata.at(i_in);
  }
  // shift to the right
  for (index_int i=first; i<last; i++) {
    index_int i_out=i-first,i_in=i_out+leftshift-1;
    outdata.at(i_out) += indata.at(i_in);
  }
  // shift to the left
  for (index_int i=first; i<last; i++) {
    index_int i_out=i-first,i_in=i_out+leftshift+1;
    outdata.at(i_out) += indata.at(i_in);
  }
  *nops = 3.*(last-first);
  return;
}

void kmeans_vec_sum(data_pointer outdata,data_pointer indata,index_int first,index_int last) {
  outdata.at(0) = 0;
  for (index_int i=first; i<last; i++) {
    index_int i_in = i-first;
    outdata.at(0) += indata.at(i_in);
  }
}

#if 0
struct coord_and_mask{ double coord[3]; int mask; };
MPI_Datatype masked_coordinate;

void add_if_mask_mixed( void *indata, void * outdata,int *dim,MPI_Datatype *type ) {
  struct coord_and_mask *incoord = (struct coord_and_mask*)indata;
  struct coord_and_mask *outcoord = (struct coord_and_mask*)outdata;
  if (incoord->mask) {
    if (outcoord->mask) {
      for (int id=0; id<3; id++) {
	outcoord->coord[id] += incoord->coord[id];
	outcoord->mask += 1;
      }
    } else {
      for (int id=0; id<3; id++) {
	outcoord->coord[id] = incoord->coord[id];
	outcoord->mask = 1;
      }
    }
  } // if the input is masked, we leave the inout alone.
}

void masked_reduct_mixed(void *data) {

  struct coord_and_mask point;
  int lengths[2]; lengths[0] = 3; lengths[1] = 1;
  MPI_Aint displs[2];
  displs[0] = (size_t)&(point.coord) - (size_t)&(point);
  displs[1] = (size_t)&(point.mask) - (size_t)&(point);
  MPI_Datatype types[2]; types[0] = MPI_DOUBLE; types[1] = MPI_INT;
  MPI_Type_create_struct(2,lengths,displs,types,&masked_coordinate);
  
  MPI_Op masked_add;
  MPI_Op_create(add_if_mask,1,&masked_add);
}

#endif
