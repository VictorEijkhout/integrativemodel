/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2023
 ****
 **** imp_functions.cxx : implementations of the support functions
 ****
 ****************************************************************/

#include <stdlib.h>
#include <stdio.h>

//#include "imp_base.h"
#include "imp_functions.h"
#include "imp_object.h"

using std::string;
using std::vector;

using std::shared_ptr;

#include "fmt/format.h"
using fmt::format;
using fmt::print;

/*!
  Set vector elements linearly. A useful test case.
 */
template<int d>
void vecsetconstant( kernel_function_args(d), double value )
{
  // index_int
  //   len  = outvector->local_domain()->volume();
  // index_int
  //   tar0 = outvector->location_of_first_index(p);

  // index_int first_index = outdistro->first_index_r(p).coord(0);
  // print("[%d] writing {} elements in object with size {}\n",
  // 	p.as_string(),len,outdata.size());
  // print("[%d] writing {} elements @ {} wich has size {}\n",
  // 	p.as_string(),len,(long)(outdata.data()),outdata.size());

  // description of the indices on which we work
  const auto& pstruct = outvector->local_domain(p);
  const coordinate<index_int,d>
    &pfirst = pstruct.first_index(),
    &plast  = pstruct.last_index(),
    out_offsets = outvector->global_domain().location_of(pfirst);
  const auto& indata = invectors.at(0)->data();
  auto& outdata = outvector->data();

  if constexpr (d==1) {
    for (index_int i=pfirst[0]; i<=plast[0]; i++) {
      index_int I = INDEX1D(i,out_offsets,out_nsize);
      //print("[{}] copy global index {}@{}:{}\n",p->as_string(),i,I,indata[I]);
      outdata.at(I) = indata.at(I);
    }
  } else throw( "Only d=1 supported for now" );

  // for ( auto& e : outdata )
  //   e = value;

  // for (index_int i=0; i<len; i++) {
  //   outdata.at(tar0+i) = value;
  // }  

  //  *flopcount += 0.;
}

#if 0
/*!
  A no-op function. This can be used as the function of an origin kernel: whatever data is
  in the output vector of the origin kernel will be used as-is.
*/
template<int d>
void vecnoset( kernel_function_args(d) )
{
  return;
}

/*!
 * A plain copy. This is often used to copy the beta distribution into the gamma.
 *
 * Subtlety: we use the output index only, so we can actually cover the case
 * where the distribution does not quite allow for copying, as in transposition.
 */
template<int d>
void veccopy( kernel_function_args(d) )
{
#include "def_out.cxx"

  // placement in the global data structures
  auto
    in_nstruct = indistro->get_numa_structure(),
    out_nstruct = outdistro->get_numa_structure();
  auto
    in_gstruct = indistro->get_global_structure(),
    out_gstruct = outdistro->get_global_structure();

  if (dim==1) {
    if (k>1) {
      for (index_int i=pfirst[0]; i<=plast[0]; i++) {
	index_int I = INDEX1D(i,out_offsets,out_nsize);
	for (int ik=0; ik<k; ik++)
	  outdata.at(k*I+ik) = indata.at(k*I+ik);
      }
    } else {
      for (index_int i=pfirst[0]; i<=plast[0]; i++) {
	index_int I = INDEX1D(i,out_offsets,out_nsize);
	//print("[{}] copy global index {}@{}:{}\n",p->as_string(),i,I,indata[I]);
	outdata.at(I) = indata.at(I);
      }
    }
  } else if (dim==2) {
    if (k>1) {
      for (index_int i=pfirst[0]; i<=plast[0]; i++) {
	for (index_int j=pfirst[1]; j<=plast[1]; j++) {
	  index_int IJ = INDEX2D(i,j,out_offsets,out_nsize);
	  for (int ik=0; ik<k; ik++)
	    outdata.at(k*IJ+ik) = indata.at(k*IJ+ik);
	}
      }
    } else {
      int done=0;
      //snippet copyloop2d
      for (index_int i=pfirst[0]; i<=plast[0]; i++) {
	for (index_int j=pfirst[1]; j<=plast[1]; j++) {
	  index_int IJ = INDEX2D(i,j,out_offsets,out_nsize);
	  if (!done) {
	    print("{} copy: {}\n",p.as_string(),indata.at(IJ)); done = 1; }
	  outdata.at(IJ) = indata.at(IJ);
	}
      }
      //snippet end
    }
  } else if (dim==3) {
    for (index_int i=pfirst[0]; i<=plast[0]; i++) {
      for (index_int j=pfirst[1]; j<=plast[1]; j++) {
	for (index_int k=pfirst[2]; k<=plast[2]; k++) {
	  index_int IJK = INDEX3D(i,j,k,out_offsets,out_nsize);
	  outdata.at(IJK) = indata.at(IJK);
	}
      }
    }
  } else
    throw(format("veccopy not implemented for d={}",dim));

  //  *flopcount += outdistro->volume(p);
}

/*!
  Delta function
 */
template<int d>
void vecdelta( kernel_function_args(d), domain_coordinate& delta)
{
  const auto outdistro = outvector->get_distribution();
  int dim = outdistro->dimensionality();

  auto outdata = outvector->get_data(p);

  // description of the indices on which we work
  auto pstruct = outdistro->get_processor_structure(p);
  domain_coordinate
    pfirst = pstruct->first_index_r(), plast = pstruct->last_index_r();

  // placement in the global data structures
  shared_ptr<multi_indexstruct> out_nstruct;
  //  multi_indexstruct out_gstruct;
  decltype(outdistro->get_numa_structure()) out_gstruct;
  try {
    out_nstruct = outdistro->get_numa_structure();
  } catch (string c) {
    throw(format("Error <<{}>> getting numa structure in vecdelta",c)); }
  try {
    out_gstruct = outdistro->get_global_structure();
  } catch (string c) {
    throw(format("Error <<{}>> getting global structure of <<{}>> in vecdelta",
		      c,outvector->get_name())); }
  domain_coordinate
    out_nsize = out_nstruct->local_size_r(),
    out_offsets = outdistro->offset_vector();

  if (dim==1) {
    for (index_int i=pfirst[0]; i<=plast[0]; i++) {
      index_int I = INDEX1D(i,out_offsets,out_nsize);
      outdata.at(I) = 0.;
    }
    if (outdistro->get_processor_structure(p)->contains_element(delta))
      outdata.at( INDEX1D(delta[0],out_offsets,out_nsize) ) = 1.;
  } else
    throw(format("Can not set delta for dim={}",dim));
}

/*!
  Set vector elements linearly. A useful test case.
 */
template<int d>
void vecsetlinear( kernel_function_args(d) )
{
  const auto outdistro = outvector->get_distribution();
  int dim = outdistro->dimensionality();
  auto outdata = outvector->get_data(p);

  // description of the indices on which we work
  auto pstruct = outdistro->get_processor_structure(p);
  domain_coordinate
    pfirst = pstruct->first_index_r(), plast = pstruct->last_index_r();

  // various structures
  auto out_nstruct = outdistro->get_numa_structure();
  auto out_gstruct = outdistro->get_global_structure();
  domain_coordinate
    out_nsize = out_nstruct->local_size_r(),
    out_gsize = out_gstruct->local_size_r(),
    out_offsets = outdistro->offset_vector();

  if (dim==1) {
    for (index_int i=pfirst[0]; i<=plast[0]; i++) {
      index_int I = INDEX1D(i,out_offsets,out_nsize);
      outdata.at(I) = (double)COORD1D(i,out_gsize);
    }
  } else if (dim==2) {
    for (index_int i=pfirst[0]; i<=plast[0]; i++) {
      for (index_int j=pfirst[1]; j<=plast[1]; j++) {
	index_int IJ = INDEX2D(i,j,out_offsets,out_nsize);
	outdata.at(IJ) = COORD2D(i,j,out_gsize);
      }
    }
  } else if (dim==3) {
    for (index_int i=pfirst[0]; i<=plast[0]; i++) {
      for (index_int j=pfirst[1]; j<=plast[1]; j++) {
	for (index_int k=pfirst[2]; k<=plast[2]; k++) {
	  index_int IJK = INDEX3D(i,j,k,out_offsets,out_nsize);
	  outdata.at(IJK) = COORD3D(i,j,k,out_gsize);
	}
      }
    }
  } else
    throw(format("vecsetlinear not implemented for d={}",dim));

  //  *flopcount += 0.;
}

/*!
  Set vector elements linearly.
 */
template<int d>
void vecsetlinear2d( kernel_function_args(d) )
{
  const auto outdistro = outvector->get_distribution();
  int dim = p.same_dimensionality( outdistro->dimensionality() );
  auto outdata = outvector->get_data(p);

  index_int
    tar0 = outdistro->location_of_first_index(outdistro,p),
    len = outdistro->volume(p);
  domain_coordinate
    gfirst = outdistro->first_index_r(p),
    lens = outdistro->local_size_r(p),
    gsizes = outdistro->global_size();

  auto bigstruct = outdistro->get_global_structure();
  const domain_coordinate first2 = outdistro->first_index_r(p); // weird: the ampersand is essential
  index_int first_index = first2.linear_location_in( bigstruct );

  if (dim==2) {
    for (int i=0; i<lens[0]; i++) {
      index_int gi = gfirst[0]+i;
      for (int j=0; j<lens[1]; j++) {
	index_int gj = gfirst[1]+j;
	outdata.at(i*lens[0]+j) = gi*gsizes[0] + gj;
      }
    }
  } else
    throw(string("dimensionality should be 2 for setlinear2d"));

  //  *flopcount += 0.;
}

template<int d>
void vecsetconstantzero( kernel_function_args(d) ) {
  vecsetconstant(kernel_function_call,0.); 
};

template<int d>
void vecsetconstantone( kernel_function_args(d) ) {
  vecsetconstant(kernel_function_call,1.); 
};

/*!
  Set vector elements linearly. A useful test case.
*/
template<int d>
void vecsetconstantp( kernel_function_args(d) ) {
  const auto outdistro = outvector->get_distribution();
  int dim;
  try {
    int dim0 = outdistro->dimensionality();
    dim = p.same_dimensionality(dim0);
  } catch (string c) {
    throw(format("Error <<c>> checking dim coordinate <<{}>> against <<{}>>",
		      c,p.as_string(),outvector->as_string()));
  }
  double value;
  try {
    int ivalue = outdistro->get_decomposition().linearize(p);
    value = (double)ivalue;
  } catch (string c) {
    throw(format("Error <<{}>> in converting coordinate <<{}>>",c,p.as_string()));
  }
  
  auto outdata = outvector->get_data(p);
  
  vector<index_int>
    pfirst = outdistro->first_index_r(p).data(),
    plast  = outdistro->last_index_r(p).data(),
    offsets = outdistro->offset_vector().data(),
    nsize = outdistro->get_numa_structure()->local_size_r().data();
  
  if (dim==1) {
    for (index_int i=pfirst[0]; i<=plast[0]; i++) {
      index_int loc = INDEX1D(i,offsets,nsize);
      //print("[{}] write {} @{}->{} in {}\n",p.as_string(),value,i,loc,(long int)outdata);
      outdata.at( loc ) = value;
    }
  } else if (dim==2) {
    for (index_int i=pfirst[0]; i<=plast[0]; i++) {
      for (index_int j=pfirst[1]; j<=plast[1]; j++) {
	outdata.at( INDEX2D(i,j,offsets,nsize) ) = value;
      }
    }
  } else if (dim==3) {
    index_int
      ioffset = offsets[0], joffset = offsets[1], koffset = offsets[2];
    for (index_int i=pfirst[0]; i<=plast[0]; i++) {
      for (index_int j=pfirst[1]; j<=plast[1]; j++) {
	for (index_int k=pfirst[2]; k<=plast[2]; k++) {
	  outdata.at( INDEX3D(i,j,k,offsets,nsize) ) = value;
	}
      }
    }
  } else
    throw(format("vecsetconstantp not implemented for d={}",dim));

  //  *flopcount += 0.;
}

/*! 
  Deceptively descriptive name for a very ad-hoc function:
  set \$! y_i = p+.5\$!
*/
template<int d>
void vector_gen(kernel_function_args(d) ) {
  const auto outdistro = outvector->get_distribution();
  int dim;
  try {
    int dim0 = outdistro->dimensionality();
    dim = p.same_dimensionality(dim0);
  } catch (string c) {
    throw(format("Error <<c>> checking dim coordinate <<{}>> against <<{}>>",
		      c,p.as_string(),outvector->as_string()));
  }

  double value;
  try {
    int ivalue = outdistro->get_decomposition().linearize(p);
    value = (double)ivalue;
  } catch (string c) {
    throw(format("Error <<{}>> in converting coordinate <<{}>>",c,p.as_string()));
  }
  //print("Set constantp on {} to {}\n",p.as_string(),value);

  int k = outdistro->get_orthogonal_dimension();
  if (k>1) throw(string("No ortho supported"));

#include "impfunc_struct_index.cxx"

  //int plinear = p.linearize(outdistro);

  if (dim==1) {
    index_int last_index = INDEX1D(plast[0],offsets,nsize);

    // VLE we should really reinstate this test
    // if (last_index>=outvector->get_distribution()->numa_local_size()) {
    //   throw(format("Last index {} outside numa area {}",
    // 			last_index,outvector->get_numa_structure().as_string()));
    // }

    // print("[{}] writing into {} starting {}\n",
    // 	       p.as_string(),(long int)outdata,
    // 	       INDEX1D(pfirst[0],offsets,nsize));
    for (index_int i=pfirst[0]; i<=plast[0]; i++) {
      index_int I = INDEX1D(i,offsets,nsize);
      outdata.at(I) = value+.5;
    }
  } else
    throw(format("vector_gen not implemented for d={}",dim));
  //  *flopcount += plast[0]-pfirst[0]+1;
}

/*
 * Shift functions
 */

//snippet omprightshiftbump
/*!
  Shift an array to the right without wrap connections.
  We leave the global first position undefined.
*/
template<int d>
void vecshiftrightbump( kernel_function_args(d) ) {
  auto invector = invectors.at(0);
  const auto indistro = invector->get_distribution(),
    outdistro = outvector->get_distribution();

  auto indata = invector->get_data(p),
    outdata = outvector->get_data(p);

  // description of the indices on which we work
  auto pstruct = outdistro->get_processor_structure(p);
  domain_coordinate
    pfirst = pstruct->first_index_r(), plast = pstruct->last_index_r();

  // placement in the global data structures
  auto in_nstruct = indistro->get_numa_structure(),
    out_nstruct = outdistro->get_numa_structure();
  auto
    in_gstruct = indistro->get_global_structure(),
    out_gstruct = outdistro->get_global_structure();
  domain_coordinate
    in_nsize = in_nstruct->local_size_r(),
    out_nsize = out_nstruct->local_size_r(),
    in_offsets = indistro->offset_vector(),
    out_offsets = outdistro->offset_vector();

  index_int pfirst0 = pfirst[0];
  if (pfirst0==0) pfirst0++;
  for (index_int i=pfirst0; i<=plast[0]; i++) {
    index_int Iout = INDEX1D(i,out_offsets,out_nsize), Iin = INDEX1D(i,in_offsets,in_nsize);
    outdata.at(Iout) = indata.at(Iin-1);
  }
  index_int len = plast[0]-pfirst0;
  //  *flopcount += len;
}
//snippet end

//snippet leftshiftbump
/*!
  Shift an array to the left without wrap connections.
  We leave the global last position undefined.
*/
template<int d>
void vecshiftleftbump( kernel_function_args(d) ) {
  auto invector = invectors.at(0);
  const auto indistro = invector->get_distribution(),
    outdistro = outvector->get_distribution();

  auto indata = invector->get_data(p),
    outdata = outvector->get_data(p);

  // description of the indices on which we work
  auto pstruct = outdistro->get_processor_structure(p);
  domain_coordinate
    pfirst = pstruct->first_index_r(), plast = pstruct->last_index_r();

  // placement in the global data structures
  auto in_nstruct = indistro->get_numa_structure(),
    out_nstruct = outdistro->get_numa_structure();
  auto
    in_gstruct = indistro->get_global_structure(),
    out_gstruct = outdistro->get_global_structure();
  domain_coordinate
    in_nsize = in_nstruct->local_size_r(),
    out_nsize = out_nstruct->local_size_r(),
    in_offsets = indistro->offset_vector(),
    out_offsets = outdistro->offset_vector();

  index_int pfirst0 = pfirst[0], plast0 = plast[0];
  if (plast0==in_gstruct->last_index_r()[0]) plast0--;
  print("p={} copies {}-{}\n",p.as_string(),pfirst0,plast0);
  for (index_int i=pfirst0; i<=plast0; i++) {
    index_int Iout = INDEX1D(i,out_offsets,out_nsize), Iin = INDEX1D(i,in_offsets,in_nsize);
    outdata.at(Iout) = indata.at(Iin+1);
  }
  index_int len = plast0-pfirst0+1;
  //  *flopcount += len;
}
//snippet end

/*
 * Nbody stuff
 */

/*!
  Compute the center of mass of an array of particles by comparing two-and-two

  - k=1: add charges
  - k=2: 0=charges added, 1=new center
 */
template<int d>
void scansumk( kernel_function_args(d),int k ) {
  auto invector = invectors.at(0);
  const auto indistro = invector->get_distribution(),
    outdistro = outvector->get_distribution();
  int
    dim = p.same_dimensionality(outdistro->dimensionality());

  auto indata = invector->get_data(p);
  int insize = indistro->volume(p);

  auto outdata = outvector->get_data(p);
  int outsize = outdistro->volume(p);

  if (k<0 || k>2)
    throw(format("scansumk k={} meaningless",k));

  if (2*outsize!=insize)
    throw(format("scansum: in/out not compatible: {} {}\n",insize,outsize));

  // description of the indices on which we work
  auto pstruct = outdistro->get_processor_structure(p),
    qstruct = indistro->get_processor_structure(p);
  domain_coordinate
    pfirst = pstruct->first_index_r(), plast = pstruct->last_index_r(),
    qfirst = qstruct->first_index_r(), qlast = qstruct->last_index_r();

  // placement in the global data structures
  auto in_nstruct = indistro->get_numa_structure(),
    out_nstruct = outdistro->get_numa_structure();
  auto
    in_gstruct = indistro->get_global_structure(),
    out_gstruct = outdistro->get_global_structure();
  domain_coordinate
    in_nsize = in_nstruct->local_size_r(),
    out_nsize = out_nstruct->local_size_r(),
    in_offsets = indistro->offset_vector(),
    out_offsets = outdistro->offset_vector();

  if (dim==1) {
    // print("[{}] summing {}-{} into {}-{}\n",
    // 	       p.as_string(),qfirst[0],qlast[0],pfirst[0],plast[0]);
    index_int Iin = INDEX1Dk(qfirst[0],in_offsets,in_nsize,k);
    for (index_int i=pfirst[0]; i<=plast[0]; i++) {
      index_int Iout = INDEX1Dk(i,out_offsets,out_nsize,k);
      double v1 = indata.at(Iin), v2 = indata.at(Iin+k), v3=v1+v2;
      outdata.at(Iout) = v3;
      if (k==2) {
	double x1 = indata.at(Iin+1), x2 = indata.at(Iin+3);
      	outdata.at(Iout+1) = sqrt( v3*x1*x1*x2*x2 / (v1*x2*x2+v2*x1*x1) );
      }
      Iin += 2*k;
    }
  } else
    throw(string("scansumk only for d=1"));

  //  *flopcount += k*outsize;
}

//! Short-cut of \ref scansumk for k=1
template<int d>
void scansum( kernel_function_args(d) ) {
  scansumk(step,p,invectors,outvector,flopcount,1);
}

/*!
  Sum an array into a scalar

  \todo do some sanity check on the size of the output
*/
template<int d>
void summing( kernel_function_args(d) ) {
  auto invector = invectors.at(0);
  const auto indistro = invector->get_distribution(),
    outdistro = outvector->get_distribution();

  int
    dim = p.same_dimensionality(indistro->dimensionality()),
    k = indistro->get_orthogonal_dimension();

  auto indata = invector->get_data(p),
    outdata = outvector->get_data(p);

  // description of the indices on which we work
  auto qstruct = indistro->get_processor_structure(p);
  domain_coordinate
    qfirst = qstruct->first_index_r(), qlast = qstruct->last_index_r();

  // placement in the global data structures
  auto in_nstruct = indistro->get_numa_structure(),
    out_nstruct = outdistro->get_numa_structure();
  auto
    in_gstruct = indistro->get_global_structure(),
    out_gstruct = outdistro->get_global_structure();
  domain_coordinate
    in_nsize = in_nstruct->local_size_r(),
    out_nsize = out_nstruct->local_size_r(),
    in_offsets = indistro->offset_vector(),
    out_offsets = outdistro->offset_vector();

  double s = 0.; index_int len=0;
  if (dim==1) {
    if (k>1) {
      for (index_int i=qfirst[0]; i<=qlast[0]; i++) {
	index_int I = INDEX1D(i,in_offsets,in_nsize);
	for (int ik=0; ik<k; ik++)
	  s += indata.at(k*I+ik);
      }
    } else {
      for (index_int i=qfirst[0]; i<=qlast[0]; i++) {
	index_int I = INDEX1D(i,in_offsets,in_nsize);
	s += indata.at(I);
      }
    }
    len = qlast[0]-qfirst[0]+1;
  } else
    throw(format("veccopy not implemented for d={}",dim));

  outdata.at(0) = s;

  //  *flopcount += len*k;

#if 0
  index_int
    tar0 = outdistro->location_of_first_index(outdistro,p),
    src0 = indistro->location_of_first_index(indistro,p),
    len = indistro->volume(p);

  double s = 0.;
  int ortho = 1;
  //printf("[%d] summing %d element, starting %e",p.coord(0),len,indata[src0]);

  for (index_int i=0; i<len; i++) {
    //printf("[%d] i=%d idx=%d data=%e\n",p,i,src0+i,indata[src0+i]);
    //print("[{}] i={}, idx={}, data={}\n",p.as_string(),i,src0+i,indata[src0+i]);
    s += indata.at(src0+i);
  }
  //printf(", giving %e\n",s);
#endif
}

/*!
  Sum an array into a scalar and take the root. This is for norms

  \todo do some sanity check on the size of the output
*/
template<int d>
void rootofsumming( kernel_function_args(d) ) {
  auto invector = invectors.at(0);
  const auto indistro = invector->get_distribution(),
    outdistro = outvector->get_distribution();
  auto indata = invector->get_data(p),
    outdata = outvector->get_data(p);

  index_int
    tar0 = outdistro->location_of_first_index(outdistro,p),
    src0 = indistro->location_of_first_index(indistro,p),
    len = indistro->volume(p);

  double s = 0.;
  int ortho = 1;

  for (index_int i=0; i<len; i++)
    s += indata.at(src0+i);

  outdata.at(0) = sqrt(s);

  //  *flopcount += len;
}

//snippet ompnormsquared
//! Compute the local part of the norm of a vector squared.
template<int d>
void local_normsquared( kernel_function_args(d) ) {
#include "def_out.cxx"
  // auto invector = invectors.at(0);
  // const auto indistro = invector->get_distribution(),
  //   outdistro = outvector->get_distribution();
  // double
  //   *indata = invector->get_data(p),
  //   *outdata = outvector->get_data(p);

  // index_int
  //   tar0 = outdistro->location_of_first_index(outdistro,p),
  //   src0 = indistro->location_of_first_index(indistro,p),
  //   len = indistro->volume(p);

  double s = 0;
  //memory_buffer w;
  for (index_int i=qfirst[0]; i<=qlast[0]; i++) {
    index_int I = INDEX1D(i,in_offsets,in_nsize);
    s += indata.at(I) * indata.at(I);
  }
  // for (index_int i=0; i<len; i++) {
  //   //format_to(w.end(),"{} ",indata[src0+i]);
  //   s += indata[src0+i]*indata[src0+i];
  // }
  // print("norm squared of {}: {} comes to {}\n",
  // 	     invector->get_name(),w.str(),s);
  outdata.at( INDEX1D(pfirst[0],out_offsets,out_nsize) ) = s;

  //  *flopcount += 2*indistro->volume(p);
}
//snippet end

//! Compute the norm of the local part of a vector.
template<int d>
void local_norm( kernel_function_args(d) ) {
  auto invector = invectors.at(0);
  const auto indistro = invector->get_distribution(),
    outdistro = outvector->get_distribution();
  auto indata = invector->get_data(p),
    outdata = outvector->get_data(p);

  index_int
    tar0 = outdistro->location_of_first_index(outdistro,p),
    src0 = indistro->location_of_first_index(indistro,p),
    len = indistro->volume(p);

  double s = 0;
  for (index_int i=0; i<len; i++) {
    s += indata.at(src0+i)*indata.at(src0+i);
  }
  outdata.at(tar0) = sqrt(s);

  //  *flopcount += len;
}

//! The local part of the inner product of two vectors.
template<int d>
void local_inner_product( kernel_function_args(d) ) {
  if (invectors.size()<2)
    throw(format("local inner product: #vectors={}, s/b 2",invectors.size()));
  auto invector = invectors.at(0), othervector = invectors.at(1);
  const auto indistro = invector->get_distribution(),
    outdistro = outvector->get_distribution(),
    otherdistro = othervector->get_distribution();

  auto indata = invector->get_data(p),
    otherdata = othervector->get_data(p),
    outdata = outvector->get_data(p);

  index_int
    tar0 = outdistro->location_of_first_index(outdistro,p),
    src0 = indistro->location_of_first_index(indistro,p),
    oth0 = otherdistro->location_of_first_index(otherdistro,p),
    len = indistro->volume(p);

  if (len!=otherdistro->volume(p))
    throw(format("Incompatible sizes {}:{} {}:{}\n",
	  invector->get_name(),len,othervector->get_name(),otherdistro->volume(p)));

  double s = 0;
  for (index_int i=0; i<len; i++) {
    s += indata.at(src0+i)*otherdata.at(oth0+i);
  }
  //printf("[%d] local normquared %e written to %d\n",p,s,outloc);
  outdata.at(tar0) = s;

  //  *flopcount += len;
}

/*!
  Pointwise square root
 */
template<int d>
void vectorroot( kernel_function_args(d) ) {
  auto invector = invectors.at(0);
  const auto indistro = invector->get_distribution(),
    outdistro = outvector->get_distribution();

  auto indata = invector->get_data(p),
    outdata = outvector->get_data(p);

  index_int
    tar0 = outdistro->location_of_first_index(outdistro,p),
    src0 = indistro->location_of_first_index(indistro,p),
    len = outdistro->volume(p);//outdistro->local_size(p);

  int ortho = 1;

  for (index_int i=0; i<len; i++) {
    // if (i==0)
    //   print("Vector root : sqrt of {}\n",indata.at(src0+i) );
    outdata.at(tar0+i) = sqrt( indata.at(src0+i) );
  }  

  //  *flopcount += len*ortho;
}

/*!
  Multiply a vector by a scalar; this is a limited case of an AXPY.
  The scalar comes in as an extra input object.
*/
template<int d>
void vecscaleby( kernel_function_args(d) ) {
#include "def_out.cxx"
  auto
    inscalar = invectors.at(1);
  const auto
    scalardistro = inscalar->get_distribution();

  double a;
  {
    try {
      scalardistro->require_type_replicated();
      if (scalardistro->local_size_r(p)[0]!=1)
	throw(string("Inscalar object not single component"));
      a = inscalar->get_data(p).at(0);
    } catch (string c) { print("Error <<{}>> getting inscalar value\n",c);
      throw(format("vecscaleby of <<{}>> by <<{}>> failed",
			invector->get_name(),inscalar->as_string()));
    }
  }
  
  if (dim==1) {
    if (k>1) {
      for (index_int i=pfirst[0]; i<=plast[0]; i++) {
	index_int I = INDEX1D(i,out_offsets,out_nsize);
	for (int ik=0; ik<k; ik++)
	  outdata.at(k*I+ik) = a * indata.at(k*I+ik);
      }
    } else {
      for (index_int i=pfirst[0]; i<=plast[0]; i++) {
	index_int I = INDEX1D(i,out_offsets,out_nsize);
	outdata.at(I) = a * indata.at(I);
      }
    }
  } else if (dim==2) {
    if (k>1) {
      for (index_int i=pfirst[0]; i<=plast[0]; i++) {
	for (index_int j=pfirst[1]; j<=plast[1]; j++) {
	  index_int IJ = INDEX2D(i,j,out_offsets,out_nsize);
	  for (int ik=0; ik<k; ik++)
	    outdata.at(k*IJ+ik) = a *indata.at(k*IJ+ik);
	}
      }
    } else {
      int done=0;
      //snippet copyloop2d
      for (index_int i=pfirst[0]; i<=plast[0]; i++) {
	for (index_int j=pfirst[1]; j<=plast[1]; j++) {
	  index_int IJ = INDEX2D(i,j,out_offsets,out_nsize);
	  outdata.at(IJ) = a * indata.at(IJ);
	}
      }
      //snippet end
    }
  } else
    throw(format("veccopy not implemented for d={}",dim));

  //  *flopcount += outdistro->volume(p);
}

//! \todo replace this by vecscalebyc with lambda
template<int d>
void vecscalebytwo( kernel_function_args(d) ) {
  vecscalebyc(step,p,invectors,outvector,flopcount,2.);
}

/*!
  Multiply a vector by a scalar; this is a limited case of an AXPY.
  The scalar comes in through the context as a (void*)(double*)&scalar
  \todo replace void ctx by actual scale
*/
//snippet ompscalevec
template<int d>
void vecscalebyc( kernel_function_args(d),double a ) {
#include "def_out.cxx"
  // auto invector = invectors.at(0);
  // const auto indistro = invector->get_distribution();
  //outdistro = outvector->get_distribution();
  // double
  //   *indata = invector->get_data(p);
  //*outdata = outvector->get_data(p);

  // // description of the indices on which we work
  // auto pstruct = outdistro->get_processor_structure(p);
  // domain_coordinate
  //   pfirst = pstruct->first_index_r(), plast = pstruct->last_index_r();

  // // placement in the global data structures
  // auto in_nstruct = indistro->get_numa_structure(),
  //   out_nstruct = outdistro->get_numa_structure();
  // auto
  //   in_gstruct = indistro->get_global_structure(),
  //   out_gstruct = outdistro->get_global_structure();
  // domain_coordinate
  //   in_nsize = in_nstruct->local_size_r(),
  //   out_nsize = out_nstruct.local_size_r(),
  //   in_offsets = indistro->offset_vector(),
  //   out_offsets = outdistro->offset_vector();

  for (index_int i=pfirst[0]; i<=plast[0]; i++) {
    index_int
      Iout = INDEX1D(i,out_offsets,out_nsize),
      Iin = INDEX1D(i,in_offsets,in_nsize);
    // if (i==pfirst[0])
    //   print("{}: scale {}, value {} by {}, starting with {}, which is local index {}->{}\n",
    // 		 p.as_string(),invector->get_name(),indata.at(Iin),a,pfirst[0],Iin,Iout);
    outdata.at(Iout) = a*indata.at(Iin);
  }

  //  *flopcount += plast[0]-pfirst[0]+1;
}
//snippet end

/*!
  Multiply a vector by a scalar; this is a limited case of an AXPY.
  The scalar comes in through the context as a (void*)(double*)&scalar,
  or as an extra input object.
*/
template<int d>
void vecscaledownby( kernel_function_args(d) ) {
#include "def_out.cxx"
  if (invectors.size()!=2)
    throw(string("vecscaledownby needs two inputs"));
  auto // invector = invectors.at(0),
    inscalar = invectors.at(1);
  const auto // indistro = invector->get_distribution(),
    // outdistro = outvector->get_distribution(),
    scalardistro = inscalar->get_distribution();

  double a,ainv;
  {
    try {
      scalardistro->require_type_replicated();
      if (scalardistro->local_size_r(p)[0]!=1)
	throw(string("Inscalar object not single component"));
      a = inscalar->get_data(p).at(0);
      ainv = 1./a;
    } catch (string c) { print("Error <<{}>>\n",c); throw("vecscaledownby failed\n"); }
    if (scalardistro->local_size_r(p)[0]!=1)
      throw("Inscalar object not single component\n");
  }
  
  // for (index_int i=0; i<len; i++) {
  //   outdata.at(tar0+i) = ainv*indata.at(src0+i);
  // }
  for (index_int i=pfirst[0]; i<=plast[0]; i++) {
    index_int I = INDEX1D(i,out_offsets,out_nsize);
    //print("[{}] copy global index {}@{}:{}\n",p->as_string(),i,I,indata.at(I));
    outdata.at(I) = a * indata.at(I);
  }
  //  *flopcount += outdistro->volume(p);
}

template<int d>
void vecscaledownbyc( kernel_function_args(d),double a ) {
  auto invector = invectors.at(0);
  //inscalar = invectors.at(1);
  const auto indistro = invector->get_distribution(),
    outdistro = outvector->get_distribution();
    //scalardistro = inscalar->get_distribution();
  auto indata = invector->get_data(p),
    outdata = outvector->get_data(p);

  index_int
    tar0 = outdistro->location_of_first_index(outdistro,p),
    src0 = indistro->location_of_first_index(indistro,p),
    len = outdistro->volume(p);

  double ainv = 1./a;
  
  for (index_int i=0; i<len; i++) {
    outdata.at(tar0+i) = ainv*indata.at(src0+i);
  }
  //  *flopcount += len;
}

/*!
  Scalar combination of two vectors into a third.
  The second vector and both scalars come in through the context
  as an doubledouble_object_struct structure.
\todo 
*/
template<int d>
void vecaxbyz( kernel_function_args(d),void *ctx ) {
  auto invector1 = invectors.at(0), invector2 = invectors.at(2);
  const auto indistro1 = invector1->get_distribution(),
    indistro2 = invector2->get_distribution(),
    outdistro = outvector->get_distribution();

  auto x1data = invector1->get_data(p),
    x2data = invector2->get_data(p);
  charcharxyz_object_struct *ssx = (charcharxyz_object_struct*)ctx;
  auto outdata = outvector->get_data(p);
  // scalars
  double
    s1 = invectors.at(1)->get_data(p).at(0),
    s2 = invectors.at(3)->get_data(p).at(0);
  if (ssx->c1=='-') s1 = -s1;
  if (ssx->c2=='-') s2 = -s2;
  
  //print("axbyz computing {} uses scalars {},{}\n",outvector->get_name(),s1,s2);
  index_int
    tar0 = outdistro->location_of_first_index(outdistro,p),
    src1 = indistro1->location_of_first_index(indistro1,p),
    src2 = indistro2->location_of_first_index(indistro2,p),
    len = outdistro->volume(p);//outdistro->get_processor_structure(p)->volume();

  //memory_buffer w;
  for (index_int i=0; i<len; i++) {
    //format_to(w.end(),"{}+{} ",x1data.at(src1+i),x2data.at(src2+i));
    outdata.at(tar0+i) = s1*x1data.at(src1+i) + s2*x2data.at(src2+i);
  }
  // print("axbyz computing {}x{} + {}x{}->{} : {}\n",
  // 	     s1,invector1->get_name(),s2,invector2->get_name(),outvector->get_name(),w.str());
  //  *flopcount += 3*len;
}

/*
 * Central difference computation
 */
template<int d>
void central_difference( kernel_function_args(d) ) {
  central_difference_damp(step,p,invectors,outvector,flopcount,1.);
}

/*
 * Central difference computation with a damping parameter
 */
//snippet centraldiff
template<int d>
void central_difference_damp( kernel_function_args(d),double damp)
{
  auto invector = invectors.at(0);
  const auto indistro = invector->get_distribution(),
    outdistro = outvector->get_distribution();
  //snippet end
  int
    dim = p.same_dimensionality(outdistro->dimensionality()),
    k = outdistro->get_orthogonal_dimension();
  if (dim>1) throw(string("Central differences only 1d"));
  if (k>1) throw(string("Central differences no ortho"));

  //snippet centraldiff
  auto outdata = outvector->get_data(p),
    indata = invector->get_data(p);
  //snippet end
  
  // description of the indices on which we work
  //snippet centraldiff
  auto pstruct = outdistro->get_processor_structure(p);
  domain_coordinate
    pfirst = pstruct->first_index_r(), plast = pstruct->last_index_r();
  //snippet end
  
  // placement in the global data structures
  auto in_nstruct = indistro->get_numa_structure(),
    out_nstruct = outdistro->get_numa_structure();
  auto
    in_gstruct = indistro->get_global_structure(),
    out_gstruct = outdistro->get_global_structure();
  domain_coordinate
    in_nsize = in_nstruct->local_size_r(),
    out_nsize = out_nstruct->local_size_r(),
    in_offsets = indistro->offset_vector(),
    out_offsets = outdistro->offset_vector();

  //snippet centraldiff
  index_int lo=pfirst[0],hi=plast[0];
  //snippet end
  if (lo==out_gstruct->first_index_r()[0]) { // dirichlet left boundary condition
    index_int i = lo;    
    index_int
      Iin  = INDEX1D(i,in_offsets,in_nsize),
      Iout = INDEX1D(i,out_offsets,out_nsize);
    outdata.at(Iout) = ( 2*indata.at(Iin) - indata.at(Iin+1) )*damp;
    //    *flopcount += 3;
    lo++;
  }
  if (hi==out_gstruct->last_index_r()[0]) {
    index_int i = hi;    
    index_int
      Iin  = INDEX1D(i,in_offsets,in_nsize),
      Iout = INDEX1D(i,out_offsets,out_nsize);
    outdata.at(Iout) = ( 2*indata.at(Iin) - indata.at(Iin-1) )*damp;
    //    *flopcount += 3;
    hi--;
  }

  // ... but then we have a regular three-point stencil
  //snippet centraldiff
  for (index_int i=lo; i<=hi; i++) {
    index_int
      Iin  = INDEX1D(i,in_offsets,in_nsize),
      Iout = INDEX1D(i,out_offsets,out_nsize);
    outdata.at(Iout) = ( 2*indata.at(Iin) - indata.at(Iin-1) - indata.at(Iin+1) )
                    *damp;
  }
  //  *flopcount += 4*(hi-lo+1);
  //snippet end
  
}

//! Recursive function for index calculation in any number of dimensions
index_int INDEXanyD(domain_coordinate &i,domain_coordinate &off,domain_coordinate &siz,int d) {
  if (d==1) {
    index_int
      id = i[0], od = off[0];
    //print("for d={}, i={}, off={}\n",d,id,od);
    return id-od;
  } else {
    index_int
      p = INDEXanyD(i,off,siz,d-1),
      sd = siz[d-1], id = i[d-1], od = off[d-1];
    //print("for d={}, prev={}, size={}, i={}, off={}\n",d,p,sd,id,od);
    return p*sd + id-od;
  }
};

//snippet centraldiffd
template<int d>
void central_difference_anyd( kernel_function_args(d) ) {
  //snippet end
  auto invector = invectors.at(0);
  const auto indistro = invector->get_distribution(),
    outdistro = outvector->get_distribution();
  int
    dim = p.same_dimensionality(outdistro->dimensionality()),
    k = outdistro->get_orthogonal_dimension();
  if (k>1) throw(string("Central differences no ortho"));

  //snippet centraldiffd
  auto outdata = outvector->get_data(p),
    indata = invector->get_data(p);
  //snippet end
  
  // description of the indices on which we work
  //snippet centraldiffd
  auto pstruct = outdistro->get_processor_structure(p);
  domain_coordinate
    pfirst = pstruct->first_index_r(), plast = pstruct->last_index_r();
  //snippet end
  
  // placement in the global data structures
  auto in_nstruct = indistro->get_numa_structure(),
    out_nstruct = outdistro->get_numa_structure();
  auto
    in_gstruct = indistro->get_global_structure(),
    out_gstruct = outdistro->get_global_structure();
  domain_coordinate
    in_nsize = in_nstruct->local_size_r(),
    out_nsize = out_nstruct->local_size_r(),
    in_offsets = indistro->offset_vector(),
    out_offsets = outdistro->offset_vector();

  //snippet centraldiffd
  auto begin = pstruct->begin();
  auto end = pstruct->end();
  for ( auto ii=begin; !( ii==end ); ++ii) {
    domain_coordinate i = *ii;
    index_int
      Iin  = INDEXanyD(i,in_offsets,in_nsize,dim),
      Iout = INDEXanyD(i,out_offsets,out_nsize,dim);
    outdata.at(Iout) = 2*indata.at(Iin); // - indata.at(Iin-1) - indata.at(Iin+1) );
  }
  //  *flopcount += 4*pstruct->volume();
  //snippet end
  
}

//! Recursive derivation of multigrid coarse levels
std::shared_ptr<indexstruct> halfinterval(index_int i) {
  return std::shared_ptr<indexstruct>{ new contiguous_indexstruct(i/2) };
}
//! Signature function for 1D multigrid
std::shared_ptr<indexstruct> doubleinterval(index_int i) {
  return std::shared_ptr<indexstruct>{ new contiguous_indexstruct(2*i,2*i+1) };
}

//! \todo change void* to string
template<int d>
void print_trace_message( kernel_function_args(d),void *ctx ) {
  auto inobj = invectors.at(0);
  string *c = (string*)(ctx);
  if (p.is_zero())
    print("{}: {}\n",*c,inobj->get_data(p).at(0));
};

#endif

template void vecsetconstant<1>( kernel_function_args(1), double value );
template void vecsetconstant<2>( kernel_function_args(2), double value );
template void vecsetconstant<3>( kernel_function_args(3), double value );
