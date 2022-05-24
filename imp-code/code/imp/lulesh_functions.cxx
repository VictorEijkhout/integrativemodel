/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-9
 ****
 **** lulesh_functions.cxx : implementations of the lulesh support functions
 ****
 ****************************************************************/

#include <stdlib.h>
#include <stdio.h>

#include "imp_base.h"
using std::shared_ptr;
#include "lulesh_functions.h"

/****
 **** Elements to local node
 ****/

//snippet lulesheverydivby2
//! Divide each component by two
domain_coordinate signature_coordinate_element_to_local( const domain_coordinate &i ) {
  return i.operate( ioperator("/2") );
};
shared_ptr<multi_indexstruct> signature_struct_element_to_local( const multi_indexstruct &i ) {
  if (!i.is_contiguous())
    throw(std::string("signature_struct_element_to_local only for contiguous"));
  auto divop = ioperator("/2");
  return shared_ptr<multi_indexstruct>
    ( new contiguous_multi_indexstruct
      (i.first_index_r().operate(divop),i.last_index_r().operate(divop)) );
};
//snippet end

/*!
  Element to local node mapping: Take every input element, and replicate it a number of times.
 */
void element_to_local_function( kernel_function_args,int dim )
{
  auto invector = invectors.at(0);
  const auto indistro = invector->get_distribution(),
    outdistro = outvector->get_distribution();
  auto indata = invector->get_data(p),
    outdata = outvector->get_data(p);

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
    in_gstruct = indistro->get_enclosing_structure(),
    out_gstruct = outdistro->get_enclosing_structure();
  domain_coordinate
    in_nsize = in_nstruct->local_size_r(),
    out_nsize = out_nstruct->local_size_r(),
    in_offsets = in_nstruct->first_index_r() - in_gstruct->first_index_r(),
    out_offsets = out_nstruct->first_index_r() - out_gstruct->first_index_r();
  if (dim==1) {
    for (index_int i=qfirst[0]; i<=qlast[0]; i++)
      for (index_int ii=2*i; ii<2*i+2; ii++) {
        index_int
          I = INDEX1D(i,in_offsets,in_nsize), II = INDEX1D(ii,out_offsets,out_nsize);
        outdata.at(II) = indata.at(I);
      }
  } else if (dim==2) {
  //snippet bcaste2ln
    for (index_int i=qfirst[0]; i<=qlast[0]; i++)
      for (index_int ii=2*i; ii<2*i+2; ii++)
        for (index_int j=qfirst[1]; j<=qlast[1]; j++)
          for (index_int jj=2*j; jj<2*j+2; jj++) {
            index_int IJ = INDEX2D(i,j,in_offsets,in_nsize),
              IIJJ = INDEX2D(ii,jj,out_offsets,out_nsize);
            outdata.at(IIJJ) = indata.at(IJ);
          }
  //snippet end
  }
  // fmt::print("Copy struct={} from {} to {}\nnuma={} global={} into\nnuma={} global={}\n\n",
  //              pstruct->as_string(),
  //              invector->get_name(),outvector->get_name(),
  //              in_nstruct->as_string(),in_gstruct->as_string(),
  //              out_nstruct->as_string(),out_gstruct->as_string());
  
  *flopcount += pstruct->volume();
}

/****
 **** Local nodes to global
 ****/

//! Compute local node numbers from global, multi-d case.
//snippet luleshng2nl
shared_ptr<multi_indexstruct> signature_local_from_global
    ( const multi_indexstruct &g,const multi_indexstruct &enc ) {
  int dim = g.get_same_dimensionality(enc.get_dimensionality());
  domain_coordinate_allones allones(dim);
  auto range = shared_ptr<multi_indexstruct>
    ( new contiguous_multi_indexstruct
      ( g.first_index_r()*2-allones,g.last_index_r()*2 ) );
  return range->intersect(enc);
};
//snippet end

//! \todo what does that local_nodes_domain do?
void local_to_global_function
    ( kernel_function_args,const multi_indexstruct &local_nodes_domain)
{
  auto invector = invectors.at(0);
  const auto indistro = invector->get_distribution(),
    outdistro = outvector->get_distribution();
  auto indata = invector->get_data(p),
    outdata = outvector->get_data(p);
  int dim = outdistro->get_same_dimensionality( indistro->get_dimensionality() );

  index_int
    tar0 = outdistro->location_of_first_index(outdistro,p),
    src0 = indistro->location_of_first_index(indistro,p),
    outlen = outdistro->volume(p),
    inlen = indistro->volume(p);

  int ortho = 1;
  if (0) {
  } else if (dim==2) {
    auto pstruct = outdistro->get_processor_structure(p);
    domain_coordinate
      pfirst = pstruct->first_index_r(), plast = pstruct->last_index_r();

    auto in_nstruct = indistro->get_numa_structure(),
      out_nstruct = outdistro->get_numa_structure();
    auto
      in_gstruct = indistro->get_enclosing_structure(),
      out_gstruct = outdistro->get_enclosing_structure();
    domain_coordinate
      out_gfirst = out_gstruct->first_index_r(), out_glast = out_gstruct->last_index_r(),
      in_nsize = in_nstruct->local_size_r(),
      out_nsize = out_nstruct->local_size_r(),
      in_offsets = in_nstruct->first_index_r() - in_gstruct->first_index_r(),
      out_offsets = out_nstruct->first_index_r() - out_gstruct->first_index_r();

    //snippet l2gfunction
    for (index_int i=pfirst[0]; i<=plast[0]; i++) {
      bool skip_first_i = i==out_gfirst[0], skip_last_i = i==out_glast[0];
      for (index_int j=pfirst[1]; j<=plast[1]; j++) {
        bool skip_first_j = j==out_gfirst[1], skip_last_j = j==out_glast[1];
        outdata.at( INDEX2D(i,j,out_offsets,out_nsize) ) = 
          ( !skip_first_i && !skip_first_j
            ? indata.at( INDEX2D(2*i-1,2*j-1,in_offsets,in_nsize) ) : 0 )
          +
          ( !skip_first_i && !skip_last_j
            ? indata.at( INDEX2D(2*i-1,2*j,in_offsets,in_nsize) ) : 0 )
          +
          ( !skip_last_i && !skip_first_j
            ? indata.at( INDEX2D(2*i,2*j-1,in_offsets,in_nsize) ) : 0 )
          +
          ( !skip_last_i && !skip_last_j
            ? indata.at( INDEX2D(2*i,2*j,in_offsets,in_nsize) ) : 0 )
          ;
      }
    }
    //snippet end
  } else if (dim==1) {
    index_int itar = tar0, isrc = src0, global_last = outdistro->global_volume();
    for (index_int g=outdistro->first_index_r(p).coord(0);
         g<=outdistro->last_index_r(p).coord(0); g++) {
      index_int e = g/2; int m = g%2;
      if (g>=2 && g<global_last-1) {
        if (m==0) {
          outdata.at(itar++) = indata.at(isrc) + indata.at(isrc+2); isrc++;
        } else {
          outdata.at(itar++) = indata.at(isrc) + indata.at(isrc+2); isrc += 3;
           }
      } else
        outdata.at(itar++) = indata.at(isrc++);
    }
  } else
    throw(fmt::format("Can not sum_mod2 for dim={}",dim));

  *flopcount += outlen;
}

/****
 **** Global node back to local
 ****/

//snippet lulesh_global_node_to_local
shared_ptr<multi_indexstruct> signature_global_node_to_local( const multi_indexstruct &l ) {
  return
    l.operate( ioperator(">>1") ) -> operate( ioperator("/2") );
};
//snippet end

/*!
  Duplicate every global node over two local nodes.
  \todo the left/right tests should be against first/last coordinate, not gsizes
  \todo what does that local_nodes_domain do?
*/
void function_global_node_to_local
    ( kernel_function_args,const multi_indexstruct &local_nodes_domain)
{
  auto invector = invectors.at(0);
  const auto indistro = invector->get_distribution(),
    outdistro = outvector->get_distribution();

  int dim = p.get_same_dimensionality(outdistro->get_dimensionality());
  auto global_nodes = invectors.at(0), local_nodes = outvector;
  const auto global_distro = global_nodes->get_distribution(),
    local_distro = local_nodes->get_distribution();

  auto local_nodes_data = local_nodes->get_data(p),
    global_nodes_data = global_nodes->get_data(p);

  // description of the indices on which we work
  auto local_nodes_struct = local_distro->get_processor_structure(p),
    global_nodes_struct = global_distro->get_processor_structure(p);
  domain_coordinate
    qfirst = global_nodes_struct->first_index_r(), qlast = global_nodes_struct->last_index_r(),
    pfirst = local_nodes_struct->first_index_r(), plast = local_nodes_struct->last_index_r();

  // placement in the global data structures
  auto in_nstruct = global_distro->get_numa_structure(),
    out_nstruct = local_distro->get_numa_structure();
  auto
    in_gstruct = global_distro->get_enclosing_structure(),
    out_gstruct = local_distro->get_enclosing_structure();
  domain_coordinate
    global_nodes_sizes = in_gstruct->local_size_r(),
    in_nsize = in_nstruct->local_size_r(),
    out_nsize = out_nstruct->local_size_r(),
    in_offsets = in_nstruct->first_index_r() - in_gstruct->first_index_r(),
    out_offsets = out_nstruct->first_index_r() - out_gstruct->first_index_r();

  index_int
    involume = global_nodes->volume(p),
    outvolume = local_nodes->volume(p);
  
  /*
   * For once we range over input indices
   */
  if (dim==2) {
    //snippet function_global_to_local
    // fmt::print("{}: inrange {}-{} x {}-{}, outrange {}-{} x {}-{}\n",
    // 	       p.as_string(),
    // 	       qfirst[0],qlast[0],qfirst[1],qlast[1],
    // 	       pfirst[0],plast[0],pfirst[1],plast[1]);
    for (index_int i=qfirst[0]; i<=qlast[0]; i++) {
      for (index_int j=qfirst[1]; j<=qlast[1]; j++) {
        bool
          left_i = i==0, right_i = i==global_nodes_sizes[0],
          left_j = j==0, right_j = j==global_nodes_sizes[1];
        index_int Iin = INDEX2D(i,j,in_offsets,in_nsize);
	if (Iin<0 || Iin>=involume)
	  throw(fmt::format("{}: Iin {},{} out of bound",p.as_string(),i,j));
	continue;
        double g = global_nodes_data.at(Iin);
        if (!left_i && !left_j) {
          index_int Iout = INDEX2D( 2*i-1,2*j-1, out_offsets,out_nsize );
	  if (Iout<0 || Iout>=outvolume)
	    throw(fmt::format("{}: Iout {},{} out of bound",p.as_string(),2*i-1,2*j-1));
          local_nodes_data.at(Iout) = g; }
        if (!right_i && !left_j) {
          index_int Iout = INDEX2D( 2*i,  2*j-1, out_offsets,out_nsize );
	  if (Iout<0 || Iout>=outvolume)
	    throw(fmt::format("{}: Iout {},{} out of bound",p.as_string(),2*i,2*j-1));
          local_nodes_data.at(Iout) = g; }
        if (!left_i && !right_j) {
          index_int Iout = INDEX2D( 2*i-1,2*j,   out_offsets,out_nsize );
	  if (Iout<0 || Iout>=outvolume)
	    throw(fmt::format("{}: Iout {},{} out of bound",p.as_string(),2*i-1,2*j));
          local_nodes_data.at(Iout) = g; }
        if (!right_i && !right_j) {
          index_int Iout = INDEX2D( 2*i,  2*j,   out_offsets,out_nsize );
	  if (Iout<0 || Iout>=outvolume)
	    throw(fmt::format("{}: Iout {},{} out of bound",p.as_string(),2*i,2*j));
          local_nodes_data.at(Iout) = g; }
      }
    }
    //snippet end
  } else
    throw(std::string("Function function_global_node_to_local only for d=2"));
}

/****
 **** And finally local nodes back to elements
 ****/

//snippet luleshsigl2e
shared_ptr<multi_indexstruct> signature_local_to_element
    ( int dim,const multi_indexstruct &i ) {
  domain_coordinate_allones allones(dim);
  auto times2 = ioperator("*2");
  return shared_ptr<multi_indexstruct>
    ( new contiguous_multi_indexstruct
      ( i.first_index_r()*2,i.last_index_r()*2+allones ) );
}
//snippet end

/*!
  Element to local node mapping: Take every input element, and replicate it a number of times.
 */
void local_node_to_element_function( kernel_function_args )
{
  auto invector = invectors.at(0);
  const auto indistro = invector->get_distribution(),
    outdistro = outvector->get_distribution();
  int dim = outdistro->get_same_dimensionality(indistro->get_dimensionality());
  auto indata = invector->get_data(p),
    outdata = outvector->get_data(p);

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
    in_gstruct = indistro->get_enclosing_structure(),
    out_gstruct = outdistro->get_enclosing_structure();
  domain_coordinate
    in_nsize = in_nstruct->local_size_r(),
    out_nsize = out_nstruct->local_size_r(),
    in_offsets = in_nstruct->first_index_r() - in_gstruct->first_index_r(),
    out_offsets = out_nstruct->first_index_r() - out_gstruct->first_index_r();
  // fmt::print("{} node to element offsets {} in numa {}\n",
  //              p.as_string(),in_offsets.as_string(),in_nstruct->as_string());

  if (dim==2) {
  //snippet lugatherl2e
    for (index_int i=pfirst[0]; i<=plast[0]; i++)
      for (index_int j=pfirst[1]; j<=plast[1]; j++) {
        index_int IJ = INDEX2D(i,j,out_offsets,out_nsize);
        outdata.at(IJ) =
          ( indata.at(INDEX2D(2*i,  2*j,   in_offsets,in_nsize)) +
            indata.at(INDEX2D(2*i,  2*j+1, in_offsets,in_nsize)) +
            indata.at(INDEX2D(2*i+1,2*j,   in_offsets,in_nsize)) +
            indata.at(INDEX2D(2*i+1,2*j+1, in_offsets,in_nsize))
            ) / 4.;                      
      }
  //snippet end
  } else {
    throw(std::string("no lulesh functions except 2d"));
  }

  *flopcount += 4*pstruct->volume();
}

