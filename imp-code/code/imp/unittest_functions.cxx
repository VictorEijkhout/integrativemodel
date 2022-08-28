/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-9
 ****
 **** unittest_functions.cxx : independent implementations of unittest functions
 ****
 ****************************************************************/

#include <stdlib.h>
#include <stdio.h>

#include "imp_base.h"
#include "unittest_functions.h"

using fmt::format;
using fmt::print;

/*!
  Test if the local execute function sees the right global size.
*/
void test_globalsize( kernel_function_args,index_int test_globalsize )
{
  auto invector = invectors.at(0);
  index_int is_globalsize = outvector->get_distribution()->global_volume();
  if (is_globalsize!=test_globalsize)
    throw(format("global is {}, s/b {}",is_globalsize,test_globalsize));
}

/*!
  Test if the local execute function sees the number of processors;
  the desired number is coming in as context
*/
void test_nprocs( kernel_function_args,int test_nprocs )
{
  auto invector = invectors.at(0);
  int is_nprocs;

  is_nprocs=invector->domains_volume();
  if (is_nprocs!=test_nprocs) {
    throw(format("Incorrect nprocs tested on input: {} s/b {}",is_nprocs,test_nprocs));
  }

  is_nprocs=outvector->domains_volume();
  if (is_nprocs!=test_nprocs) {
    throw(format("Incorrect nprocs tested on output: {} s/b {}",is_nprocs,test_nprocs));
  }
}

void test_distr_nprocs( kernel_function_args,int test_nprocs )
{
  auto invector = invectors.at(0);
  int is_nprocs;

  is_nprocs=outvector->domains_volume();
  if (is_nprocs!=test_nprocs)
    throw(format("Incorrect distribution nprocs tested on output: {} s/b {}",
		      is_nprocs,test_nprocs));

  is_nprocs=invector->domains_volume();
  if (is_nprocs!=test_nprocs)
    throw(format("Incorrect distribution nprocs tested on input: {} s/b {}",
		      is_nprocs,test_nprocs));

}

/*!
  Three point averaging with bump connections:
  the halo has the same size as the alpha domain
  \todo this ignores non-zero first index and such.
*/ 
void threepointsumbump( kernel_function_args ) {
  auto invector = invectors.at(0);
  const auto indistro = invector->get_distribution(),
    outdistro = outvector->get_distribution();

  if (outdistro->dimensionality()>1)
    throw(std::string("threepointsumbump: only for 1-d"));

  auto indata = invector->get_data(p),
    outdata = outvector->get_data(p);

  auto pstruct = outdistro->get_processor_structure(p);
  auto nstruct = outdistro->get_numa_structure();
  auto
    gstruct = outdistro->get_global_structure();
  domain_coordinate
    pfirst = pstruct->first_index_r(), plast = pstruct->last_index_r(),
    nfirst = nstruct->first_index_r(),
    gfirst = gstruct->first_index_r(), glast = gstruct->last_index_r(),
    nsize = nstruct->local_size_r(),
    offsets = nfirst-gfirst;

  index_int
    tar0 = outdistro->location_of_first_index(outdistro,p),
    src0 = indistro->get_numa_structure()
           ->linear_location_of(outdistro->get_processor_structure(p)),
    len = outdistro->volume(p);

  index_int
    gsize = outdistro->global_volume(),
    ingsize = indistro->global_size().coord(0);


  //snippet threepointsumbump
  int ilo = 0, ilen = len;
  if (pfirst==gfirst) { //(mfirst==gfirst) {
    // first element is globally first: the invector does not stick out to the left
    int i = 0; 
    outdata.at(tar0+i) = indata.at(src0+i) + indata.at(src0+i+1);
    ilo++;
  }
  if (plast==glast) { //(mlast==glast) {
     // local last is globally last; just compute and lower the length
    index_int i = ilen-1;
    outdata.at(tar0+i) = indata.at(src0+i-1) + indata.at(src0+i);
    // fmt::print("global last {} from {},{} giving {}\n",
    // 	       glast,indata[src0+i-1),indata[src0+i),outdata[tar0+i));
    ilen--;
  }
  for (index_int i=ilo; i<ilen; i++) {
    // regular case: the invector sticks out one to the left
    outdata.at(tar0+i) = indata.at(src0+i-1)+indata.at(src0+i)+indata.at(src0+i+1);
  }
  //snippet end
  *flopcount += 2*len;
}

