/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-9
 ****
 **** MPI implementations of the support functions
 ****
 ****************************************************************/

#include <stdlib.h>

#include "imp_base.h"
#include "unittest_functions.h"

//! \todo eliminate mention of distribution in these unittest functions
void vecset( kernel_function_args )
{
  const auto outdistro = outvector->get_distribution();

  auto outdata = outvector->get_data(p);
  // } catch (std::string c) { fmt::print("Error <<{}>> in vecset\n",c);
  //   throw(fmt::format("vecset of object <<{}>> failed",outvector->get_name())); };

  index_int n = outdistro->volume(p);

  for (index_int i=0; i<n; i++) {
    outdata.at(i) = 1.;
  }
  *flopcount += n;
}

//snippet mpishiftleft
void vecshiftleftmodulo( kernel_function_args )
{
  auto invector = invectors.at(0);
  const auto indistro = invector->get_distribution(),
    outdistro = outvector->get_distribution();

  auto indata = invector->get_data(p),
    outdata = outvector->get_data(p);

  index_int
    tar0 = outdistro->first_index_r(p).coord(0)-outdistro->numa_first_index().coord(0),
    src0 = indistro->first_index_r(p).coord(0)-indistro->numa_first_index().coord(0),
    len = outdistro->volume(p);

  for (index_int i=0; i<len; i++) {
      outdata.at(tar0+i) = indata.at(src0+i+1);
  }
  *flopcount += len;
}
//snippet end

void vecshiftrightmodulo( kernel_function_args )
{
  auto invector = invectors.at(0);
  const auto indistro = invector->get_distribution(),
    outdistro = outvector->get_distribution();

  auto indata = invector->get_data(p);
  auto outdata = outvector->get_data(p);

  index_int n = outdistro->volume(p); // the halo is one more

  for (index_int i=0; i<n; i++) {
    outdata.at(i) = indata.at(i);
  }
  *flopcount += n;
}

//! \todo make the void an explicit integer
void ksumming( kernel_function_args,void *ctx )
{
  auto invector = invectors.at(0);
  const auto indistro = invector->get_distribution(),
    outdistro = outvector->get_distribution();

  int *k = (int*)ctx;
  auto outdata = outvector->get_data(p);
  auto indata = invector->get_data(p);
  index_int n = indistro->volume(p);
  for (int ik=0; ik<(*k); ik++) {
    double s = 0;
    for (index_int i=0; i<n; i++) {
      double in = indata.at(ik+i*(*k));
      s += in;
    } 
    outdata.at(ik) = s;
  }
  *flopcount += (*k)*n;
}

void threepointsummod( kernel_function_args )
{
  auto invector = invectors.at(0);
  const auto indistro = invector->get_distribution(),
    outdistro = outvector->get_distribution();

  auto outdata = outvector->get_data(p);
  auto indata = invector->get_data(p);
  index_int n = outdistro->volume(p);
  double s = 0;
//snippet mpi3pmod
  for (index_int i=0; i<n; i++)
    outdata.at(i) = indata.at(i)+indata.at(i+1)+indata.at(i+2);
//snippet end
  *flopcount += n;
}

/*
 * Auxiliary stuff
 */
int pointfunc33(int i,int my_first) {return my_first+i;}
