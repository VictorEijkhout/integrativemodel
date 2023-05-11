/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2023
 ****
 **** imp_kernel.cxx: Implementations of the kernel base classes
 ****
 ****************************************************************/

#include "imp_kernel.h"
#include <cassert>
#include <cmath>

using std::vector;
using std::shared_ptr;
using fmt::format;

template<int d>
kernel<d>::kernel( shared_ptr<object<d>> out )
  : output(out) {;
};

template class kernel<1>;
template class kernel<2>;
template class kernel<3>;

