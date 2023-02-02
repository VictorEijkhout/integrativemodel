/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2023
 ****
 **** basic utilities
 ****
 ****************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "utils.h"

std::function< int(int,int) > ipower =
  [] (int base,int power) -> int {
    auto iipower =
      [] ( auto&& self, int base,int power ) -> int {
	if (power==0) return 1;
	else return base * self(self,base,power-1);
      };
    return iipower(iipower,base,power);
  };

bool hasarg_from_argcv(const char *name,int nargs,char **the_args) {
  for (int iarg=1; iarg<nargs; ++iarg) {
    char *arg = the_args[iarg];
    if (arg[0]=='-' && !strcmp(arg+1,name)) {
      return iarg;
    }
  }
  return 0;
}

/*! Return an integer commandline argument with default value
*/
int iarg_from_argcv(const char *name,int vdef,int nargs,char **the_args) {
  int value = vdef;
  for (int iarg=0; iarg<nargs; iarg++) {
    char *arg = the_args[iarg];
    if (arg[0]=='-' && !strcmp(arg+1,name) && iarg+1<nargs) {
      value = atoi(the_args[iarg+1]);
      //printf("found <<%s>> as %d\n",name,value);
      break;
    }
  }
  return value;
}

/*! Return a string commandline argument with default value

  \todo restore this implementation
*/
const char* sarg_from_argcv(const char *name,const char *vdef,int nargs,char **the_args) {
  return vdef;
}
