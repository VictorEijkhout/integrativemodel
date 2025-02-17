################################################################
####
#### This makefile is part of the prototype implementation of
#### the Integrative Model for Parallelism
####
#### copyright Victor Eijkhout 2014-2023
####
#### Mode-independent unit tests
####
################################################################

info ::
	@echo "Base code of an IMP backend"
	@echo "Available commands:"

include Make.inc
include Make.common
CPPSTANDARD = 20

##
## Library stuff
##
info ::
	@echo "make implib"
.PHONY: implib
IMP_OBJS = imp_env.o imp_coord.o
implib : ${IMP_OBJS}
	${CXX_COMPILER} -o libimp.dylib \
	    -dynamiclib -Wl,-undefined -Wl,dynamic_lookup \
	    ${IMP_OBJS}
imp_entity.o : imp_entity.h
imp_env.o : imp_env.h
imp_coord.o : imp_coord.h

##
## Unittests, independent of any mode
##

UNITTESTS = unittest_indexstruct
.SECONDEXPANSION:
unittest_indexstruct : unittest_indexstruct.o unittest_main.o indexstruct.o imp_coord.o
	${CXX_LINKER} ${CXXFLAGS} ${CXXUNITFLAGS} \
	    -o $@ $^ \
	    ${TEST_LIBS}
unittest_indexstruct.o indexstruct.o : indexstruct.hpp
unittest_test : unittest_test.o unittest_main.o indexstruct.o imp_coord.o
	${CXX_LINKER} ${CXXFLAGS} ${CXXUNITFLAGS} \
	    -o $@ $^ \
	    ${TEST_LIBS}
unittest_test.o : indexstruct.hpp

include ../imp/Make.regress
REGRESSIONTESTS = unittest_indexstruct
unittest :
	@export u=${UNITTEST} ; \
	  echo "\n================\nTesting" $$u ; \
	  rm -f $$u ; make $$u >/dev/null 2>&1 ; \
	  if [ ! -f $$u ] ; then echo "Could not make $$u"; exit 1; fi ; \
	  ./$$u 2>&1 | grep "\(passed\|failed\)" > unittest.out 2>&1 ; \
	  cat unittest.out

imp_functions.o : def_out.cxx impfunc_struct_index.cxx
