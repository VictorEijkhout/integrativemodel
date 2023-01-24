# -*- Makefile -*-
################################################################
####
#### This makefile is part of the prototype implementation of
#### the Integrative Model for Parallelism
####
#### copyright Victor Eijkhout 2014-2023
####
#### Unit tests for the OpenMP backend of IMP
####
################################################################
info ::
	@echo "OpenMP simulation of an IMP backend code"
	@echo
	@echo "Available commands:"
	@echo

include ../imp/Make.inc
include ../imp/Make.common
IMPCODEDIR = ../imp
OMPCODEDIR = ../omp
MPICODEDIR = ../mpi
include ../imp/Make.imp

CXXFLAGS = ${CXX_OPTFLAGS} -I${IMPCODEDIR} -I${CPPFORMAT_INC} \
    -I${LMOD_FMTLIB_INC} -I${GSL_INC}
CXXLIBS = ${CPPFORMAT_LIB}/${LIBFORMAT} \
	  -L${LMOD_CATCH2_LIB} -lCatch2Main -lCatch2 \
	  ${LIBCXX}
%.o : %.cxx
	${CXX_COMPILER} ${CXXFLAGS} ${OMPFLAGS} ${CXXUNITFLAGS} \
	    -I${LMOD_CATCH2_INC} -I${EIGEN_INC} -I${GSL_INC} \
	    -c $<

####
#### Unit tests
####
info ::
	@echo "make unittest_TYPE (choices: ${NEW_TESTS})"
	@echo "    unittest_env"
	@echo "    unittest_decomp"

unittest_main.o : \
    omp_env.h ${IMPCODEDIR}/imp_env.h \
    omp_static_vars.h ${IMPCODEDIR}/imp_static_vars.h
NEW_TESTS = 

#
# environment
#
NEW_TESTS := unittest_env
ENV_OBJS := omp_env.o ${IMPCODEDIR}/imp_env.o
unittest_env : unittest_main.o unittest_env.o ${ENV_OBJS}
	@echo "Linking $@"
	${CXX_LINKER} ${CXXFLAGS} ${CXXUNITFLAGS} -o $@ ${OMPFLAGS} \
	  unittest_main.o $@.o ${ENV_OBJS} \
	${CXXLIBS}
omp_env.o : omp_env.h ${IMPCODEDIR}/imp_env.h
${IMPCODEDIR}/imp_env.o : ${IMPCODEDIR}/imp_env.cxx ${IMPCODEDIR}/imp_env.h
	cd ${IMPCODEDIR} && make imp_env.o

#
# coordinate
#
NEW_TESTS += unittest_coord
COORD_OBJS = ${IMPCODEDIR}/imp_coord.o ${ENV_OBJS}
unittest_coord : unittest_main.o unittest_coord.o ${COORD_OBJS}
	@echo "Linking $@"
	${CXX_LINKER} ${CXXFLAGS} ${CXXUNITFLAGS} -o $@ ${OMPFLAGS} \
	  unittest_main.o $@.o ${COORD_OBJS} \
	  ${CXXLIBS}
${IMPCODEDIR}/imp_coord.o : ${IMPCODEDIR}/imp_coord.cxx ${IMPCODEDIR}/imp_coord.h
	cd ${IMPCODEDIR} && make imp_coord.o

#
# decomposition
#
NEW_TESTS += unittest_decomp
DECOMP_OBJS = omp_decomp.o ${IMPCODEDIR}/imp_decomp.o ${IMPCODEDIR}/indexstruct.o \
	${COORD_OBJS}
unittest_decomp : unittest_main.o unittest_decomp.o ${DECOMP_OBJS}
	@echo "Linking $@"
	${OMPCXX} ${CXXFLAGS} ${CXXUNITFLAGS} -o $@ ${OMPFLAGS} \
	  unittest_main.o $@.o ${DECOMP_OBJS} \
	  ${CXXLIBS}
omp_decomp.o : omp_decomp.h ${IMPCODEDIR}/imp_decomp.h
${IMPCODEDIR}/imp_decomp.o : ${IMPCODEDIR}/imp_decomp.cxx ${IMPCODEDIR}/imp_decomp.h
	@cd ${IMPCODEDIR} && make --no-print-directory imp_decomp.o

##
## functions
##
omp_unittest_functions.o : ../imp/unittest_functions.h
${patsubst %,%.o,${UNITTESTS}} : \
  ../imp/imp_base.h ../imp/imp_ops.h \
  ../imp/unittest_functions.h \
  omp_base.h omp_ops.h omp_static_vars.h

${patsubst %,%.o,${UNITTESTS}} : omp_base.h ../imp/imp_base.h \
    ../imp/unittest_functions.h omp_static_vars.h

EXECUTABLES = threepoint 

.SECONDEXPANSION:
# threepoint executable
${EXECUTABLES} : \
    ../imp/imp_base.o ../imp/utils.o \
    threepoint_kernel.o threepoint_local.o \
    $$@.o omp_base.o 
	${CXX_LINKER} -o $@ ${OMPFLAGS} $^
	@${SYMUTIL} $@

# non-trivial dependencies
threepoint.cxx : ../imp/template_threepoint.cxx
	cat $< | sed s/IMP_/omp_/g > $@
threepoint.o omp_base.o : omp_base.h ../imp/imp_base.h ../imp/utils.h
clean ::
	@/bin/rm -f ${EXECUTABLES} ${EXECUTABLE_SOURCES}

include ../imp/Make.regress
info ::
	@echo "    [NP=nnnn default=${NP}]"
NP = 4
BASICBROKENOPTIONS="~[mask]~[embed]~[modulo]~[multi]~[sync]~[extend]"
BROKENOPTIONS=
REGRESSIONTESTS = unittest_distribution unittest_sparse unittest_struct unittest_collective unittest_ops unittest_nbody unittest_cg
# unittest_operator unittest_ir unittest_ops unittest_cg"
UNITTEST = unittest_struct
unittest :
	@export u=${UNITTEST} \
	  && echo ; echo "================Testing" $$u ; echo \
	  && rm -f $$u ; make $$u >/dev/null 2>&1 \
	  && if [ ! -f $$u ] ; then echo "Could not make $$u"; exit 1; fi \
	  && OMP_NUM_THREADS=${NP} $$u ${BASICBROKENOPTIONS}${BROKENOPTIONS}"~[hide]" \
	       2>&1 | grep "\(passed\|failed\)" > unittest.out 2>&1 \
	  && cat unittest.out \
	  && ntests=$$(( ntests+1 ))

clean ::
	@/bin/rm -f unittest
