# -*- Makefile -*-
################################################################
####
#### This makefile is part of the prototype implementation of
#### the Integrative Model for Parallelism
####
#### copyright Victor Eijkhout 2014-2025
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

CXXFLAGS = ${CXX_OPTFLAGS} ${OMPFLAGS} 
%.o : %.cxx
	@tgt=$@ && tgtdir=$(dir $$tgt ) \
	echo "Generic OMP compilation of $<" \
	 && ${CXX_COMPILER} ${CXXFLAGS} ${OMPFLAGS} ${CXXUNITFLAGS} ${TEST_INCLUDES} \
	    -c $<

####
#### Unit tests
####
info ::
	@echo "make unittest_TYPE (choices: ${NEW_TESTS})"

##
## Gradually building up tests
##
unittest_main.o : \
    omp_env.h ${IMPCODEDIR}/imp_env.h \
    omp_static_vars.h ${IMPCODEDIR}/imp_static_vars.h
NEW_TESTS = 
RPATH = ${TACC_CATCH2_LIB} ${TACC_FMTLIB_LIB}
.PHONY: rpath
rpath :
	@if [ ! -z "${RPATH}" ] ; then \
	      for r in ${RPATH} ; do \
	          rpath="$$rpath -Wl,-rpath,$${r}" \
	      ; done \
	    ; fi \
	 && echo $${rpath}
.PHONY: link_unittest
link_unittest :
	@make --no-print-directory ECHO=${ECHO} \
	    unittest_main.o ${UNITTEST}.o ${OBJS}
	@echo "Linking ${UNITTEST}"
	@cmdline="${CXX_LINKER} ${CXXFLAGS} -o ${UNITTEST} \
	    unittest_main.o ${UNITTEST}.o ${OBJS} ${TEST_LIBS}" \
	 && cmdline="$${cmdline} $$( make --no-print-directory rpath )" \
	 && if [ ! -z "${ECHO}" ] ; then echo "cmdline=$${cmdline}" ; fi \
	 && eval $${cmdline}

#
# environment
#
NEW_TESTS := unittest_env
ENV_OBJS := omp_env.o ${IMPCODEDIR}/imp_env.o ${IMPCODEDIR}/utils.o
unittest_env : ${ENV_OBJS}
	@make --no-print-directory link_unittest \
	  UNITTEST=$@ OBJS="${ENV_OBJS}" ECHO=${ECHO}
omp_env.o : omp_env.h ${IMPCODEDIR}/imp_env.h

#
# coordinate
#
NEW_TESTS += unittest_coord
COORD_OBJS = ${IMPCODEDIR}/imp_coord.o ${ENV_OBJS}
unittest_coord : unittest_main.o unittest_coord.o ${COORD_OBJS}
	@make --no-print-directory link_unittest \
	  UNITTEST=$@ OBJS="${COORD_OBJS}" ECHO=${ECHO}
unittest_coord.o : ${IMPCODEDIR}/imp_coord.h

#
# decomposition
#
NEW_TESTS += unittest_decomp
DECOMP_OBJS = omp_decomp.o ${IMPCODEDIR}/imp_decomp.o ${IMPCODEDIR}/indexstruct.o \
	${COORD_OBJS}
unittest_decomp : unittest_main.o unittest_decomp.o ${DECOMP_OBJS}
	@make --no-print-directory link_unittest \
	  UNITTEST=$@ OBJS="${DECOMP_OBJS}" ECHO=${ECHO}
omp_decomp.o : omp_decomp.h ${IMPCODEDIR}/imp_decomp.h

#
# distribution
#
NEW_TESTS += unittest_distribution
DISTR_OBJS = omp_distribution.o ${IMPCODEDIR}/imp_distribution.o \
    ${DECOMP_OBJS}
unittest_distribution : unittest_main.o unittest_distribution.o ${DISTR_OBJS}
	@make --no-print-directory link_unittest \
	  UNITTEST=$@ OBJS="${DISTR_OBJS}" ECHO=${ECHO}
omp_distribution.o : omp_distribution.h ${IMPCODEDIR}/imp_distribution.h

#
# object
#
NEW_TESTS += unittest_object
OBJ_OBJS = omp_object.o ${IMPCODEDIR}/imp_object.o \
    ${DISTR_OBJS}
unittest_object : unittest_main.o unittest_object.o ${OBJ_OBJS}
	@make --no-print-directory link_unittest \
	  UNITTEST=$@ OBJS="${OBJ_OBJS}" ECHO=${ECHO}
omp_object.o : omp_object.h ${IMPCODEDIR}/imp_object.h

info ::
	@echo "make newtests : do new unit tests"
	@echo "    [ NP=... (default: ${NP}) ]"
.PHONY: newtests
NP = 4
newtests :
	for t in ${NEW_TESTS} ; do \
	    make $$t \
	     && echo && echo "Unittesting: $$t" && echo \
	     && OMP_NUM_THREADS=${NP} $$t \
	; done

# ##
# ## stuff
# ##
# include ../imp/Make.regress
# info ::
# 	@echo "    [NP=nnnn default=${NP}]"
# NP = 4
# BASICBROKENOPTIONS="~[mask]~[embed]~[modulo]~[multi]~[sync]~[extend]"
# BROKENOPTIONS=
# REGRESSIONTESTS = unittest_distribution unittest_sparse unittest_struct unittest_collective unittest_ops unittest_nbody unittest_cg
# # unittest_operator unittest_ir unittest_ops unittest_cg"
# UNITTEST = unittest_struct
# unittest :
# 	@export u=${UNITTEST} \
# 	  && echo ; echo "================Testing" $$u ; echo \
# 	  && rm -f $$u ; make $$u >/dev/null 2>&1 \
# 	  && if [ ! -f $$u ] ; then echo "Could not make $$u"; exit 1; fi \
# 	  && OMP_NUM_THREADS=${NP} $$u ${BASICBROKENOPTIONS}${BROKENOPTIONS}"~[hide]" \
# 	       2>&1 | grep "\(passed\|failed\)" > unittest.out 2>&1 \
# 	  && cat unittest.out \
# 	  && ntests=$$(( ntests+1 ))

clean ::
	@/bin/rm -f unittest
