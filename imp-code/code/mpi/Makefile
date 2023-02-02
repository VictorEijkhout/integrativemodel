# -*- makefile -*-
################################################################
####
#### This makefile is part of the prototype implementation of
#### the Integrative Model for Parallelism
####
#### copyright Victor Eijkhout 2014-2022
####
#### Unit tests for the MPI backend of IMP
####
################################################################
info ::
	@echo "MPI simulation of an IMP backend code"
	@echo
	@echo "Available commands:"
	@echo

include ../imp/Make.inc
include ../imp/Make.common
IMPCODEDIR = ../imp
OMPCODEDIR = ../omp
MPICODEDIR = ../mpi
## include ../imp/Make.imp

CXXFLAGS = ${CXX_OPTFLAGS} -I${IMPCODEDIR} -I${CPPFORMAT_INC}
%.o : %.cxx
	@tgt=$@ && tgtdir=$(dir $$tgt ) \
	echo "Generic MPI compilation of $<" \
	 && ${MPICXX} ${CXXFLAGS} ${MPIFLAGS} ${CXXUNITFLAGS} \
	    -I${LMOD_CATCH2_INC} -I${EIGEN_INC} -I${GSL_INC} \
	    -c $<

LIBCXX = -lstdc++

####
#### Unit tests
####
info ::
	@echo "make unittest_TYPE (choices: ${UNITTESTS})"

##
## Gradually building up tests
##
unittest_main.o : \
    mpi_env.h ${IMPCODEDIR}/imp_env.h \
    mpi_static_vars.h ${IMPCODEDIR}/imp_static_vars.h
NEW_TESTS = 

info ::
	@echo "Test from the group up:"
	@echo "    unittest_env"
	@echo "    unittest_decomp"
#
# environment
#
NEW_TESTS := unittest_env
ENV_OBJS := mpi_env.o ${IMPCODEDIR}/imp_env.o ${IMPCODEDIR}/utils.o
unittest_env : unittest_main.o unittest_env.o ${ENV_OBJS}
	@echo "Linking $@"
	${MPICXX} ${CXXFLAGS} ${CXXUNITFLAGS} -o $@ ${MPIFLAGS} \
	  unittest_main.o $@.o \
	  ${ENV_OBJS} \
	  ${CPPFORMAT_LIB}/${LIBFORMAT} \
	  -L${LMOD_CATCH2_LIB} -lCatch2Main -lCatch2 \
	  ${LIBCXX}
mpi_env.o : mpi_env.h ${IMPCODEDIR}/imp_env.h
${IMPCODEDIR}/imp_env.o : ${IMPCODEDIR}/imp_env.cxx ${IMPCODEDIR}/imp_env.h
	cd ${IMPCODEDIR} && make imp_env.o
${IMPCODEDIR}/utils.o : ${IMPCODEDIR}/utils.h
	cd ${IMPCODEDIR} && make utils.o

#
# coordinate
#
NEW_TESTS += unittest_coord
COORD_OBJS = ${IMPCODEDIR}/imp_coord.o ${ENV_OBJS}
unittest_coord : unittest_main.o unittest_coord.o ${COORD_OBJS}
	@echo "Linking $@"
	${MPICXX} ${CXXFLAGS} ${CXXUNITFLAGS} -o $@ ${MPIFLAGS} \
	  unittest_main.o $@.o \
	  ${COORD_OBJS} \
	  ${CPPFORMAT_LIB}/${LIBFORMAT} \
	  -L${LMOD_CATCH2_LIB} -lCatch2Main -lCatch2 \
	  ${LIBCXX}
${IMPCODEDIR}/imp_coord.o : ${IMPCODEDIR}/imp_coord.cxx ${IMPCODEDIR}/imp_coord.h
	cd ${IMPCODEDIR} && make imp_coord.o

#
# architecture and decomposition
#
NEW_TESTS += unittest_decomp
DECOMP_OBJS = mpi_decomp.o ${IMPCODEDIR}/imp_decomp.o ${IMPCODEDIR}/indexstruct.o \
	${COORD_OBJS}
unittest_decomp : unittest_main.o unittest_decomp.o ${DECOMP_OBJS}
	@echo "Linking $@"
	${MPICXX} ${CXXFLAGS} ${CXXUNITFLAGS} -o $@ ${MPIFLAGS} \
	  unittest_main.o $@.o \
	  ${DECOMP_OBJS} \
	  ${CPPFORMAT_LIB}/${LIBFORMAT} \
	  -L${LMOD_CATCH2_LIB} -lCatch2Main -lCatch2 \
	  ${LIBCXX}
mpi_decomp.o : mpi_decomp.h ${IMPCODEDIR}/imp_decomp.h
${IMPCODEDIR}/imp_decomp.o : ${IMPCODEDIR}/imp_decomp.cxx ${IMPCODEDIR}/imp_decomp.h
	@cd ${IMPCODEDIR} && make --no-print-directory imp_decomp.o

#
# distribution
#
NEW_TESTS += unittest_distribution
DISTR_OBJS = mpi_distribution.o ${IMPCODEDIR}/imp_distribution.o \
    ${DECOMP_OBJS}
unittest_distribution : unittest_main.o unittest_distribution.o ${DISTR_OBJS}
	@echo "Linking $@"
	${MPICXX} ${CXXFLAGS} ${CXXUNITFLAGS} -o $@ ${MPIFLAGS} \
	  unittest_main.o $@.o \
	  ${DISTR_OBJS} \
	  ${CPPFORMAT_LIB}/${LIBFORMAT} \
	  -L${LMOD_CATCH2_LIB} -lCatch2Main -lCatch2 \
	  ${LIBCXX}
mpi_distribution.o : mpi_distribution.h ${IMPCODEDIR}/imp_distribution.h
${IMPCODEDIR}/imp_distribution.o : ${IMPCODEDIR}/imp_distribution.cxx ${IMPCODEDIR}/imp_distribution.h
	@cd ${IMPCODEDIR} && make --no-print-directory imp_distribution.o


${IMPCODEDIR}/indexstruct.o : ${IMPCODEDIR}/indexstruct.cxx ${IMPCODEDIR}/indexstruct.hpp
	@cd ${IMPCODEDIR} && make --no-print-directory indexstruct.o

info ::
	@echo "make newtests : do new unit tests"
.PHONY: newtests
newtests :
	for t in ${NEW_TESTS} ; do \
	    make $$t \
	     && echo && echo "Unittesting: $$t" && echo \
	     && mpiexec -n 4 $$t \
	; done

#
# old rules
#
${patsubst %,%.o,${UNITTESTS}} : \
    ${IMPCODEDIR}/unittest_functions.h ${IMPCODEDIR}/imp_functions.h \
    ${IMPCODEDIR}/indexstruct.hpp \
    mpi_base.h ../imp/imp_base.h mpi_ops.h ../imp/imp_ops.h mpi_static_vars.h

# ${patsubst %,${IMPCODEDIR}/%.o,${UNITTESTS}} : \
#             ${patsubst %,${IMPCODEDIR}/%.cxx,${UNITTESTS}}
# 	cd ${IMPCODEDIR} && make $@

# this rule is slightly over generous
unittest_kmeans_main.o unittest_kmeans.o mpi_apps/mpi_kmeans_kernel.o : \
    ${IMPCODEDIR}/kmeans_functions.h

include ../imp/Make.regress
NP = 4
BASICBROKENOPTIONS = "~[mask]~[reuse]~[bin]~[modulo]~[embed]~[extend]"
REGRESSIONTESTS = unittest_proc unittest_distribution unittest_sparse \
    unittest_struct unittest_tree unittest_over \
    unittest_collective unittest_operator unittest_ops \
    unittest_nbody unittest_cg unittest_lulesh unittest_kmeans
# unittest_over 
info ::
	@echo "make unittest UNITTEST=..."
UNITTEST = unittest_struct
unittest :
	@export u=${UNITTEST} \
	  && echo && echo "================Testing" $$u && echo \
	  && rm -f $$u ; make $$u >/dev/null 2>&1 \
	  && if [ ! -f $$u ] ; then echo "Could not make $$u"; exit 1; fi \
	  && ${MPIRUN} -np ${NP} $$u ${BASICBROKENOPTIONS}${BROKENOPTIONS}"~[hide]" \
	       2>&1 | grep "\(passed\|failed\)" \
	       > unittest.out 2>&1 \
	  && cat unittest.out

info ::
	@echo "make irtesting [IRFILTER=word]"
IRFILTER = ""
irtesting :
	/bin/rm -f *.ir ; \
	make unittest_ir && mpirun -np 4 unittest_ir `if [ ! -z ${IRFILTER} ] ; then echo [${IRFILTER}] ; fi` && \
	  for i in *.ir ; do echo "==== $$i"; cat $$i ; echo ; done

####
#### Executables: done in mpi_apps directory
####
info ::
	@echo "make applications : re-extract into mpi_apps directory"
.PHONY: applications
EXECUTABLES := threepoint kmeans norm power cg
applications :
	for a in ${EXECUTABLES} ; do \
	  cat ${IMPCODEDIR}/template_$$a.cxx | sed s/IMP_/mpi_/g > mpi_apps/mpi_$$a.cxx ; \
	  done

clean ::
	@/bin/rm -f unittest ${UNITTESTS} ${NEW_TESTS}
	@cd mpi_apps ; make clean
