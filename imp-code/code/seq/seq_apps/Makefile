################################################################
####
#### This makefile is part of the prototype implementation of
#### the Integrative Model for Parallelism
####
#### copyright Victor Eijkhout 2014-2024
####
#### SEQ applications
####
################################################################

info ::
	@echo "SEQ applications in the IMP model"
	@echo "Available commands:"

MODE = seq
info ::
	@echo "make EXECUTABLE (where EXECUTABLES=${EXECUTABLES})"
EXECUTABLES := power cg cg0 cgr cgm gropp nbody heat lulesh-2d lulesh kmeans sstep \
    laplace-bilinear grids balance

.SECONDEXPANSION:
include ../../imp/Make.inc
include ../../imp/Make.common
IMPCODEDIR = ../../imp
SEQCODEDIR = ../../seq
OMPCODEDIR = ../../omp
MPICODEDIR = ../../mpi
include ../../imp/Make.imp
include ../../imp/Make.apps

CXXFLAGS = ${CXX_OPTFLAGS} -I. -I.. -I../../imp \
    -I${LMOD_FMTLIB_INC} -I${GSL_INC}
OPTLEVEL = 2
%.o : %.cxx
	${MPICXX} ${CXXFLAGS} ${CXXUNITFLAGS} \
	    -I${CATCHDIR} \
	    `if [ "$@" = "balance.o" ] ; then echo "-I${EIGEN_INC}" ; fi` \
	    -c $<

EXECUTABLE_SOURCES := ${patsubst %,%.cxx,${EXECUTABLES}}
EXECUTABLE_OBJECTS := ${patsubst %,%.o,${EXECUTABLES}}

EXECUTABLE_OBJS = \
    ${SEQCODEDIR}/seq_base.o ${SEQCODEDIR}/seq_functions.o ${APPS_OBJS}
${EXECUTABLES} :: $$@.cxx $$@.o ${EXECUTABLE_OBJS}
	${MPICXX} ${CXXFLAGS} -o $@ ${MPIFLAGS} $@.o \
	    ${EXECUTABLE_OBJS} ${SPECIFIC_OBJECTS_$@} \
	    ${LMOD_FMTLIB_LIB}/${LIBFORMAT} ${LIBCXX}

# non-trivial dependencies
${EXECUTABLE_OBJECTS} ../seq_base.o :: \
    ../seq_base.h ../../imp/imp_base.h \
    ../seq_ops.h ../../imp/imp_ops.h \
    ../../imp/utils.h ../../imp/template_common_header.h ${MODE}_specific_header.h
SPECIFIC_OBJECTS_cg := ${SEQCODEDIR}/seq_apps/seq_cg_kernel.o

##
## for regression testing
##
.PHONY: runapp
NP = 4
info ::
	@echo "     [NP=nnn, default=${NP}]"
APP  = heat
runapp :
	@./${APP} -summary

clean ::
	@/bin/rm -f ${EXECUTABLES} ${EXECUTABLE_SOURCES} *.stf.* *.o[0-9]* \
	  *.itapr *.prot *.stf stampedetest.o* vncserver.out
	@/bin/rm -rf *.dSYM
