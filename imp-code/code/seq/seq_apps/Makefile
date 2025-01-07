################################################################
####
#### This makefile is part of the prototype implementation of
#### the Integrative Model for Parallelism
####
#### copyright Victor Eijkhout 2014-2025
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
EXECUTABLES := 

.SECONDEXPANSION: 
include ../../imp/Make.inc
include ../../imp/Make.common
IMPCODEDIR = ../../imp
SEQCODEDIR = ../../seq
OMPCODEDIR = ../../omp
MPICODEDIR = ../../mpi
include ../../imp/Make.imp
include ../../imp/Make.apps

CXX = mpicxx
CXXFLAGS = ${CXX_OPTFLAGS} -I. -I.. -I../../imp -DIMPisSEQ -I${TACC_FMTLIB_INC}
OPTLEVEL = 2
%.o : %.cxx
	${CXX} ${CXXFLAGS} ${SEQFLAGS} ${CXXUNITFLAGS} -I${CATCHDIR} -c $<

EXECUTABLE_SOURCES := ${patsubst %,%.cxx,${EXECUTABLES}}
EXECUTABLE_OBJECTS := ${patsubst %,%.o,${EXECUTABLES}}

EXECUTABLE_OBJS = \
    ${SEQCODEDIR}/seq_base.o ${SEQCODEDIR}/seq_functions.o ${APPS_OBJS}
${EXECUTABLES} :: $$@.cxx $$@.o ${EXECUTABLE_OBJS}
	${CXX} ${CXXFLAGS} -o $@ ${SEQFLAGS} $@.o \
	    ${EXECUTABLE_OBJS} ${SPECIFIC_OBJECTS_$@} \
	    ${TACC_FMTLIB_LIB}/${LIBFORMAT} ${LIBCXX}

# non-trivial dependencies
${EXECUTABLE_OBJECTS} ../seq_base.o : \
    ../seq_base.h ../../imp/imp_base.h \
    ../seq_ops.h ../../imp/imp_ops.h \
    ../../imp/utils.h ../../imp/template_common_header.h ${MODE}_specific_header.h

##
## for regression testing
##
.PHONY: runapp
APP  = heat
NTHREADS = 4
info ::
	@echo "     [NTHREADS=nnn, default=${NTHREADS}]"
runapp :
	@OMP_NUM_THREADS=${NTHREADS} ./${APP} -summary

clean ::
	@/bin/rm -f ${EXECUTABLES} ${EXECUTABLE_SOURCES}
