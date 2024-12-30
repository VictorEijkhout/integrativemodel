# -*- makefile -*-
################################################################
####
#### This make include file is part of the prototype implementation of
#### the Integrative Model for Parallelism
####
#### copyright Victor Eijkhout 2014-2022
####
#### Make.inc :
#### IMP settings for Victor's laptop
####
################################################################

IMPROOT = ${HOME}/Current/Ilib/i-mp/code
SYMUTIL = /bin/true
MPIRUN = mpirun
OPTLEVEL = 0

####
#### catch
####
CATCH2_COMPILE_OPTIONS = $$( pkg-config --cflags catch2 )
## -I${TACC_CATCH2_INC} 
CATCH2_LINK_OPTIONS    = $$( pkg-config --libs catch2 )
## -L${TACC_CATCH2_LIB} -lCatch2Main -lCatch2

####
#### Eigen
####
EIGEN_INC = ${HOME}/Installation/eigen/installation-3.3.7/include/eigen3/

####
#### Electric Fence
####
EFENCE_LIB = /opt/local/lib/libefence.a

####
#### fmtlib
####
FMT_COMPILE_OPTIONS = $$( pkg-config --cflags fmt )
## ${TACC_FMTLIB_INC}
FMT_LINK_OPTIONS = $$( pkg-config --libs fmt )
## ${TACC_FMTLIB_LIB}
## LIBFORMAT = libfmt.a

####
#### GSL
####
GSL_INC = ${TACC_MSGSL_INC}

COVERITYBIN = ${HOME}/Installation/coverity/bin

MPLDIR = ${HOME}/Installation/mpl/mpl

LIBCXX_intel = 
LIBCXX_gcc   = 
## -lc++
LIBCXX = ${LIBCXX_${TACC_FAMILY_COMPILER}}

####
#### OpenMP
####
INCOMP_clang = -I/opt/local/include/libomp
LIBOMP_clang =  -L/opt/local/lib/libomp -lomp
INCOMP = ${INCOMP_${TACC_FAMILY_COMPILER}}
LIBOMP = ${LIBOMP_${TACC_FAMILY_COMPILER}}

TEST_INCLUDES = \
    -DG -I${TACC_GSL_INC} \
    -I../imp \
    ${CATCH2_COMPILE_OPTIONS} \
    ${FMT_COMPILE_OPTIONS}
##     -I${EIGEN_INC} 
## for now removing: -I${GSL_INC} 
TEST_LIBS = \
    ${CATCH2_LINK_OPTIONS} \
    ${FMT_LINK_OPTIONS} \
    ${LIBCXX}

IMPCODEDIR = ../imp
OMPCODEDIR = ../omp
MPICODEDIR = ../mpi
