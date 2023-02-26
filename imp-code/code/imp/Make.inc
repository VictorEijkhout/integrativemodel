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
CATCH2_COMPILE_OPTIONS = -I${LMOD_CATCH2_INC} 
CATCH2_LINK_OPTIONS = -L${LMOD_CATCH2_LIB} -lCatch2Main -lCatch2

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
CPPFORMAT_INC = ${LMOD_FMTLIB_INC}
CPPFORMAT_LIB = ${LMOD_FMTLIB_LIB}
LIBFORMAT = libfmt.a

####
#### GSL
####
GSL_INC = ${HOME}/Installation/gsl/gsl-lite/include

COVERITYBIN = ${HOME}/Installation/coverity/bin

MPLDIR = ${HOME}/Installation/mpl/mpl

LIBCXX_intel = 
LIBCXX_gcc   = -lc++
LIBCXX = ${LIBCXX_${LMOD_FAMILY_COMPILER}}

####
#### OpenMP
####
INCOMP_clang = -I/opt/local/include/libomp
LIBOMP_clang =  -L/opt/local/lib/libomp -lomp
INCOMP = ${INCOMP_${LMOD_FAMILY_COMPILER}}
LIBOMP = ${LIBOMP_${LMOD_FAMILY_COMPILER}}

TEST_INCLUDES = \
    -I${IMPCODEDIR} \
    ${CATCH2_COMPILE_OPTIONS} \
    -I${EIGEN_INC} -I${GSL_INC} -I${CPPFORMAT_INC}
TEST_LIBS = \
    ${CPPFORMAT_LIB}/${LIBFORMAT} \
    ${CATCH2_LINK_OPTIONS} \
    ${LIBCXX}

IMPCODEDIR = ../imp
OMPCODEDIR = ../omp
MPICODEDIR = ../mpi
