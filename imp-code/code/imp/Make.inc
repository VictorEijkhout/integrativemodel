# -*- makefile -*-
################################################################
####
#### This make include file is part of the prototype implementation of
#### the Integrative Model for Parallelism
####
#### copyright Victor Eijkhout 2014-8
####
#### Make.inc :
#### IMP settings for Victor's laptop
####
################################################################

IMPROOT = ${HOME}/Current/Ilib/i-mp/code
SYMUTIL = /bin/true
MPIRUN = mpirun
OPTLEVEL = 0

MPICXX = mpicxx
TACC_FAMILY_COMPILER = clang

####
#### catch
####
CATCHDIR = ${HOME}/Installation/catch/Catch2/single_include/catch2

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
CPPFORMAT_INC = ${HOME}/Installation/fmtlib/installation/include
CPPFORMAT_LIB = ${HOME}/Installation/fmtlib/installation/lib
LIBFORMAT = libfmt.a

####
#### GSL
####
GSL_INC = ${HOME}/Installation/gsl/gsl-lite/include

COVERITYBIN = ${HOME}/Installation/coverity/bin

MPLDIR = ${HOME}/Installation/mpl/mpl

LIBCXX_intel = 
LIBCXX_gcc   = -lc++
LIBCXX = ${LIBCXX_${TACC_FAMILY_COMPILER}}

####
#### OpenMP
####
INCOMP_clang = -I/opt/local/include/libomp
LIBOMP_clang =  -L/opt/local/lib/libomp -lomp
INCOMP = ${INCOMP_${TACC_FAMILY_COMPILER}}
LIBOMP = ${LIBOMP_${TACC_FAMILY_COMPILER}}
