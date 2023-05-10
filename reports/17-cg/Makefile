################################################################
####
#### This makefile is part of the documentation of
#### the Integrative Model for Parallelism
####
#### copyright Victor Eijkhout 2014-6
####
#### 17 : Conjugate gradients
####
################################################################

include ../../Make.inc
include ../Make.inc
FILE = cg
INEXDIR = "../.."

%.o : %.c
	${CC} ${PETSC_CC_INCLUDES} -c $<
powerpetsc : powerpetsc.o
	${CLINKER} -o $@ $^
include ${PETSC_DIR}/lib/petsc/conf/variables

FILELONGNAME = IMP-17\ Conjugate\ gradients
upload :: copy_to_source_repo
