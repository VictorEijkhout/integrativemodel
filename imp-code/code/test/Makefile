# -*- makefile -*-
################################################################
####
#### This makefile is part of the prototype implementation of
#### the Integrative Model for Parallelism
####
#### copyright Victor Eijkhout 2014-2023
####
#### makefile for small tests
####
################################################################

info ::

include ../imp/Make.common
include ../imp/Make.inc

PROGRAMS = construct1 inner inttype findit vecarray
CATCHES =  requirethrow

CPPSTANDARD = 17
% : %.cxx
	g++ -o $@ -std=c++${CPPSTANDARD} $^
.SECONDEXPANSION:
${CATCHES} : $$@.cxx
	g++ -g -o$@ -std=c++20 $^ \
	    -I${TACC_CATCH2_INC} \
	    -L${TACC_CATCH2_LIB} -lCatch2Main -lCatch2

clean ::
	@/bin/rm -f ${PROGRAMS} ${CATCHES}

