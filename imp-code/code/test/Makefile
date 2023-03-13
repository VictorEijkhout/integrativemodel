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

PROGRAMS = findit inttype vecarray

% : %.cxx
	g++ -o $@ -std=c++17 $^

clean ::
	@/bin/rm -f ${PROGRAMS}
