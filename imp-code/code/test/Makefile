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

PROGRAMS = construct1 inttype findit vecarray

CPPSTANDARD = 17
% : %.cxx
	g++ -o $@ -std=c++${CPPSTANDARD} $^

clean ::
	@/bin/rm -f ${PROGRAMS}
