# -*- makefile -*-
################################################################
####
#### This makefile is part of the prototype implementation of
#### the Integrative Model for Parallelism
####
#### copyright Victor Eijkhout 2014-2024
####
#### Unit tests for the SEQ backend of IMP
####
################################################################

info ::
	@echo "SEQ simulation of an IMP backend code"
	@echo
	@echo "Available commands:"
	@echo

include ../imp/Make.inc
include ../imp/Make.common
include ../imp/Make.imp

MODE = seq

####
#### Unit tests
####
info ::
	@echo "make unittest_TYPE (choices: ${NEW_TESTS})"

##
## Gradually building up tests
##
unittest_main.o : \
    seq_env.h ${IMPCODEDIR}/imp_env.h \
    seq_static_vars.h ${IMPCODEDIR}/imp_static_vars.h
NEW_TESTS = 
RPATH = ${TACC_CATCH2_LIB} ${TACC_FMTLIB_LIB}
.PHONY: rpath
rpath :
	@if [ ! -z "${RPATH}" ] ; then \
	      for r in ${RPATH} ; do \
	          rpath="$$rpath -Wl,-rpath,$${r}" \
	      ; done \
	    ; fi \
	 && echo $${rpath}
.PHONY: link_unittest
link_unittest :
	@make --no-print-directory unittest_main.o ${UNITTEST}.o ECHO=${ECHO}
	@echo "Linking ${UNITTEST}"
	@cmdline="${CXX_LINKER} ${CXXFLAGS} -o ${UNITTEST} \
	    unittest_main.o ${UNITTEST}.o ${ENV_OBJS} ${TEST_LIBS}" \
	 && cmdline="$${cmdline} $$( make --no-print-directory rpath )" \
	 && if [ ! -z "${ECHO}" ] ; then echo "cmdline=$${cmdline}" ; fi \
	 && eval $${cmdline}

#
# environment
#
NEW_TESTS += unittest_env
ENV_OBJS := seq_env.o ${IMPCODEDIR}/imp_env.o ${IMPCODEDIR}/utils.o
unittest_env : unittest_main.o unittest_env.o ${ENV_OBJS}
	@make --no-print-directory link_unittest \
	  UNITTEST=$@ OBJS="${ENV_OBJS}" ECHO=${ECHO}
seq_env.o : seq_env.h ${IMPCODEDIR}/imp_env.h

#
# coordinate
#
NEW_TESTS += unittest_coord
COORD_OBJS = ${IMPCODEDIR}/imp_coord.o ${ENV_OBJS}
unittest_coord : unittest_main.o unittest_coord.o ${COORD_OBJS}
	@make --no-print-directory link_unittest \
	  UNITTEST=$@ OBJS="${COORD_OBJS}" ECHO=${ECHO}
unittest_coord.o : ${IMPCODEDIR}/imp_coord.h

#
# architecture and decomposition
#
NEW_TESTS += unittest_decomp
DECOMP_OBJS = seq_decomp.o ${IMPCODEDIR}/imp_decomp.o ${IMPCODEDIR}/indexstruct.o \
	${COORD_OBJS}
unittest_decomp : unittest_main.o unittest_decomp.o ${DECOMP_OBJS}
	@echo "Linking $@"
	${CXX_LINKER} ${CXXFLAGS} -o $@ ${SEQFLAGS} \
	    unittest_main.o $@.o ${DECOMP_OBJS} \
	    "$$( make --no-print-directory rpath )" ${TEST_LIBS}
unittest_decomp.o seq_decomp.o : seq_decomp.h ${IMPCODEDIR}/imp_decomp.h

#
# distribution
#
NEW_TESTS += unittest_distribution
DISTR_OBJS = seq_distribution.o ${IMPCODEDIR}/imp_distribution.o \
    ${DECOMP_OBJS}
unittest_distribution : unittest_main.o unittest_distribution.o ${DISTR_OBJS}
	@echo "Linking $@"
	${CXX_LINKER} ${CXXFLAGS} -o $@ ${SEQFLAGS} \
	    unittest_main.o $@.o ${DISTR_OBJS} \
	    ${TEST_LIBS}
unittest_distribution.o seq_distribution.o : seq_distribution.h ${IMPCODEDIR}/imp_distribution.h

#
# object
#
NEW_TESTS += unittest_object
OBJECT_OBJS = seq_object.o ${IMPCODEDIR}/imp_object.o \
    ${DISTR_OBJS}
unittest_object : unittest_main.o unittest_object.o ${OBJECT_OBJS}
	@echo "Linking $@"
	${CXX_LINKER} ${CXXFLAGS} -o $@ ${SEQFLAGS} \
	    unittest_main.o $@.o ${OBJECT_OBJS} \
	    ${TEST_LIBS}
unittest_object.o seq_object.o : seq_object.h ${IMPCODEDIR}/imp_object.h

#
# kernel
#
NEW_TESTS += unittest_kernel
KERNEL_OBJS = seq_kernel.o ${IMPCODEDIR}/imp_kernel.o ${IMPCODEDIR}/imp_functions.o \
    ${OBJECT_OBJS}
unittest_kernel : unittest_main.o unittest_kernel.o ${KERNEL_OBJS}
	@echo "Linking $@"
	${CXX_LINKER} ${CXXFLAGS} -o $@ ${SEQFLAGS} \
	    unittest_main.o $@.o ${KERNEL_OBJS} \
	    ${TEST_LIBS}
unittest_kernel.o seq_kernel.o : seq_kernel.h ${IMPCODEDIR}/imp_kernel.h

info ::
	@echo "make newtests : do new unit tests"
	@echo "    [ NP=... (default: ${NP}) ]"
.PHONY: newtests
NP = 4
newtests :
	for t in ${NEW_TESTS} ; do \
	    make $$t \
	     && echo && echo "Unittesting: $$t" && echo \
	     && ./$$t \
	; done

#
# old rules
#
${patsubst %,%.o,${UNITTESTS}} : \
    ${IMPCODEDIR}/unittest_functions.h ${IMPCODEDIR}/imp_functions.h \
    ${IMPCODEDIR}/indexstruct.hpp \
    seq_base.h ../imp/imp_base.h seq_ops.h ../imp/imp_ops.h seq_static_vars.h

# ${patsubst %,${IMPCODEDIR}/%.o,${UNITTESTS}} : \
#             ${patsubst %,${IMPCODEDIR}/%.cxx,${UNITTESTS}}
# 	cd ${IMPCODEDIR} && make $@

# this rule is slightly over generous
unittest_kmeans_main.o unittest_kmeans.o seq_apps/seq_kmeans_kernel.o : \
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
	  && ./$$u ${BASICBROKENOPTIONS}${BROKENOPTIONS}"~[hide]" \
	       2>&1 | grep "\(passed\|failed\)" \
	       > unittest.out 2>&1 \
	  && cat unittest.out

info ::
	@echo "make irtesting [IRFILTER=word]"
IRFILTER = ""
irtesting :
	/bin/rm -f *.ir ; \
	make unittest_ir && ./unittest_ir `if [ ! -z ${IRFILTER} ] ; then echo [${IRFILTER}] ; fi` && \
	  for i in *.ir ; do echo "==== $$i"; cat $$i ; echo ; done

####
#### Executables: done in seq_apps directory
####
info ::
	@echo "make applications : re-extract into seq_apps directory"
.PHONY: applications
EXECUTABLES := threepoint kmeans norm power cg
applications :
	for a in ${EXECUTABLES} ; do \
	  cat ${IMPCODEDIR}/template_$$a.cxx | sed s/IMP_/seq_/g > seq_apps/seq_$$a.cxx ; \
	  done

clean ::
	@echo " .. cleaning ${MODE} apps and tests" 
	@/bin/rm -f unittest ${UNITTESTS} ${NEW_TESTS}
	@cd seq_apps && make clean
