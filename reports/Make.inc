# -*- makefile -*-
################################################################
####
#### This make include file is part of the documentation of
#### the Integrative Model for Parallelism
####
#### copyright Victor Eijkhout 2014-9
####
#### series/Make.inc : make rules specific for the report series
####
################################################################

info ::
	@echo "make ${FILE}"

DOCTYPE = paper
EXTRATEXINPUTS = :..:../..:../../everything:../../graphics:../../graphs:${HOME}/Current/istc/hpc-book-private:${HOME}/Current/istc/hpc-book-private/graphics
EXTRABSTINPUTS = 
UPLOADEXTRADIR = series

info ::
	@echo "make clean"
.PHONY: clean
clean ::
	@/bin/rm -f *~ *.{aux,blg,ind,log,nav}

info ::
	@echo "make snippets"
.PHONY: snippets
snippets :
	cd ../.. ; make snippets

REPO_REPORT_DIR = ${HOME}/Current/Ilib/imp-code/docs
copy_to_source_repo :
	@if [ -z "${FILELONGNAME}" ] ; then \
	  echo "Need FILELONGNAME for copy to source repo" ; exit 1 ; fi
	@cp ${FILE}.pdf ${REPO_REPORT_DIR}/${FILELONGNAME}.pdf
