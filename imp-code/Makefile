# -*- makefile -*-
################################################################
####
#### This makefile is part of the prototype implementation of
#### the Integrative Model for Parallelism
####
#### copyright Victor Eijkhout 2014-2023
####
#### Root level makefile for IMP
####
################################################################

info ::

include code/imp/Make.inc

####
#### documentation : use doxygen to make LaTeX & html documentation
####
.PHONY: documentation
info ::
	@echo "make documentation"
documentation:
	for m in mpi omp product ; do ( cd code/$${m}/$${m}_apps ; make extract ) ; done
	doxygen docs/doxy.conf
	cd docs/latex ; \
	  sed -i -e '/input.class/s/$$/\\vfill\\pagebreak/' refman.tex && \
	  for t in *.tex ; do \
	    cp $$t tmp.tex && \
	    awk -f ../../docs/doxycode.awk tmp.tex > $$t ; \
	  done && \
	  pdflatex refman && pdflatex refman && pdflatex refman
	make upload
#cp ../docs/latex/refman.pdf ../docs
mergecode :
	  for t in *.tex ; do \
	    cp $$t tmp.tex && \
	    awk -f ../../docs/doxycode.awk tmp.tex > $$t ; \
	  done && \
	  [ 1 == 1 ]
clean ::
	@cd docs/html && ls | xargs rm -rf
	@cd docs/latex && ls | xargs rm -rf

####
#### upload : if you are victor, upload docs to dropbox
####
info ::
	@if [ `whoami` = "eijkhout" ] ; then \
	  echo "make upload" ; fi
.PHONY: upload
upload :
	-@cd docs \
	 && ln -f latex/refman.pdf .. \
	 && if [ `whoami` = "eijkhout" ] ; then \
	      echo "Uploading refman.pdf to Dropbox" \
	       && mkdir -p ${HOME}/DropBox/Scicomp/IMP \
	       && cp latex/refman.pdf ${HOME}/DropBox/Scicomp/IMP \
	    ; fi
notarget :
	  for f in *.tex ; do \
	    sed -e 's/\[H\]/[htp]/' -ibak $$f ; done ; \
	  pdflatex refman ; pdflatex refman


info ::
	@echo "make clean"
clean ::
	@rm -f *~
	@cd code && make clean
