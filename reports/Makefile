################################################################
####
#### This makefile is part of the documentation of
#### the Integrative Model for Parallelism
####
#### copyright Victor Eijkhout 2014/5
####
#### Makefile for reports series
####
################################################################

info ::
	@echo "Available make targets:"

info ::
	@echo "make pdf: format all reports"
.PHONY: pdf
pdf :
	@for d in * ; do \
	  if [ -d "$$d" ] ; then \
	    echo "Formatting $$d" ; \
	    ( cd $$d ; make pdf ) ; \
	  fi ; \
	done

info ::
	@echo "make upload"
.PHONY: upload
upload : bib
	@for d in * ; do \
	  if [ -d "$$d" ] ; then \
	    echo "uploading in `pwd`/$$d" ; \
	    ( cd $$d ; make upload ) ; \
	  fi ; \
	done
info ::
	@echo "make bib : copy imp.bib to BIBINPUTS directory"
.PHONY : bib
bib :
	@cp imp.bib ${HOME}/Projects/TeX/bib/


info ::
	@echo "make clean"
.PHONY: clean
clean ::
	@/bin/rm -f *~
	@for d in * ; do \
	  if [ -d "$$d" ] ; then \
	    ( cd "$$d" ; \
	      if [ -f Makefile ] ; then \
	        echo "cleaning writeup/$$d" ; \
	        make clean ; fi \
	    ) ; \
	  fi ; \
	done
