/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2025
 ****
 **** seq_decomp.cxx: Implementations of the SEQ decompositions classes
 ****
 ****************************************************************/

#include "seq_env.h"
#include "seq_decomp.h"

using fmt::print;

//! Multi-d decomposition from default grid from environment
template<int d>
seq_decomposition<d>::seq_decomposition( const seq_environment& env )
  : seq_decomposition<d>( endpoint<int,d>(env.nprocs()) ) {
};

//! Multi-d decomposition from explicit processor grid layout
template<int d>
seq_decomposition<d>::seq_decomposition
    ( const coordinate<int,d> &grid )
      : decomposition<d>(grid) {
  for ( int procid=0; procid<grid.span(); procid++ ) {
    decomposition<d>::local_procs.push_back
      ( decomposition<d>::domain_layout().location_of_linear(procid) );
  }
};

/*
 * Stuff related to this SEQ process
 */

//! Our process rank. \todo derive this from the coordinate?
template<int d>
int seq_decomposition<d>::procno() const {
  throw( "no procno for seq" );
};

template<int d>
std::string seq_decomposition<d>::as_string() const {
  return "seqdecomp"; // fmt::format("SEQ decomposition <<{}>>",decomposition::as_string());
};

template class seq_decomposition<1>;
template class seq_decomposition<2>;
template class seq_decomposition<3>;
