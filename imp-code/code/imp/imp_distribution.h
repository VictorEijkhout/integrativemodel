// -*- c++ -*-
/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2023
 ****
 **** imp_decomp.h: Header file for the decomposition base classes
 ****
 ****************************************************************/

#pragma once

#include "imp_decomp.h"

template<int d>
class domain : indexstructure<index_int,d> {
public:
  domain( contiguous_indexstruct<index_int,d> ci )
    : indexstructure<index_int,d>::indexstructure<index_int,d>( ci ) {};
  domain( coordinate<index_int,d> c )
    : domain( contiguous_indexstruct<index_int,d>( coordinate<index_int,d>(0),c ) ) {};
};

//! Different types of distributions
enum class distribution_type : int { orthogonal,replicated };

template<int d>
class distribution {
protected:
  static inline int distribution_number{0};
  int my_distribution_number;
  //! extent of the domain
  coordinate<index_int,d> omega;
  //! orthogonal product of extents in all dimensions
  std::array<
    std::vector< indexstructure<index_int,1> >, // vector of length #proc-in-dimension
    d> patches;
public:
  distribution( const coordinate<index_int,d>&, const decomposition<d>&,
		distribution_type=distribution_type::orthogonal );
public:
  const indexstructure<index_int,d>& local_domain() const;
  bool compatible_with( const distribution<d>& other ) const;
  void throw_incompatible_with( const distribution<d>& other ) const;
protected:
  indexstructure<index_int,d> _local_domain;
};
