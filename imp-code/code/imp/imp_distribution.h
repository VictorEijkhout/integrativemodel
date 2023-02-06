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
  coordinate<index_int,d> omega;
  std::array<
    std::vector< indexstructure<index_int,1> >,
    d> patches;
public:
  distribution( const coordinate<index_int,d>&, const decomposition<d>&,
		distribution_type=distribution_type::orthogonal );
public:
  const indexstructure<index_int,d>& local_domain() const;
protected:
  indexstructure<index_int,d> _local_domain;

#if 0
  /*
   * Numa stuff
   */
protected:
  indexstructure<index_int,d> numa_structure;
  auto get_numa_structure() { return numa_structure; };
  const auto& get_numa_structure() const { return numa_structure; };
public:
  coordinate<index_int,d> numa_first_index() const {
    return get_numa_structure()->first_index(); };
  auto numa_offset() const {
    auto numa_loc = get_numa_structure()->first_index_r()[0];
    auto global_loc = get_global_structure()->first_index_r()[0];
    return numa_loc -global_loc;
  };
  index_int numa_local_size() { return get_numa_structure()->volume(); }; //!< \todo remove
  index_int numa_size() { return get_numa_structure()->volume(); };
#endif
};
