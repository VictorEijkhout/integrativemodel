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

#include "indexstruct.hpp"
#include "imp_decomp.h"

template<int d>
class domain : public indexstructure<index_int,d> {
public:
  domain( contiguous_indexstruct<index_int,d> ci )
    : indexstructure<index_int,d>::indexstructure<index_int,d>( ci ) {};
  domain( coordinate<index_int,d> c )
    : domain<d>( contiguous_indexstruct<index_int,d>( coordinate<index_int,d>(0),c ) ) {};
  /*! default constructor,
   * needed for the _local_domain member of the `distribution' class,
   * because that is constructed, not instantiated
   */
  domain()
    : indexstructure<index_int,d>( empty_indexstruct<index_int,d>() ) {}
};

//! Different types of distributions
enum class distribution_type : int { orthogonal,replicated };

/*! Distribution class.
 * For now only implementation:
 * d-dimensional distribution as orthogonal product of 1-d block distributions
 */
template<int d>
class distribution {
protected:
  static inline int distribution_number{0};
  int my_distribution_number;
  distribution_type my_distribution_type;
  //! extent of the domain
  coordinate<index_int,d> omega;
  //! processor grid
  decomposition<d> my_decomposition;
  //! orthogonal product of extents in all dimensions
  std::array<
    std::vector< indexstructure<index_int,1> >, // vector of length #proc-in-dimension
    d> patches;
public:
  distribution( const coordinate<index_int,d>&, const decomposition<d>&,
		distribution_type=distribution_type::orthogonal );
  void assert_compatible_with( const distribution<d>& other ) const;
  bool compatible_with( const distribution<d>& other ) const;
  void assert_replicated() const;

protected:
  //indexstructure<index_int,d> _local_domain;
  domain<d> _local_domain;
public:
  //const indexstructure<index_int,d>& local_domain() const;
  const domain<d>& local_domain() const;
protected:
  //indexstructure<index_int,d> _global_domain;
  domain<d> _global_domain;
public:
  //const indexstructure<index_int,d>& global_domain() const;
  const domain<d>& global_domain() const;

  // // new distribution by operating
  // virtual distribution<d> operate( const ioperator<index_int,d>& ) const;
};
