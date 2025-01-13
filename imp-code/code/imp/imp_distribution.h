// -*- c++ -*-
/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2025
 ****
 **** imp_distribution.h: base header for distributions
 ****
 ****************************************************************/

#pragma once

#include "indexstruct.hpp"
#include "imp_decomp.h"

template<int d>
class domain : public indexstructure<index_int,d> {
public:
  //! Make a domain from an indexstructure
  domain( const indexstructure<index_int,d>& idx )
    : indexstructure<index_int,d>( idx ) {
  };
  //! Make a domain from a contiguous indexstruct
  domain( contiguous_indexstruct<index_int,d> ci )
    : indexstructure<index_int,d>::indexstructure<index_int,d>( ci ) {
  };
  //! Make a domain from a presumable exclusive upper bound
  domain( coordinate<index_int,d> c )
    : domain<d>( contiguous_indexstruct<index_int,d>( coordinate<index_int,d>(0),c-1 ) ) {
  };
  // /*! default constructor,
  //  * needed for the _local_domain member of the `distribution' class,
  //  * because that is constructed, not instantiated
  //  */
  // domain()
  //   : indexstructure<index_int,d>() {
  // };
};

//! Different types of distributions
enum class distribution_type : int { orthogonal,replicated };

/*! Distribution class: a domain over a process grid.
 *
 * For now only implementation:
 * d-dimensional distribution as orthogonal product of 1-d block distributions
 */
template<int d>
class distribution {
protected:
  static inline int distribution_number{0};
  int my_distribution_number;
  distribution_type my_distribution_type;
  //! processor grid
  decomposition<d> my_decomposition;
  //! orthogonal product of extents in all dimensions
  std::array<
    std::vector< indexstructure<index_int,1> >, // vector of length #proc-in-dimension
    d> patches;
public:
  distribution( const domain<d>&, const decomposition<d>&,
		distribution_type=distribution_type::orthogonal );
  const decomposition<d> get_decomposition() const;
  void assert_compatible_with( const distribution<d>& other ) const;
  bool compatible_with( const distribution<d>& other ) const;
  void assert_replicated() const;

  /*
   * Local and global domain
   */
protected:
  domain<d> _global_domain;
  // this is set mode-specific
  std::vector<domain<d>> _local_domains,_all_domains;
public:
  const domain<d>& local_domain() const;
  const domain<d>& local_domain(const coordinate<int,d>& p) const;
  const std::vector<domain<d>>& local_domains() const;
  const std::vector<domain<d>>& all_domains() const;
  const domain<d>& global_domain() const;

  /*
   * Polymorphism
   */
  std::function < index_int( const coordinate<int,d> &p) >
    location_of_first_index {
      [] ( const coordinate<int,d> &p) -> index_int {
	throw(std::string("not implemented: distribution::location_of_first_index")); } };
  std::function< distribution<d> ( const ioperator<index_int,d>& ) >
    operate {
      [] ( const ioperator<index_int,d>& ) -> distribution<d> {
	throw(std::string("not implemented: distribution::operate")); } };
};
