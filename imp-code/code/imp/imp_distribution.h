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
  //! Make a domain from an indexstructure with inclusive upper bound
  domain( const indexstructure<index_int,d>& idx )
    : indexstructure<index_int,d>( idx ) {
    //std::cout << "domain from istruct " << idx.as_string() << '\n';
  };
  //! Make a domain from a indexstruct with inclusive upper bound
  domain( contiguous_indexstruct<index_int,d> ci )
    : indexstructure<index_int,d>::indexstructure<index_int,d>( ci ) {
    //std::cout << "domain from contiguous " << ci.as_string() << '\n';
  };
  //! Make a domain from a presumable exclusive upper bound
  domain( coordinate<index_int,d> c )
    : domain<d>( contiguous_indexstruct<index_int,d>( coordinate<index_int,d>(0),c-1 ) ) {
    //std::cout << "domain from coordinate " << c.as_string() << '\n';
  };
  /*! default constructor,
   * needed for the _local_domain member of the `distribution' class,
   * because that is constructed, not instantiated
   */
  domain()
    : indexstructure<index_int,d>() {};
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
  //! processor grid
  decomposition<d> my_decomposition;
  //! orthogonal product of extents in all dimensions
  std::array<
    std::vector< indexstructure<index_int,1> >, // vector of length #proc-in-dimension
    d> patches;
public:
  distribution( const domain<d>&, const decomposition<d>&,
		distribution_type=distribution_type::orthogonal );
  void assert_compatible_with( const distribution<d>& other ) const;
  bool compatible_with( const distribution<d>& other ) const;
  void assert_replicated() const;

  /*
   * Local and global domain
   */
protected:
  domain<d> _local_domain;
  domain<d> _global_domain;
public:
  const domain<d>& local_domain() const;
  const domain<d>& global_domain() const;

  // // new distribution by operating, only exists in derived classes. hm.
  // virtual distribution<d> operate( const ioperator<index_int,d>& ) const;
};
