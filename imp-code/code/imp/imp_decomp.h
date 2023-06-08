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

#include <array>
#include <vector>
#include <string>
#include <functional>

#include "utils.h"
#include "imp_entity.h"
#include "imp_env.h"
#include "imp_coord.h"
#include "indexstruct.hpp"

template<int d>
class distribution;

/*!
  A decomposition is a layout of all available processors
  in a d-dimensional grid.
  It contains (through inheritance) a vector of the local domains.
  For MPI that will be a single domain, for OpenMP all, because shared memory.

  \todo why a vector of coordinates? isn't having the corners enough?
 */
template<int d>
class decomposition {
protected:
  std::vector<coordinate<int,d>> local_procs;
public:
  //! No default constructor
  decomposition()=delete;
  //! Constructor from explicit endpoint coordinate
  decomposition( const coordinate<int,d> nd );
  //! Constructor from environment: uses the endpoint coordinate of the env
  decomposition( const environment& env );
private:
  //! A vector of the sizes in all the dimensions
  coordinate<int,d> _process_grid;
public:
  //! Getting the process grid itself is strictly a utility function
  const coordinate<int,d> &process_grid() const { return _process_grid; };
public:
  std::vector<index_int> split_points_d( const coordinate<index_int,d>& c,int id ) const;
  int linear_location_of( const coordinate<int,d>& ) const;

  std::function< coordinate<int,d>() > this_proc{
    [] () -> coordinate<int,d> { throw( "Function this_proc not defined" ); } };

  //! How many processors do we have in dimension `nd'?
  int size_of_dimension(int nd) const;
  //! Conversion from grid coordinate to linear numbering.
  int linearize( const coordinate<int,d> &p ) const;
  //! Conversion from linearly numbered process to coordinate in grid
  coordinate<int,d> coordinate_from_linear(int p) const;

  /*
   * Domain handling
   */
public:
  int local_volume() const;
  int global_volume() const;
  //! Get a domain by local number; see \ref get_local_domain_number for global for translation
  std::function<  coordinate<int,d>() > local_domain{
    [] () -> coordinate<int,d> { throw("no local domain function defined"); } };
  //! The local number of domains.
  int local_ndomains() const { return local_procs.size(); };
  int domain_local_number( const coordinate<int,d>& ) const;

  virtual std::string as_string() const;

  std::function< std::shared_ptr<distribution<d>>(index_int) > new_block_distribution;

  /*
   * Ranging
   */
protected:
  class iter {
  private:
    int count{0};
    decomposition<d>& decomp;
  public:
    iter( decomposition<d>&decomp, int c ) : decomp(decomp),count(c) {};
    void operator++();
    bool operator!=( const decomposition<d>::iter& ) const;
    bool operator==( const decomposition<d>::iter& ) const;
    coordinate<int,d> operator*() const;
  };
public:
  iter begin();
  iter end();
};
