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

class message;
template<int d>
class distribution;
class task;

/*!
  A decomposition is a layout of all available processors
  in a d-dimensional grid.
  It contains (through inheritance) a vector of the local domains.
  For MPI that will be a single domain, for OpenMP all, because shared memory.

  This class is virtual because of the `this_proc' method.
 */
template<int d>
class decomposition : public std::vector<coordinate<int,d>> {
public:
  decomposition() {}; //!< default constructor
  //! Constructor from explicit endpoint coordinate
  decomposition( const coordinate<int,d> nd );
  //! Constructor from environment: uses the endpoint coordinate of the env
  decomposition( const environment& env );
private:
  //! A vector of the sizes in all the dimensions
  coordinate<int,d> _domain_layout;
  coordinate<int,d> closecorner,farcorner;
public:
  std::vector<index_int> split_points_d( const coordinate<index_int,d>& c,int id ) const;
  //  void set_corners();
  const coordinate<int,d> &domain_layout() const { return _domain_layout; };
  const coordinate<int,d> &get_origin_processor() const;
  const coordinate<int,d> &get_farpoint_processor() const;
  int linear_location_of( const coordinate<int,d>& ) const;

  // virtual const coordinate<int,d>& this_proc() const = 0;
  std::function< coordinate<int,d>() > this_proc{
    [] () -> coordinate<int,d> { throw( "Function this_proc not defined" ); } };

  //! How many processors do we have in dimension `nd'?
  int size_of_dimension(int nd) const;
  //! \todo do we really need this?
  //  auto get_global_domain_descriptor() { return domain_layout.data(); };
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
  // //! Return the domains object, for 1d only
  const std::vector< coordinate<int,d> > get_domains() const { return *this; };
  // //! Get a domain by local number; see \ref get_local_domain_number for global for translation
  std::function<  coordinate<int,d>() > local_domain{
    [] () -> coordinate<int,d> { throw("no local domain function defined"); } };
  const coordinate<int,d> &first_local_domain() const;
  const coordinate<int,d> &last_local_domain() const;
  //! The local number of domains.
  int local_ndomains() const { return this->size(); };
  int domain_local_number( const coordinate<int,d>& ) const;

  virtual std::string as_string() const;

  std::function< std::shared_ptr<distribution<d>>(index_int) > new_block_distribution;

protected:
  bool range_linear{true},range_twoside{false};
public:
  void set_range_twoside() { 
    range_linear = false; range_twoside = true;
  };

protected:
  std::shared_ptr<decomposition> embedded_decomposition{nullptr};
  void copy_embedded_decomposition( decomposition &other );
public:
  const decomposition &get_embedded_decomposition() const;

//   /*
//    * Ranging
//    */
// protected:
//   int iterate_count{0};
//   coordinate<int,d> cur_coord;
// public:
//   decomposition &begin();
//   decomposition &end();
//   void operator++();
//   bool operator!=( const decomposition& ) const;
//   bool operator==( const decomposition& ) const;
//   coordinate<int,d> &operator*();

};


#if 0

/****
 **** Architecture
 ****/

template<int d>
class architecture {
protected:
  coordinate<int,d> endpoint;
public:
  architecture( int np )
    : endpoint( coordinate<int,d>(np) ) {
  };
};

/*!
  Collective strategy

  - undefined : set at architecture creation
  - all_ptp : naive by all sends and received
  - group : two level grouping
  - tree : recursive grouping by two
  - MPI : (not implemented) using MPI collectives
 */
enum class collective_strategy{ UNDEFINED,ALL_PTP,GROUP,RECURSIVE,MPI };

/*!
  Definition of the synchronization protocol. 
  Overlap with \ref architecture_type needs to be figured out.
  Observe partial ordering: MPIplusCrap >= MPI.
 */
enum class protocol_type { UNDEFINED,OPENMP,OPENMPinMPI,MPI,MPIRMA,MPIplusOMP,MPItimesOMP };
std::string protocol_as_string(protocol_type p);

/*!
  Architecture types. We need to edit this list (and edit the definition of
  \ref parallel_indexstruct::parallel_indexstruct) every time we add one.

  - SHARED : OpenMP
  - SPMD : MPI
  - ISLANDS : Product?
  - ??? : Hybrid?
  - ??? : Accelerator?
 */
enum class architecture_type { UNDEFINED,SHARED,SPMD,ISLANDS };
std::string architecture_type_as_string( architecture_type type );

/*!  
  A base class for architecture data. All accessor routines throw
  an exception by default; they are overridden in the mode-specific
  classes. Interestingly, there is no data in this class; we leave
  that again to the modes.

  We probably need to add access functions here if we introduce new modes.

  \todo reinstate disabled collective strategy handling
  \todo install lambda for getting environment
 */
class message;
class architecture : public entity {
protected:
public: // data
  int beta_has_local_addres_space{-1}; //!< \todo why is this public data?
public: // methods
  //! Default constructor
  architecture();
  //! architecture constructor will mostly be called through derived architectures.
  architecture(int m,int n) : architecture(n) { arch_procid = m; };
  architecture(int n) { // : entity(entity_cookie::ARCHITECTURE) {
    arch_nprocs = n; }; //set_name("some architecture"); };
  //! A mode-dependent switch to turn on tricky optimizations
  virtual void set_power_mode() {};

  /*
   * Type handling
   */
public: //! make this protected again
  architecture_type type{architecture_type::UNDEFINED};
  architecture *embedded_architecture{nullptr};
public:
  const architecture_type &get_architecture_type() const { return type; };
  int has_type(architecture_type t) const { return get_architecture_type()==t; };
  architecture *get_embedded_architecture() const {
    if (embedded_architecture==nullptr)
      throw(std::string("embedded architecture is not set"));
    return embedded_architecture; };

  /*
   * Over decomposition
   */
protected:
  int over_factor{1};
public:
  void set_over_factor(int o) { over_factor = o; };
  int get_over_factor() const { return over_factor; };

  /*
   * Protocols
   */
public: //! make this protected again
  protocol_type protocol{protocol_type::UNDEFINED};
public:
  protocol_type get_protocol() const { return protocol; };
  int get_is_mpi_based() const { return protocol>=protocol_type::MPI; };
  void set_protocol_is_embedded() { protocol = protocol_type::OPENMPinMPI; };
  int get_protocol_is_embedded() const { return protocol==protocol_type::OPENMPinMPI; };
protected:
  bool split_execution{false};
public:
  void set_split_execution(bool split=true) { split_execution = split; };
  auto get_split_execution() const { return split_execution; };

  /*
   * Processor handling
   */
public: //! make this protected again
  int arch_procid{-1},arch_nprocs{-1};
public:
  int nprocs() const {
    if (arch_nprocs<0) throw(std::string("Number of processors uninitialized"));
    return arch_nprocs; };
  //! Processor count of the embedded architecture
  int embedded_nprocs() const { return get_embedded_architecture()->nprocs(); };
  //! Total number of processors over top level plus embedded
  int product_nprocs() const { return nprocs()*embedded_nprocs(); };
  //! MPI has local identification, OMP doesn't
  std::function< int(void) > mytid{
    [] (void) -> int { throw(std::string("no mytid in general")); } };
  virtual int is_first_proc(int p) const { return p==0; };
  virtual int is_last_proc(int p)  const { return p==arch_nprocs-1; };
protected:
  coordinate coordinates;
public:
  coordinate get_proc_origin(int d) const; // far corner of the processor grid
  coordinate get_proc_endpoint(int d) const; // far corner of the processor grid
  coordinate get_proc_layout(int d) const; // far corner of the processor grid
  coordinate get_coordinates() const {
    if (coordinates==nullptr) throw(std::string("No coordinates translation set"));
    return coordinates; };
protected:
  int random_sourcing{0}; //!< Random resolution of MPI redundancy.
  int rma{0}; //!< Use MPI one-sided.
public:
  void set_random_sourcing() { random_sourcing = 1; };
  int has_random_sourcing() { return random_sourcing; };
  void set_use_rma() { rma = 1; };
  int get_use_rma() { return rma; };

  /*
   * Various properties
   */
protected:
  int can_embed_in_beta{0},can_message_overlap{0};
public:
  //! Declare that betas are in global space.
  void set_can_embed_in_beta() { set_can_embed_in_beta(1); };
  //! Set the embedding explicitly. Good for the unittests.
  void set_can_embed_in_beta(int can) { can_embed_in_beta = can; };
  int get_can_embed_in_beta() const { return can_embed_in_beta; };
  void set_can_message_overlap() { can_message_overlap = 1; };
  int get_can_message_overlap() const { return can_message_overlap; };

  /*
   * Collectives
   */
protected:
  collective_strategy strategy{collective_strategy::GROUP};
public:
  collective_strategy get_collective_strategy() const { return strategy; };
  int has_collective_strategy(collective_strategy s) const { return strategy==s; };
  void set_collective_strategy(int s);
  void set_collective_strategy(collective_strategy s);
  void set_collective_strategy_ptp() { strategy = collective_strategy::ALL_PTP; };
  void set_collective_strategy_group() { strategy = collective_strategy::GROUP; };
  void set_collective_strategy_recursive() { strategy = collective_strategy::RECURSIVE; };
  std::string strategy_as_string() const {
    switch (strategy) {
    case collective_strategy::ALL_PTP : return std::string("point-to-point"); break;
    case collective_strategy::GROUP : return std::string("grouped"); break;
    case collective_strategy::RECURSIVE : return std::string("treewise"); break;
    case collective_strategy::MPI : return std::string("using mpi"); break;
    default : return std::string("undefined"); }
  };

  /*
   * Timing
   */
  std::chrono::system_clock::time_point unsynchronized_timer() {
    return std::chrono::system_clock::now(); };
  //! \todo reinstall the barrier
  std::chrono::system_clock::time_point synchronized_timer() {
    //barrier();
    return unsynchronized_timer(); };
  
  /*
   * Tracing and output
   */
protected:
public:
  std::function< void(std::string) > print_trace{ [] (std::string c) -> void { return; } };
  void set_stdio_print() { print_trace = [] (std::string c) { fmt::print("{}\n",c); }; };
  virtual std::string summary();

  /*
   * Distribution factory
   */
  virtual std::shared_ptr<distribution> new_scalar_distribution() ;
  // {
  //   throw(std::string("Error: can not call base case")); };

  // std::function< void(architecture&,std::shared_ptr<message>,
  // 		      std::string& /* char*,int */ ) > message_as_buffer {
  //   [] (architecture &a,std::shared_ptr<message>m,
  // 	// char *buf,int len) -> void {
  // 	std::string &b) -> void {
  //     throw(std::string("message_as_buffer not set")); }
  // };
  
  // I/O
  virtual std::string as_string() const;
};

#endif
